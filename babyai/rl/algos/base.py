from abc import ABC, abstractmethod
import torch
import numpy
import random

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel0, acmodel1, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, use_comm, n, aux_info):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses

        """
        # Store parameters

        self.env = ParallelEnv(envs)
        
        self.acmodel0 = acmodel0
        self.acmodel0.train()
        
        self.acmodel1 = acmodel1
        self.acmodel1.train()
        
        self.num_frames_per_proc = num_frames_per_proc
        self.discount            = discount
        self.lr                  = lr
        self.gae_lambda          = gae_lambda
        self.entropy_coef        = entropy_coef
        self.value_loss_coef     = value_loss_coef
        self.max_grad_norm       = max_grad_norm
        self.recurrence          = recurrence
        self.preprocess_obss     = preprocess_obss or default_preprocess_obss
        self.reshape_reward      = reshape_reward
        self.use_comm            = use_comm
        self.n                   = n
        self.aux_info            = aux_info

        # Store helpers values

        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs  = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs


        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)
        
        self.step_count  = torch.zeros(shape[1], device=self.device, dtype=torch.uint8)
        self.comm        = torch.zeros(shape[1], device=self.device, dtype=torch.uint8)
        self.comms       = torch.zeros(*shape,   device=self.device, dtype=torch.uint8)
        self.mask        = torch.ones(shape[1],  device=self.device)
        self.masks       = torch.zeros(*shape,   device=self.device)
        self.actions     = torch.zeros(*shape,   device=self.device, dtype=torch.int)
        self.values0     = torch.zeros(*shape,   device=self.device)
        self.values1     = torch.zeros(*shape,   device=self.device)
        self.rewards     = torch.zeros(*shape,   device=self.device)
        self.advantages0 = torch.zeros(*shape,   device=self.device)
        self.advantages1 = torch.zeros(*shape,   device=self.device)
        self.log_probs0  = torch.zeros(*shape,   device=self.device)
        self.log_probs1  = torch.zeros(*shape,   device=self.device)

        self.globs, self.obs = self.env.reset()
        self.globss          = [None]*(shape[0])
        self.obss            = [None]*(shape[0])

        self.memory0   = torch.zeros(shape[1], self.acmodel0.memory_size, device=self.device)
        self.memories0 = torch.zeros(*shape,   self.acmodel0.memory_size, device=self.device)
        
        self.memory1   = torch.zeros(shape[1], self.acmodel1.memory_size, device=self.device)
        self.memories1 = torch.zeros(*shape,   self.acmodel1.memory_size, device=self.device)
        
        self.msg  = torch.zeros(shape[1], self.acmodel0.max_len_msg, self.acmodel0.num_symbols, device=self.device)
        self.msgs = torch.zeros(*shape,   self.acmodel0.max_len_msg, self.acmodel0.num_symbols, device=self.device)

        if self.aux_info:
            self.aux_info_collector = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values

        self.log_episode_return          = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames      = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter    = 0
        self.log_return          = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames      = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """
        value0  = torch.zeros(self.num_procs, device=self.device)
        memory0 = torch.zeros(self.num_procs, self.acmodel0.memory_size, device=self.device)
        
        action  = torch.zeros(self.num_procs, device=self.device, dtype=torch.long)
        value1  = torch.zeros(self.num_procs, device=self.device)
        memory1 = torch.zeros(self.num_procs, self.acmodel1.memory_size, device=self.device)
        
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            preprocessed_globs = self.preprocess_obss(self.globs, device=self.device)
            preprocessed_obs   = self.preprocess_obss(self.obs,   device=self.device)
            
            with torch.no_grad():
                
                if self.use_comm:
                    self.comm = self.step_count % self.n == 0
                
                    if torch.any(self.comm):
                        # blind the scout to instructions
                        preprocessed_globs.instr[self.comm] *= 0
                        
                        model_results0  = self.acmodel0(preprocessed_globs[  self.comm], self.memory0[    self.comm] * self.mask[    self.comm].unsqueeze(1))
                        
                        self.msg[self.comm] = model_results0['message']
                        
                        model_results1A = self.acmodel1(preprocessed_obs[    self.comm], self.memory1[    self.comm] * self.mask[    self.comm].unsqueeze(1), msg=(self.msg[    self.comm]))
                    
                    if torch.any(1 - self.comm):
                        model_results1B = self.acmodel1(preprocessed_obs[1 - self.comm], self.memory1[1 - self.comm] * self.mask[1 - self.comm].unsqueeze(1), msg=(self.msg[1 - self.comm]))
                else:
                    model_results1B = self.acmodel1(preprocessed_obs, self.memory1 * self.mask.unsqueeze(1))
                
                if torch.any(self.comm):
                    dists_speaker      = model_results0['dists_speaker']
                    value0[self.comm]  = model_results0['value']
                    memory0[self.comm] = model_results0['memory']
                    
                    distA              = model_results1A['dist']
                    value1[self.comm]  = model_results1A['value']
                    memory1[self.comm] = model_results1A['memory']
                    
                if torch.any(1 - self.comm):
                    distB                  = model_results1B['dist']
                    value1[1 - self.comm]  = model_results1B['value']
                    memory1[1 - self.comm] = model_results1B['memory']
                    
            if torch.any(self.comm):
                actionA           = distA.sample()
                action[self.comm] = actionA
            
            if torch.any(1 - self.comm):
                actionB               = distB.sample()
                action[1 - self.comm] = actionB
            
            globs, obs, reward, done, step_count, env_info = self.env.step(action.cpu().numpy())
            
            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
            
            # Update experiences values

            self.globss[i] = self.globs
            self.globs     = globs
            
            self.obss[i] = self.obs
            self.obs     = obs

            self.memories0[i] = self.memory0
            self.memory0      = memory0

            self.memories1[i] = self.memory1
            self.memory1      = memory1

            self.masks[i]   = self.mask
            self.mask       = 1 - torch.tensor(done,       device=self.device, dtype=torch.float)
            self.step_count =     torch.tensor(step_count, device=self.device, dtype=torch.float) * self.mask
            self.actions[i] = action
            self.values0[i] = value0
            self.values1[i] = value1
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            if torch.any(self.comm):
                self.log_probs0[i,     self.comm] = self.acmodel0.speaker_log_prob(dists_speaker, self.msg[self.comm])
                self.log_probs1[i,     self.comm] = distA.log_prob(actionA)
            if torch.any(1 - self.comm):
                self.log_probs1[i, 1 - self.comm] = distB.log_prob(actionB)
            self.comms[i] = self.comm

            self.msgs[i] = self.msg

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

            # Update log values
            
            self.log_episode_return          += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames      += torch.ones(self.num_procs, device=self.device)
            
            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return          *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames      *= self.mask

        # Add advantage and return to experiences

        preprocessed_globs = self.preprocess_obss(self.globs, device=self.device)
        preprocessed_obs   = self.preprocess_obss(self.obs,   device=self.device)
        
        with torch.no_grad():
            if self.use_comm:
                self.comm = self.step_count % self.n == 0
                
                next_value = torch.zeros(self.num_procs, device=self.device)
                
                if torch.any(self.comm):
                    # blind the scout to instructions
                    preprocessed_globs.instr[self.comm] *= 0
                    
                    self.msg[self.comm] = self.acmodel0(preprocessed_globs[self.comm],    self.memory0[self.comm] * self.mask[self.comm].unsqueeze(1))['message']
                    
                    next_value[    self.comm] = self.acmodel1(preprocessed_obs[    self.comm], self.memory1[    self.comm] * self.mask[    self.comm].unsqueeze(1), msg=(self.msg[    self.comm]))['value']
                
                if torch.any(1 - self.comm):
                    next_value[1 - self.comm] = self.acmodel1(preprocessed_obs[1 - self.comm], self.memory1[1 - self.comm] * self.mask[1 - self.comm].unsqueeze(1), msg=(self.msg[1 - self.comm]))['value']
            else:
                next_value = self.acmodel1(preprocessed_obs, self.memory1 * self.mask.unsqueeze(1))['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask      = self.masks[i+1]       if i < self.num_frames_per_proc - 1 else self.mask
            next_value     = self.values1[i+1]     if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages1[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta0              = self.rewards[i] + self.discount * next_value * next_mask - self.values0[i]
            delta1              = self.rewards[i] + self.discount * next_value * next_mask - self.values1[i]
            self.advantages0[i] = delta0 + self.discount * self.gae_lambda * next_advantage * next_mask
            self.advantages1[i] = delta1 + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps       = DictList()
        exps.globs = [self.globss[i][j]
                       for j in range(self.num_procs)
                       for i in range(self.num_frames_per_proc)]
        exps.obs   = [self.obss[i][j]
                       for j in range(self.num_procs)
                       for i in range(self.num_frames_per_proc)]
        
        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory0   = self.memories0.transpose(0, 1).reshape(-1, *self.memories0.shape[2:])
        exps.memory1   = self.memories1.transpose(0, 1).reshape(-1, *self.memories1.shape[2:])
        
        exps.message   = self.msgs.transpose(0, 1).reshape(-1, *self.msgs.shape[2:])
        
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.comm       = self.comms.transpose(0, 1).reshape(-1)
        exps.action     = self.actions.transpose(0, 1).reshape(-1)
        exps.value0     = self.values0.transpose(0, 1).reshape(-1)
        exps.value1     = self.values1.transpose(0, 1).reshape(-1)
        exps.reward     = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage0 = self.advantages0.transpose(0, 1).reshape(-1)
        exps.advantage1 = self.advantages1.transpose(0, 1).reshape(-1)
        exps.returnn0   = exps.value0 + exps.advantage0
        exps.returnn1   = exps.value1 + exps.advantage1
        exps.log_prob0  = self.log_probs0.transpose(0, 1).reshape(-1)
        exps.log_prob1  = self.log_probs1.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Preprocess experiences

        exps.globs = self.preprocess_obss(exps.globs, device=self.device)
        exps.obs   = self.preprocess_obss(exps.obs,   device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode":          self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode":      self.log_num_frames[-keep:],
            "num_frames":                  self.num_frames,
            "episodes_done":               self.log_done_counter,
        }

        self.log_done_counter    = 0
        self.log_return          = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames      = self.log_num_frames[-self.num_procs:]

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass
