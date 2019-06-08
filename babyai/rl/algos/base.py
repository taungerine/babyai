from abc import ABC, abstractmethod
import torch
import numpy
import random

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel0, acmodel1, n, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, use_comm, ignorant_scout, aux_info):
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
        self.ignorant_scout      = ignorant_scout
        self.n                   = n
        self.aux_info            = aux_info

        # Store helpers values

        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs   = len(envs)
        self.num_frames  = self.num_frames_per_proc * self.num_procs
        self.num_frames0 = 0
        self.num_frames1 = 0


        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)
        
        self.scouting   = torch.zeros(shape[1], device=self.device, dtype=torch.uint8)
        self.scoutings  = torch.zeros(*shape,   device=self.device, dtype=torch.uint8)
        self.mask       = torch.ones(shape[1],  device=self.device)
        self.masks      = torch.zeros(*shape,   device=self.device)
        self.mask2      = torch.ones(shape[1],  device=self.device)
        self.masks2     = torch.zeros(*shape,   device=self.device)
        self.mask3      = torch.ones(shape[1],  device=self.device)
        self.masks3     = torch.zeros(*shape,   device=self.device)
        self.actions    = torch.zeros(*shape,   device=self.device, dtype=torch.int)
        self.values     = torch.zeros(*shape,   device=self.device)
        self.rewards    = torch.zeros(*shape,   device=self.device)
        self.advantages = torch.zeros(*shape,   device=self.device)
        self.log_probs  = torch.zeros(*shape,   device=self.device)

        self.globs, self.obs, reward, done, step_count = self.env.reset(self.scouting.cpu().numpy())
        self.globss          = [None]*(shape[0])
        self.obss            = [None]*(shape[0])
        
        self.step_count = torch.tensor(step_count, device=self.device)

        self.memory0   = torch.zeros(shape[1], self.acmodel0.memory_size, device=self.device)
        self.memories0 = torch.zeros(*shape,   self.acmodel0.memory_size, device=self.device)
        
        self.memory1   = torch.zeros(shape[1], self.acmodel1.memory_size, device=self.device)
        self.memories1 = torch.zeros(*shape,   self.acmodel1.memory_size, device=self.device)
        
        self.msg  = torch.zeros(shape[1], self.acmodel0.max_len_msg, self.acmodel0.num_symbols, device=self.device)
        self.msgs = torch.zeros(*shape,   self.acmodel0.max_len_msg, self.acmodel0.num_symbols, device=self.device)
        
        self.msgs_out = torch.zeros(*shape, self.acmodel0.max_len_msg, self.acmodel0.num_symbols, device=self.device)

        if self.aux_info:
            self.aux_info_collector = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values

        self.log_episode_return          = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames0     = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames1     = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter    = 0
        self.log_return          = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames0     = [0] * self.num_procs
        self.log_num_frames1     = [0] * self.num_procs

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
        action  = torch.zeros(self.num_procs, device=self.device, dtype=torch.long)
        value   = torch.zeros(self.num_procs, device=self.device)
        memory0 = torch.zeros(self.num_procs, self.acmodel0.memory_size, device=self.device)
        memory1 = torch.zeros(self.num_procs, self.acmodel1.memory_size, device=self.device)
        msg     = torch.zeros(self.num_procs, self.acmodel0.max_len_msg, self.acmodel0.num_symbols, device=self.device)
        
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            preprocessed_globs = self.preprocess_obss(self.globs, device=self.device)
            preprocessed_obs   = self.preprocess_obss(self.obs,   device=self.device)
            
            with torch.no_grad():
                # scout if step_count % n == 0, except if previous step was already scouting
                self.scouting = (self.step_count % self.n == 0) * (1 - self.scouting)
                
                if torch.any(self.scouting):
                    if self.ignorant_scout:
                        # blind the scout to instructions
                        preprocessed_globs.instr[self.scouting] *= 0
                    
                    model_results0 = self.acmodel0(preprocessed_globs[    self.scouting], self.memory0[    self.scouting] * self.mask2[    self.scouting].unsqueeze(1))
                
                if torch.any(1 - self.scouting):
                    
                    if self.use_comm:
                        model_results1 = self.acmodel1(preprocessed_obs[1 - self.scouting], self.memory1[1 - self.scouting] * self.mask3[1 - self.scouting].unsqueeze(1), msg=self.msg[1 - self.scouting])
                    else:
                        model_results1 = self.acmodel1(preprocessed_obs[1 - self.scouting], self.memory1[1 - self.scouting] * self.mask3[1 - self.scouting].unsqueeze(1))
                
                if torch.any(self.scouting):
                    dist0                  = model_results0['dist']
                    value[self.scouting]   = model_results0['value']
                    memory0[self.scouting] = model_results0['memory']
                    msg[self.scouting]     = model_results0['message']
                    dists_speaker          = model_results0['dists_speaker']
                    
                if torch.any(1 - self.scouting):
                    dist1                      = model_results1['dist']
                    value[1 - self.scouting]   = model_results1['value']
                    memory1[1 - self.scouting] = model_results1['memory']
                    
            if torch.any(self.scouting):
                action0               = dist0.sample()
                action[self.scouting] = action0
            
            if torch.any(1 - self.scouting):
                action1                   = dist1.sample()
                action[1 - self.scouting] = action1
            
            globs, obs, reward, done, step_count = self.env.step(action.cpu().numpy(), self.scouting.cpu().numpy())
            
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
            self.masks2[i]  = self.mask2
            self.masks3[i]  = self.mask3
            self.mask       = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.mask3      = 1 - torch.tensor(done, device=self.device, dtype=torch.float) * (1 - self.scouting).float()
            self.mask3[self.scouting] *= self.mask2[self.scouting]
            self.mask2      = 1 - torch.tensor(done, device=self.device, dtype=torch.float) * (1 - self.scouting).float()
            self.step_count =     torch.tensor(step_count, device=self.device)
            self.actions[i] = action
            self.values[i]  = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            if torch.any(self.scouting):
                self.log_probs[i, self.scouting]     = self.acmodel0.speaker_log_prob(dists_speaker, msg[self.scouting])
            if torch.any(1 - self.scouting):
                self.log_probs[i, 1 - self.scouting] = dist1.log_prob(action1)
            self.scoutings[i] = self.scouting

            self.msgs[i]                 = self.msg
            self.msg[self.scouting]      = msg[self.scouting]
            #self.msg[1 - self.scouting]  = self.msg[1 - self.scouting] # repeat scout's message
            
            self.msgs_out[i] = msg

            # Update log values
            
            self.log_episode_return          += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames0     += self.scouting.float()
            self.log_episode_num_frames1     += torch.ones(self.num_procs, device=self.device) - self.scouting.float()
            
            self.num_frames0 +=     (self.scouting).sum().item()
            self.num_frames1 += (1 - self.scouting).sum().item()
            
            for i, done_ in enumerate(done):
                if done_ and not self.scouting[i]:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames0.append(self.log_episode_num_frames0[i].item())
                    self.log_num_frames1.append(self.log_episode_num_frames1[i].item())

            self.log_episode_return          *= self.mask2
            self.log_episode_reshaped_return *= self.mask2
            self.log_episode_num_frames0     *= self.mask2
            self.log_episode_num_frames1     *= self.mask2

        # Add advantage and return to experiences

        preprocessed_globs = self.preprocess_obss(self.globs, device=self.device)
        preprocessed_obs   = self.preprocess_obss(self.obs,   device=self.device)
        
        with torch.no_grad():
            next_value = torch.zeros(self.num_procs, device=self.device)
            
            # scout if step_count % n == 0, except if previous step was already scouting
            self.scouting = (self.step_count % self.n == 0) * (1 - self.scouting)
            
            if torch.any(self.scouting):
                if self.ignorant_scout:
                    # blind the scout to instructions
                    preprocessed_globs.instr[self.scouting] *= 0
                
                self.msg[self.scouting] = self.acmodel0(preprocessed_globs[self.scouting], self.memory0[self.scouting] * self.mask2[self.scouting].unsqueeze(1))['message']
                
            if self.use_comm:
                next_value = self.acmodel1(preprocessed_obs, self.memory1 * self.mask3.unsqueeze(1), msg=self.msg)['value']
            else:
                next_value = self.acmodel1(preprocessed_obs, self.memory1 * self.mask3.unsqueeze(1))['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask      = self.masks2[i+1]     if i < self.num_frames_per_proc - 1 else self.mask2
            next_value     = self.values[i+1]     if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0
            
            delta              = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
            
            if i < self.num_frames_per_proc - 2:
                if torch.any(self.scoutings[i+1]):
                    next_value = self.values[    i+2][self.scoutings[i+1]]
                    delta              = self.rewards[i][self.scoutings[i+1]] + self.discount * next_value * next_mask[self.scoutings[i+1]] - self.values[i][self.scoutings[i+1]]
                    self.advantages[i][self.scoutings[i+1]] = delta + self.discount * self.gae_lambda * self.advantages[i+2][self.scoutings[i+1]] * next_mask[self.scoutings[i+1]]

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
        exps.memory0     = self.memories0.transpose(0, 1).reshape(-1, *self.memories0.shape[2:])

        exps.memory1     = self.memories1.transpose(0, 1).reshape(-1, *self.memories1.shape[2:])
        
        exps.message     = self.msgs.transpose(0, 1).reshape(-1, *self.msgs.shape[2:])
        
        exps.message_out = self.msgs_out.transpose(0, 1).reshape(-1, *self.msgs.shape[2:])
        
        # T x P -> P x T -> (P * T) x 1
        exps.mask2 = self.masks2.transpose(0, 1).reshape(-1).unsqueeze(1)
        exps.mask3 = self.masks3.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.scouting         = self.scoutings.transpose(0, 1).reshape(-1)
        exps.action           = self.actions.transpose(0, 1).reshape(-1)
        exps.value            = self.values.transpose(0, 1).reshape(-1)
        exps.reward           = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage        = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn          = exps.value + exps.advantage
        exps.log_prob         = self.log_probs.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Preprocess experiences

        exps.globs = self.preprocess_obss(exps.globs, device=self.device)
        exps.obs   = self.preprocess_obss(exps.obs,   device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log0 = {
            "return_per_episode":          self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode":      self.log_num_frames0[-keep:],
            "num_frames":                  self.num_frames0,
            "episodes_done":               self.log_done_counter,
        }

        log1 = {
            "return_per_episode":          self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode":      self.log_num_frames1[-keep:],
            "num_frames":                  self.num_frames1,
            "episodes_done":               self.log_done_counter,
        }

        self.log_done_counter    = 0
        self.log_return          = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames0     = self.log_num_frames0[-self.num_procs:]
        self.log_num_frames1     = self.log_num_frames1[-self.num_procs:]
        self.num_frames0         = 0
        self.num_frames1         = 0

        return exps, log0, log1

    @abstractmethod
    def update_parameters(self):
        pass
