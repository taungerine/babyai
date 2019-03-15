from abc import ABC, abstractmethod
import torch
import numpy
import random

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs0, envs1, acmodel0, acmodel1, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, aux_info):
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

        self.env0     = ParallelEnv(envs0)
        self.acmodel0 = acmodel0
        self.acmodel0.train()
        
        self.env1     = ParallelEnv(envs1)
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
        self.aux_info            = aux_info

        # Store helpers values

        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs  = len(envs0)
        self.num_frames = self.num_frames_per_proc * self.num_procs


        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs0  = self.env0.reset()
        self.obss0 = [None]*(shape[0])
        
        self.obs1  = self.env1.reset()
        self.obss1 = [None]*(shape[0])

        self.memory0   = torch.zeros(shape[1], self.acmodel0.memory_size, device=self.device)
        self.memories0 = torch.zeros(*shape,   self.acmodel0.memory_size, device=self.device)
        
        self.memory1   = torch.zeros(shape[1], self.acmodel1.memory_size, device=self.device)
        self.memories1 = torch.zeros(*shape,   self.acmodel1.memory_size, device=self.device)
        
        self.msg0  = torch.zeros(          self.acmodel0.max_len_msg, shape[1], self.acmodel0.num_symbols, device=self.device)
        self.msgs0 = torch.zeros(shape[0], self.acmodel0.max_len_msg, shape[1], self.acmodel0.num_symbols, device=self.device)
        
        self.msg1  = torch.zeros(          self.acmodel1.max_len_msg, shape[1], self.acmodel1.num_symbols, device=self.device)
        self.msgs1 = torch.zeros(shape[0], self.acmodel1.max_len_msg, shape[1], self.acmodel1.num_symbols, device=self.device)

        self.rng_states0 = torch.zeros(*shape, *torch.get_rng_state().shape, dtype=torch.uint8)
        if torch.cuda.is_available():
            self.cuda_rng_states0 = torch.zeros(*shape, *torch.cuda.get_rng_state().shape, dtype=torch.uint8)
        
        self.rng_states1 = torch.zeros(*shape, *torch.get_rng_state().shape, dtype=torch.uint8)
        if torch.cuda.is_available():
            self.cuda_rng_states1 = torch.zeros(*shape, *torch.cuda.get_rng_state().shape, dtype=torch.uint8)
        
        self.mask0              = torch.ones(shape[1], device=self.device)
        self.masks0             = torch.zeros(*shape,  device=self.device)
        self.actions0           = torch.zeros(*shape,  device=self.device, dtype=torch.int)
        self.values0            = torch.zeros(*shape,  device=self.device)
        self.rewards0           = torch.zeros(*shape,  device=self.device)
        self.advantages0        = torch.zeros(*shape,  device=self.device)
        self.log_probs0         = torch.zeros(*shape,  device=self.device)
        self.speaker_log_probs0 = torch.zeros(*shape,  device=self.device)
        
        self.mask1              = torch.ones(shape[1], device=self.device)
        self.masks1             = torch.zeros(*shape,  device=self.device)
        self.actions1           = torch.zeros(*shape,  device=self.device, dtype=torch.int)
        self.values1            = torch.zeros(*shape,  device=self.device)
        self.rewards1           = torch.zeros(*shape,  device=self.device)
        self.advantages1        = torch.zeros(*shape,  device=self.device)
        self.log_probs1         = torch.zeros(*shape,  device=self.device)
        self.speaker_log_probs1 = torch.zeros(*shape,  device=self.device)

        if self.aux_info:
            self.aux_info_collector0 = ExtraInfoCollector(self.aux_info, shape, self.device)
            self.aux_info_collector1 = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values

        self.log_episode_return0          = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return0 = torch.zeros(self.num_procs, device=self.device)
        
        self.log_episode_return1          = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return1 = torch.zeros(self.num_procs, device=self.device)
        
        self.log_episode_num_frames0      = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames1      = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter0    = 0
        self.log_return0          = [0] * self.num_procs
        self.log_reshaped_return0 = [0] * self.num_procs
        self.log_num_frames0      = [0] * self.num_procs
        
        self.log_done_counter1    = 0
        self.log_return1          = [0] * self.num_procs
        self.log_reshaped_return1 = [0] * self.num_procs
        self.log_num_frames1      = [0] * self.num_procs
        
        self.been_done0 = torch.zeros(self.num_procs, device=self.device)
        self.been_done1 = torch.zeros(self.num_procs, device=self.device)

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
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs0 = self.preprocess_obss(self.obs0, device=self.device)
            
            preprocessed_obs1 = self.preprocess_obss(self.obs1, device=self.device)
            
            with torch.no_grad():
                
                model_results0     = self.acmodel0(preprocessed_obs1, self.memory0 * self.mask0.unsqueeze(1)) ### NOTE
                
                dist0               = model_results0['dist'] ### NOTE
                value0              = model_results0['value']
                memory0             = model_results0['memory']
                msg0                = model_results0['message']
                dists_speaker0      = model_results0['dists_speaker']
                extra_predictions0  = model_results0['extra_predictions']
                self.rng_states0[i] = model_results0['rng_states']
                if torch.cuda.is_available():
                    self.cuda_rng_states0[i] = model_results0['cuda_rng_states']
                
                preprocessed_obs0.instr *= 0
                preprocessed_obs0.image *= 0
                model_results1     = self.acmodel1(preprocessed_obs0, self.memory1 * self.mask1.unsqueeze(1), msg=(msg0.transpose(0, 1) * self.mask1.unsqueeze(1).unsqueeze(2)).transpose(0, 1)) ### NOTE
                
                dist1               = model_results1['dist']
                value1              = model_results1['value']
                memory1             = model_results1['memory']
                msg1                = model_results1['message']
                dists_speaker1      = model_results1['dists_speaker']
                extra_predictions1  = model_results1['extra_predictions']
                self.rng_states1[i] = model_results1['rng_states']
                if torch.cuda.is_available():
                    self.cuda_rng_states1[i] = model_results1['cuda_rng_states']
            
            #state = torch.get_rng_state()
            action0 = dist0.sample()
            
            #torch.set_rng_state(state)
            action1 = dist1.sample()

            obs0, reward0, done0, env_info0 = self.env0.step(action0.cpu().numpy())
            
            obs1, reward1, done1, env_info1 = self.env1.step(action1.cpu().numpy())
            
            # mask any rewards based on (previous) been_done
            rewardos0 = [0] * self.num_procs
            rewardos1 = [0] * self.num_procs
            for j in range(self.num_procs):
                rewardos0[j] = reward0[j] * (1 - self.been_done0[j].item())
                rewardos1[j] = reward1[j] * (1 - self.been_done1[j].item())
        
            reward0 = tuple(rewardos0)
            reward1 = tuple(rewardos1)
            
            #reward0 = tuple(0.5*r0 + 0.5*r1 for r0, r1 in zip(reward0, reward1)) ### NOTE
            #reward1 = reward0
            
            # reward sender agent (0) equally for success of receiver agent (1) ### NOTE
            reward0 = reward1
            
            self.been_done0 = (1 - (1 - self.been_done0) * (1 - torch.tensor(done0, device=self.device, dtype=torch.float)))
            self.been_done1 = (1 - (1 - self.been_done1) * (1 - torch.tensor(done1, device=self.device, dtype=torch.float)))
            both_done       = self.been_done0 * self.been_done1
            
            # reset if receiver agent (1) is done ### NOTE
            both_done = self.been_done1
            
            obs0 = self.env0.sync_reset(both_done, obs0)
            obs1 = self.env1.sync_reset(both_done, obs1)
            
            if self.aux_info:
                env_info0 = self.aux_info_collector0.process(env_info0)
                # env_info0 = self.process_aux_info0(env_info0)
                
                env_info1 = self.aux_info_collector1.process(env_info1)
                # env_info1 = self.process_aux_info1(env_info1)

            # Update experiences values

            self.obss0[i] = self.obs0
            self.obs0     = obs0
            
            self.obss1[i] = self.obs1
            self.obs1     = obs1

            self.memories0[i] = self.memory0
            self.memory0      = memory0
            
            self.memories1[i] = self.memory1
            self.memory1      = memory1
            
            self.msgs0[i] = self.msg0
            self.msg0     = msg0
            
            self.msgs1[i] = self.msg1
            self.msg1     = msg1

            self.masks0[i]   = self.mask0
            #self.mask0       = 1 - torch.tensor(done0, device=self.device, dtype=torch.float)
            self.mask0       = 1 - both_done
            self.actions0[i] = action0
            self.values0[i]  = value0
            if self.reshape_reward is not None:
                self.rewards0[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs0, action0, reward0, done0)
                ], device=self.device)
            else:
                self.rewards0[i] = torch.tensor(reward0, device=self.device)
            self.log_probs0[i]         = dist0.log_prob(action0)
            self.speaker_log_probs0[i] = self.acmodel0.speaker_log_prob(dists_speaker0, msg0)
            
            self.masks1[i]   = self.mask1
            #self.mask1       = 1 - torch.tensor(done1, device=self.device, dtype=torch.float)
            self.mask1       = 1 - both_done
            self.actions1[i] = action1
            self.values1[i]  = value1
            if self.reshape_reward is not None:
                self.rewards1[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs1, action1, reward1, done1)
                ], device=self.device)
            else:
                self.rewards1[i] = torch.tensor(reward1, device=self.device)
            self.log_probs1[i]         = dist1.log_prob(action1)
            self.speaker_log_probs1[i] = self.acmodel1.speaker_log_prob(dists_speaker1, msg1)

            if self.aux_info:
                self.aux_info_collector0.fill_dictionaries(i, env_info0, extra_predictions0)
                
                self.aux_info_collector1.fill_dictionaries(i, env_info1, extra_predictions1)

            # Update log values

            self.log_episode_return0          += torch.tensor(reward0, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return0 += self.rewards0[i]
            
            self.log_episode_return1          += torch.tensor(reward1, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return1 += self.rewards1[i]
            
            self.log_episode_num_frames0 += torch.ones(self.num_procs, device=self.device)
            self.log_episode_num_frames1 += torch.ones(self.num_procs, device=self.device)
            
            #for i, done_ in enumerate(done0):
            for i in range(self.num_procs):
                #if done_:
                if both_done[i]:
                    self.log_done_counter0 += 1
                    self.log_return0.append(self.log_episode_return0[i].item())
                    self.log_reshaped_return0.append(self.log_episode_reshaped_return0[i].item())
                    self.log_num_frames0.append(self.log_episode_num_frames0[i].item())
            
            #for i, done_ in enumerate(done1):
                #if done_:
                    self.log_done_counter1 += 1
                    self.log_return1.append(self.log_episode_return1[i].item())
                    self.log_reshaped_return1.append(self.log_episode_reshaped_return1[i].item())
                    self.log_num_frames1.append(self.log_episode_num_frames1[i].item())

            # if both are done, reset both to not done
            self.been_done0 *= (1 - both_done)
            self.been_done1 *= (1 - both_done)

            self.log_episode_return0          *= self.mask0
            self.log_episode_reshaped_return0 *= self.mask0
            self.log_episode_num_frames0      *= self.mask0

            self.log_episode_return1          *= self.mask1
            self.log_episode_reshaped_return1 *= self.mask1
            self.log_episode_num_frames1      *= self.mask1

        # Add advantage and return to experiences

        preprocessed_obs0 = self.preprocess_obss(self.obs0, device=self.device)
        preprocessed_obs1 = self.preprocess_obss(self.obs1, device=self.device)
        
        with torch.no_grad():
            tmp         = self.acmodel0(preprocessed_obs1, self.memory0 * self.mask0.unsqueeze(1)) ### NOTE
            next_value0 = tmp['value']
        
            preprocessed_obs0.instr *= 0
            preprocessed_obs0.image *= 0
            next_value1 = self.acmodel1(preprocessed_obs0, self.memory1 * self.mask1.unsqueeze(1), msg=(tmp['message'].transpose(0, 1) * self.mask1.unsqueeze(1).unsqueeze(2)).transpose(0, 1))['value'] ### NOTE

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask0      = self.masks0[i+1]      if i < self.num_frames_per_proc - 1 else self.mask0
            next_value0     = self.values0[i+1]     if i < self.num_frames_per_proc - 1 else next_value0
            next_advantage0 = self.advantages0[i+1] if i < self.num_frames_per_proc - 1 else 0
            
            next_mask1      = self.masks1[i+1]      if i < self.num_frames_per_proc - 1 else self.mask1
            next_value1     = self.values1[i+1]     if i < self.num_frames_per_proc - 1 else next_value1
            next_advantage1 = self.advantages1[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta0              = self.rewards0[i] + self.discount * next_value0 * next_mask0 - self.values0[i]
            self.advantages0[i] = delta0 + self.discount * self.gae_lambda * next_advantage0 * next_mask0
            
            delta1              = self.rewards1[i] + self.discount * next_value1 * next_mask1 - self.values1[i]
            self.advantages1[i] = delta1 + self.discount * self.gae_lambda * next_advantage1 * next_mask1

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps0     = DictList()
        exps0.obs = [self.obss0[i][j]
                     for j in range(self.num_procs)
                     for i in range(self.num_frames_per_proc)]
        
        exps1     = DictList()
        exps1.obs = [self.obss1[i][j]
                     for j in range(self.num_procs)
                     for i in range(self.num_frames_per_proc)]
        
        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps0.memory = self.memories0.transpose(0, 1).reshape(-1, *self.memories0.shape[2:])
        
        exps1.memory = self.memories1.transpose(0, 1).reshape(-1, *self.memories1.shape[2:])
        
        exps0.message = self.msgs0.transpose(1, 2).transpose(0, 1).reshape(-1, self.acmodel0.max_len_msg, self.acmodel0.num_symbols)
        
        exps1.message = self.msgs1.transpose(1, 2).transpose(0, 1).reshape(-1, self.acmodel1.max_len_msg, self.acmodel1.num_symbols)
        
        exps0.rng_states = self.rng_states0.transpose(0, 1).reshape(-1, *self.rng_states0.shape[2:])
        if torch.cuda.is_available():
            exps0.cuda_rng_states = self.cuda_rng_states0.transpose(0, 1).reshape(-1, *self.cuda_rng_states0.shape[2:])
        
        exps1.rng_states = self.rng_states1.transpose(0, 1).reshape(-1, *self.rng_states1.shape[2:])
        if torch.cuda.is_available():
            exps1.cuda_rng_states = self.cuda_rng_states1.transpose(0, 1).reshape(-1, *self.cuda_rng_states1.shape[2:])
        
        # T x P -> P x T -> (P * T) x 1
        exps0.mask = self.masks0.transpose(0, 1).reshape(-1).unsqueeze(1)
        
        exps1.mask = self.masks1.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps0.action           = self.actions0.transpose(0, 1).reshape(-1)
        exps0.value            = self.values0.transpose(0, 1).reshape(-1)
        exps0.reward           = self.rewards0.transpose(0, 1).reshape(-1)
        exps0.advantage        = self.advantages0.transpose(0, 1).reshape(-1)
        exps0.returnn          = exps0.value + exps0.advantage
        exps0.log_prob         = self.log_probs0.transpose(0, 1).reshape(-1)
        exps0.speaker_log_prob = self.speaker_log_probs0.transpose(0, 1).reshape(-1)
        
        exps1.action           = self.actions1.transpose(0, 1).reshape(-1)
        exps1.value            = self.values1.transpose(0, 1).reshape(-1)
        exps1.reward           = self.rewards1.transpose(0, 1).reshape(-1)
        exps1.advantage        = self.advantages1.transpose(0, 1).reshape(-1)
        exps1.returnn          = exps1.value + exps1.advantage
        exps1.log_prob         = self.log_probs1.transpose(0, 1).reshape(-1)
        exps1.speaker_log_prob = self.speaker_log_probs1.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps0 = self.aux_info_collector0.end_collection(exps0)
        
            exps1 = self.aux_info_collector1.end_collection(exps1)

        # Preprocess experiences

        exps0.obs = self.preprocess_obss(exps0.obs, device=self.device)

        exps1.obs = self.preprocess_obss(exps1.obs, device=self.device)

        # Log some values

        keep0 = max(self.log_done_counter0, self.num_procs)

        keep1 = max(self.log_done_counter1, self.num_procs)

        log0 = {
            "return_per_episode":          self.log_return0[-keep0:],
            "reshaped_return_per_episode": self.log_reshaped_return0[-keep0:],
            "num_frames_per_episode":      self.log_num_frames0[-keep0:],
            "num_frames":                  self.num_frames,
            "episodes_done":               self.log_done_counter0,
        }

        log1 = {
            "return_per_episode":          self.log_return1[-keep1:],
            "reshaped_return_per_episode": self.log_reshaped_return1[-keep1:],
            "num_frames_per_episode":      self.log_num_frames1[-keep1:],
            "num_frames":                  self.num_frames,
            "episodes_done":               self.log_done_counter1,
        }

        self.log_done_counter0    = 0
        self.log_return0          = self.log_return0[-self.num_procs:]
        self.log_reshaped_return0 = self.log_reshaped_return0[-self.num_procs:]
        self.log_num_frames0      = self.log_num_frames0[-self.num_procs:]

        self.log_done_counter1    = 0
        self.log_return1          = self.log_return1[-self.num_procs:]
        self.log_reshaped_return1 = self.log_reshaped_return1[-self.num_procs:]
        self.log_num_frames1      = self.log_num_frames1[-self.num_procs:]

        return exps0, log0, exps1, log1

    @abstractmethod
    def update_parameters(self):
        pass
