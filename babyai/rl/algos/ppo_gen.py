import numpy
import torch
import torch.nn.functional as F
import math


from babyai.rl.algos.base import BaseAlgo


class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel0, acmodel1, n, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, use_comm=True, ignorant_scout=False, aux_info=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel0, acmodel1, n, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, use_comm, ignorant_scout,
                         aux_info)
        
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0
        
        self.optimizer0 = torch.optim.Adam(self.acmodel0.parameters(), lr, (beta1, beta2), eps=adam_eps)
        self.optimizer1 = torch.optim.Adam(self.acmodel1.parameters(), lr, (beta1, beta2), eps=adam_eps)
        
        self.batch_num = 0
        
    
    
        self.buffer = None
    
        self.update = 0
    
        self.objs = 0

    def update_parameters(self):
        # Collect experiences

        exps, log0, log1 = self.collect_experiences()
        '''
        exps is a DictList with the following keys ['obs', 'memory', 'mask', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obj.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''
        
        if self.update == 0:
            self.objs = ((3 < exps.globs[0].image[:, :, 0].numpy()) * (exps.globs[0].image[:, :, 0].numpy() < 8)).sum()
        
        data = torch.zeros(self.num_frames, 4 + self.acmodel0.max_len_msg + 3 + 5 * self.objs, dtype=torch.uint8)
        
        k     = 0
        k_max = 0
        for i in range(self.num_procs):
            new_episode = False
            for j in range(self.num_frames_per_proc):
                if exps.scouting[i * self.num_frames_per_proc + j]:
                    continue
                
                if exps.mask1[i * self.num_frames_per_proc + j] == 0:
                    new_episode = True
                    k_max       = k
                
                if new_episode:
                    index = i * self.num_frames_per_proc + j
                    
                    # agent coordinates
                    data[k, 0:2]  = exps.agent_loc[index]
                    
                    # goal type
                    data[k, 2]    = exps.goal_type[index]
                    
                    # goal color
                    data[k, 3]    = exps.goal_color[index]
                    
                    # message
                    data[k, 4:4+self.acmodel0.max_len_msg] = exps.message[index].argmax(-1)
                    
                    # agent action
                    data[k, 4+self.acmodel0.max_len_msg]   = exps.action[index]
                    
                    # new episode
                    data[k, 4+self.acmodel0.max_len_msg+1]   = 1 - exps.mask1[index]
                    
                    # success
                    data[k, 4+self.acmodel0.max_len_msg+2]   = math.ceil(exps.reward[index].clamp(0, 1).item())
                    
                    globs = exps.globs[index].image[:, :, :].numpy()
                    x, y = ((3 < globs[:, :, 0]) * (globs[:, :, 0] < 8) + (13 < globs[:, :, 0]) * (globs[:, :, 0] < 18)).nonzero()
                    
                    for b in range(self.objs):
                        if b < len(x):
                            # object x coordinate
                            data[k, 4+self.acmodel0.max_len_msg+3 + 5*b    ] = x[b].item()
                            
                            # object y coordinate
                            data[k, 4+self.acmodel0.max_len_msg+3 + 5*b + 1] = y[b].item()
                            
                            # object type
                            data[k, 4+self.acmodel0.max_len_msg+3 + 5*b + 2] = globs[x[b], y[b], 0].item() % 10
                            
                            # object color
                            data[k, 4+self.acmodel0.max_len_msg+3 + 5*b + 3] = globs[x[b], y[b], 1].item()
                        
                            # object lockedness
                            data[k, 4+self.acmodel0.max_len_msg+3 + 5*b + 4] = globs[x[b], y[b], 2].item()
                        
                        else:
                            data[k, 4+self.acmodel0.max_len_msg+3 + 5*b:4+self.acmodel0.max_len_msg+3 + 5*b + 5] = torch.zeros(5)
                    
                    k += 1
    
            k = k_max
        
        if self.buffer is None:
            self.buffer = data[:k_max].numpy()
        else:
            self.buffer = numpy.concatenate((self.buffer, data[:k_max].numpy()))
        
        # Log some values

        log0["entropy"]     = 0
        log0["value"]       = 0
        log0["policy_loss"] = 0
        log0["value_loss"]  = 0
        log0["loss"]        = 0
        log0["grad_norm"]   = 0
        
        log1["entropy"]     = 0
        log1["value"]       = 0
        log1["policy_loss"] = 0
        log1["value_loss"]  = 0
        log1["loss"]        = 0
        log1["grad_norm"]   = 0
        
        self.update += 1

        return log0, log1

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
