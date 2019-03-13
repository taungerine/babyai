import numpy
import torch
import torch.nn.functional as F


from babyai.rl.algos.base import BaseAlgo


class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs0, envs1, acmodel0, acmodel1, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, aux_info=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs0, envs1, acmodel0, acmodel1, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         aux_info)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer0 = torch.optim.Adam(self.acmodel0.parameters(), lr, (beta1, beta2), eps=adam_eps)
        
        self.optimizer1 = torch.optim.Adam(self.acmodel1.parameters(), lr, (beta1, beta2), eps=adam_eps)
        
        self.batch_num = 0

    def update_parameters(self):
        # Collect experiences

        exps0, logs0, exps1, logs1 = self.collect_experiences()
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

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies0     = []
            log_values0        = []
            log_policy_losses0 = []
            log_value_losses0  = []
            log_grad_norms0    = []
            
            log_entropies1     = []
            log_values1        = []
            log_policy_losses1 = []
            log_value_losses1  = []
            log_grad_norms1    = []

            log_losses0 = []
            log_losses1 = []

            '''
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one. Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            '''

            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches
                # Initialize batch values

                batch_entropy0     = 0
                batch_value0       = 0
                batch_policy_loss0 = 0
                batch_value_loss0  = 0
                batch_loss0        = 0
                
                batch_entropy1     = 0
                batch_value1       = 0
                batch_policy_loss1 = 0
                batch_value_loss1  = 0
                batch_loss1        = 0

                # Initialize memory
                memory0 = exps0.memory[inds]
                
                memory1 = exps1.memory[inds]
                
                msg0 = exps0.message[inds].transpose(0, 1)
                
                msg1 = exps1.message[inds].transpose(0, 1)

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb0 = exps0[inds + i]

                    sb1 = exps1[inds + i]

                    # Compute loss

                    model_results0     = self.acmodel0(sb1.obs, memory0 * sb0.mask, msg=(msg1.transpose(0, 1) * sb0.mask.unsqueeze(2)).transpose(0, 1))

                    model_results1     = self.acmodel1(sb0.obs, memory1 * sb1.mask, msg=(msg0.transpose(0, 1) * sb1.mask.unsqueeze(2)).transpose(0, 1))
                    
                    dist0              = model_results0['dist']
                    value0             = model_results0['value']
                    memory0            = model_results0['memory']
                    msg0               = model_results0['message']
                    extra_predictions0 = model_results0['extra_predictions']
                    
                    dist1              = model_results1['dist']
                    value1             = model_results1['value']
                    memory1            = model_results1['memory']
                    msg1               = model_results1['message']
                    extra_predictions1 = model_results1['extra_predictions']
                    
                    entropy0 = dist0.entropy().mean()
                    
                    entropy1 = dist1.entropy().mean()

                    ratio0       = torch.exp(dist0.log_prob(sb0.action) - sb0.log_prob)
                    surr10       = ratio0 * sb0.advantage
                    surr20       = torch.clamp(ratio0, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb0.advantage
                    policy_loss0 = -torch.min(surr10, surr20).mean()
                    
                    ratio1       = torch.exp(dist1.log_prob(sb1.action) - sb1.log_prob)
                    surr11       = ratio1 * sb1.advantage
                    surr21       = torch.clamp(ratio1, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb1.advantage
                    policy_loss1 = -torch.min(surr11, surr21).mean()

                    value_clipped0 = sb0.value + torch.clamp(value0 - sb0.value, -self.clip_eps, self.clip_eps)
                    surr10         = (value0 - sb0.returnn).pow(2)
                    surr20         = (value_clipped0 - sb0.returnn).pow(2)
                    value_loss0    = torch.max(surr10, surr20).mean()
                    
                    value_clipped1 = sb1.value + torch.clamp(value1 - sb1.value, -self.clip_eps, self.clip_eps)
                    surr11         = (value1 - sb1.returnn).pow(2)
                    surr21         = (value_clipped1 - sb1.returnn).pow(2)
                    value_loss1    = torch.max(surr11, surr21).mean()

                    loss0 = policy_loss0 - self.entropy_coef * entropy0 + self.value_loss_coef * value_loss0
                    
                    loss1 = policy_loss1 - self.entropy_coef * entropy1 + self.value_loss_coef * value_loss1

                    # Update batch values

                    batch_entropy0     += entropy0.item()
                    batch_value0       += value0.mean().item()
                    batch_policy_loss0 += policy_loss0.item()
                    batch_value_loss0  += value_loss0.item()
                    batch_loss0        += loss0
                    
                    batch_entropy1     += entropy1.item()
                    batch_value1       += value1.mean().item()
                    batch_policy_loss1 += policy_loss1.item()
                    batch_value_loss1  += value_loss1.item()
                    batch_loss1        += loss1

                    # Update memories for next epoch

                    if i < self.recurrence - 1:
                        exps0.memory[inds + i + 1] = memory0.detach()
                        
                        exps1.memory[inds + i + 1] = memory1.detach()
            
                        exps0.message[inds + i + 1] = msg0.transpose(0, 1).detach()
            
                        exps1.message[inds + i + 1] = msg1.transpose(0, 1).detach()

                # Update batch values

                batch_entropy0     /= self.recurrence
                batch_value0       /= self.recurrence
                batch_policy_loss0 /= self.recurrence
                batch_value_loss0  /= self.recurrence
                batch_loss0        /= self.recurrence
                
                batch_entropy1     /= self.recurrence
                batch_value1       /= self.recurrence
                batch_policy_loss1 /= self.recurrence
                batch_value_loss1  /= self.recurrence
                batch_loss1        /= self.recurrence

                # Update actor-critic

                self.optimizer0.zero_grad()
                batch_loss0.backward(retain_graph=True)
                grad_norm0 = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel0.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel0.parameters(), self.max_grad_norm)
                self.optimizer0.step()
                
                self.optimizer1.zero_grad()
                batch_loss1.backward(retain_graph=True)
                grad_norm1 = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel1.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel1.parameters(), self.max_grad_norm)
                self.optimizer1.step()

                # Update log values

                log_entropies0.append(batch_entropy0)
                log_values0.append(batch_value0)
                log_policy_losses0.append(batch_policy_loss0)
                log_value_losses0.append(batch_value_loss0)
                log_grad_norms0.append(grad_norm0.item())
                log_losses0.append(batch_loss0.item())

                log_entropies1.append(batch_entropy1)
                log_values1.append(batch_value1)
                log_policy_losses1.append(batch_policy_loss1)
                log_value_losses1.append(batch_value_loss1)
                log_grad_norms1.append(grad_norm1.item())
                log_losses1.append(batch_loss1.item())

        # Log some values

        logs0["entropy"]     = numpy.mean(log_entropies0)
        logs0["value"]       = numpy.mean(log_values0)
        logs0["policy_loss"] = numpy.mean(log_policy_losses0)
        logs0["value_loss"]  = numpy.mean(log_value_losses0)
        logs0["grad_norm"]   = numpy.mean(log_grad_norms0)
        logs0["loss"]        = numpy.mean(log_losses0)
        
        logs1["entropy"]     = numpy.mean(log_entropies1)
        logs1["value"]       = numpy.mean(log_values1)
        logs1["policy_loss"] = numpy.mean(log_policy_losses1)
        logs1["value_loss"]  = numpy.mean(log_value_losses1)
        logs1["grad_norm"]   = numpy.mean(log_grad_norms1)
        logs1["loss"]        = numpy.mean(log_losses1)

        return logs0, logs1

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
