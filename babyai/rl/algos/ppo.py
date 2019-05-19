import numpy
import torch
import torch.nn.functional as F


from babyai.rl.algos.base import BaseAlgo


class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel0, acmodel1, n, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, use_comm=True, aux_info=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel0, acmodel1, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, use_comm, n,
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

        exps, logs = self.collect_experiences()
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

            log_entropies     = []
            log_values        = []
            log_policy_losses = []
            log_value_losses  = []
            log_grad_norms    = []

            log_losses = []

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

                # Initialize
                value0  = torch.zeros(inds.shape[0], device=self.device)
                value1  = torch.zeros(inds.shape[0], device=self.device)
                memory0 = exps.memory0[inds]
                memory1 = exps.memory1[inds]
                msg     = exps.message[inds]
                
                entropies1 = torch.zeros(inds.shape[0], device=self.device)
                log_prob0  = torch.zeros(inds.shape[0], device=self.device)
                log_prob1  = torch.zeros(inds.shape[0], device=self.device)

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss
                    
                    if self.use_comm:
                    
                        if torch.any(sb.comm):
                            # blind the scout to instructions
                            sb.globs.instr[sb.comm] *= 0
                            
                            model_results0  = self.acmodel0(sb.globs[  sb.comm], memory0[    sb.comm] * sb.mask[    sb.comm], msg_out=msg[sb.comm])
                            
                            msg[sb.comm] = model_results0['message']
                            
                            model_results1A = self.acmodel1(sb.obs[    sb.comm], memory1[    sb.comm] * sb.mask[    sb.comm], msg=(msg[    sb.comm]))
                        
                        if torch.any(1 - sb.comm):
                            model_results1B = self.acmodel1(sb.obs[1 - sb.comm], memory1[1 - sb.comm] * sb.mask[1 - sb.comm], msg=(msg[1 - sb.comm]))
                    else:
                        model_results1B = self.acmodel1(sb.obs, memory1 * sb.mask, msg=msg)
                    
                    if torch.any(sb.comm):
                        dists_speaker    = model_results0['dists_speaker']
                        value0[sb.comm]  = model_results0['value']
                        memory0[sb.comm] = model_results0['memory']
                        
                        distA            = model_results1A['dist']
                        value1[sb.comm]  = model_results1A['value']
                        memory1[sb.comm] = model_results1A['memory']
                                
                    if torch.any(1 - sb.comm):
                        distB                = model_results1B['dist']
                        value1[1 - sb.comm]  = model_results1B['value']
                        memory1[1 - sb.comm] = model_results1B['memory']
                    
                    if torch.any(sb.comm):
                        entropy0                = self.acmodel0.speaker_entropy(dists_speaker).mean()
                        entropies1[    sb.comm] = distA.entropy()
                    if torch.any(1 - sb.comm):
                        entropies1[1 - sb.comm] = distB.entropy()
                    entropy1 = entropies1.mean()
                    
                    if torch.any(sb.comm):
                        log_prob0[    sb.comm] = self.acmodel0.speaker_log_prob(dists_speaker, msg[sb.comm])
                        log_prob1[    sb.comm] = distA.log_prob(sb.action[    sb.comm])
                    if torch.any(1 - sb.comm):
                        log_prob1[1 - sb.comm] = distB.log_prob(sb.action[1 - sb.comm])
                    
                    if torch.any(sb.comm):
                        ratio0       = torch.exp(log_prob0[sb.comm] - sb.log_prob0[sb.comm])
                        surr10       = ratio0 * sb.advantage0[sb.comm]
                        surr20       = torch.clamp(ratio0, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage0[sb.comm]
                        policy_loss0 = -torch.min(surr10, surr20).mean()
                        
                        value_clipped0 = sb.value0[sb.comm] + torch.clamp(value0[sb.comm] - sb.value0[sb.comm], -self.clip_eps, self.clip_eps)
                        surr10         = (value0[sb.comm] - sb.returnn0[sb.comm]).pow(2)
                        surr20         = (value_clipped0 - sb.returnn0[sb.comm]).pow(2)
                        value_loss0    = torch.max(surr10, surr20).mean()
                        
                        loss0 = policy_loss0 - self.entropy_coef * entropy0 + self.value_loss_coef * value_loss0
                        
                        # Update batch values
                        
                        batch_entropy0     += entropy0.item()
                        batch_value0       += value0.mean().item()
                        batch_policy_loss0 += policy_loss0.item()
                        batch_value_loss0  += value_loss0.item()
                        batch_loss0        += loss0
                    
                    ratio1       = torch.exp(log_prob1 - sb.log_prob1)
                    surr11       = ratio1 * sb.advantage1
                    surr21       = torch.clamp(ratio1, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage1
                    policy_loss1 = -torch.min(surr11, surr21).mean()
                    
                    value_clipped1 = sb.value1 + torch.clamp(value1 - sb.value1, -self.clip_eps, self.clip_eps)
                    surr11         = (value1 - sb.returnn1).pow(2)
                    surr21         = (value_clipped1 - sb.returnn1).pow(2)
                    value_loss1    = torch.max(surr11, surr21).mean()
                    
                    loss1 = policy_loss1 - self.entropy_coef * entropy1 + self.value_loss_coef * value_loss1

                    # Update batch values
                    
                    batch_entropy1     += entropy1.item()
                    batch_value1       += value1.mean().item()
                    batch_policy_loss1 += policy_loss1.item()
                    batch_value_loss1  += value_loss1.item()
                    batch_loss1        += loss1

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
                self.optimizer1.zero_grad()
                batch_loss0.backward()
                batch_loss1.backward()
                grad_norm0 = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel0.parameters() if p.grad is not None) ** 0.5
                grad_norm1 = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel1.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel0.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.acmodel1.parameters(), self.max_grad_norm)
                self.optimizer0.step()
                self.optimizer1.step()
                
                # Update log values

                log_entropies.append(batch_entropy0 + batch_entropy1)
                log_values.append(batch_value0 + batch_value1)
                log_policy_losses.append(batch_policy_loss0 + batch_policy_loss1)
                log_value_losses.append(batch_value_loss0 + batch_value_loss1)
                log_grad_norms.append(grad_norm0.item() + grad_norm1.item())
                log_losses.append(batch_loss0.item() + batch_loss1.item())

        # Log some values

        logs["entropy"]     = numpy.mean(log_entropies)
        logs["value"]       = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"]  = numpy.mean(log_value_losses)
        logs["grad_norm"]   = numpy.mean(log_grad_norms)
        logs["loss"]        = numpy.mean(log_losses)

        return logs, logs

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
