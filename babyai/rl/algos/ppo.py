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

        # Initialize log values
        
        log_entropies0     = []
        log_values0        = []
        log_policy_losses0 = []
        log_value_losses0  = []
        log_grad_norms0    = []
        
        log_losses0 = []
        
        log_entropies1     = []
        log_values1        = []
        log_policy_losses1 = []
        log_value_losses1  = []
        log_grad_norms1    = []
        
        log_losses1 = []
        
        for _ in range(self.epochs):
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
                
                batch_loss         = 0

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
                value   = torch.zeros(inds.shape[0], device=self.device)
                memory0 = exps.memory0[inds]
                memory1 = exps.memory1[inds]
                msg     = exps.message[inds]
                
                entropies = torch.zeros(inds.shape[0], device=self.device)
                log_prob  = torch.zeros(inds.shape[0], device=self.device)

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss
                    
                    if torch.any(sb.scouting):
                        if self.ignorant_scout:
                            # blind the scout to instructions
                            sb.globs.instr[sb.scouting] *= 0
                        
                        model_results0 = self.acmodel0(sb.globs[    sb.scouting], memory0[    sb.scouting] * sb.mask0[    sb.scouting], msg_out=sb.message_out[sb.scouting])
                    
                    if torch.any(1 - sb.scouting):
                        
                        if self.use_comm:
                            model_results1 = self.acmodel1(sb.obs[1 - sb.scouting], memory1[1 - sb.scouting] * sb.mask1[1 - sb.scouting], msg=(msg[1 - sb.scouting]))
                        else:
                            model_results1 = self.acmodel1(sb.obs[1 - sb.scouting], memory1[1 - sb.scouting] * sb.mask1[1 - sb.scouting])
                    
                    if torch.any(sb.scouting):
                        value[sb.scouting]   = model_results0['value']
                        memory0[sb.scouting] = model_results0['memory']
                        msg[sb.scouting]     = model_results0['message']
                        dists_speaker        = model_results0['dists_speaker']
                    
                    if torch.any(1 - sb.scouting):
                        dist1                    = model_results1['dist']
                        value[1 - sb.scouting]   = model_results1['value']
                        memory1[1 - sb.scouting] = model_results1['memory']
                    
                    if torch.any(sb.scouting):
                        entropies[sb.scouting]     = self.acmodel0.speaker_entropy(dists_speaker)
                    if torch.any(1 - sb.scouting):
                        entropies[1 - sb.scouting] = dist1.entropy()
                    entropy  = entropies.mean()
                    entropy0 = entropies[    sb.scouting].sum() # for log
                    entropy1 = entropies[1 - sb.scouting].sum() # for log
                    
                    if torch.any(sb.scouting):
                        log_prob[sb.scouting]     = self.acmodel0.speaker_log_prob(dists_speaker, msg[sb.scouting])
                    if torch.any(1 - sb.scouting):
                        log_prob[1 - sb.scouting] = dist1.log_prob(sb.action[1 - sb.scouting])
                    
                    ratio        = torch.exp(log_prob - sb.log_prob)
                    surr1        = ratio * sb.advantage
                    surr2        = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss  = -torch.min(surr1, surr2).mean()
                    policy_loss0 = -torch.min(surr1, surr2)[    sb.scouting].sum() # for log
                    policy_loss1 = -torch.min(surr1, surr2)[1 - sb.scouting].sum() # for log
                    
                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1         = (value - sb.returnn).pow(2)
                    surr2         = (value_clipped - sb.returnn).pow(2)
                    value_loss    = torch.max(surr1, surr2).mean()
                    value_loss0   = torch.max(surr1, surr2)[    sb.scouting].sum() # for log
                    value_loss1   = torch.max(surr1, surr2)[1 - sb.scouting].sum() # for log
                    
                    loss  = policy_loss  - self.entropy_coef * entropy  + self.value_loss_coef * value_loss
                    loss0 = policy_loss0 - self.entropy_coef * entropy0 + self.value_loss_coef * value_loss0 # for log
                    loss1 = policy_loss1 - self.entropy_coef * entropy1 + self.value_loss_coef * value_loss1 # for log

                    # Update batch values
                    
                    batch_loss         += loss
                    
                    batch_entropy0     += entropy0.item()
                    batch_value0       += value[    sb.scouting].sum().item()
                    batch_policy_loss0 += policy_loss0.item()
                    batch_value_loss0  += value_loss0.item()
                    batch_loss0        += loss0
                    
                    batch_entropy1     += entropy1.item()
                    batch_value1       += value[1 - sb.scouting].sum().item()
                    batch_policy_loss1 += policy_loss1.item()
                    batch_value_loss1  += value_loss1.item()
                    batch_loss1        += loss1

                # Update batch values
                
                batch_loss /= self.recurrence

                # Update actor-critic
                
                self.optimizer0.zero_grad()
                self.optimizer1.zero_grad()
                batch_loss.backward()
                grad_norm0 = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel0.parameters() if p.grad is not None) ** 0.5
                grad_norm1 = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel1.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel0.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.acmodel1.parameters(), self.max_grad_norm)
                self.optimizer0.step()
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

        log0["entropy"]     = numpy.sum(log_entropies0)     / (log0["num_frames"] * self.epochs)
        log0["value"]       = numpy.sum(log_values0)        / (log0["num_frames"] * self.epochs)
        log0["policy_loss"] = numpy.sum(log_policy_losses0) / (log0["num_frames"] * self.epochs)
        log0["value_loss"]  = numpy.sum(log_value_losses0)  / (log0["num_frames"] * self.epochs)
        log0["loss"]        = numpy.sum(log_losses0)        / (log0["num_frames"] * self.epochs)
        log0["grad_norm"]   = numpy.mean(log_grad_norms0)
        
        log1["entropy"]     = numpy.sum(log_entropies1)     / (log1["num_frames"] * self.epochs)
        log1["value"]       = numpy.sum(log_values1)        / (log1["num_frames"] * self.epochs)
        log1["policy_loss"] = numpy.sum(log_policy_losses1) / (log1["num_frames"] * self.epochs)
        log1["value_loss"]  = numpy.sum(log_value_losses1)  / (log1["num_frames"] * self.epochs)
        log1["loss"]        = numpy.sum(log_losses1)        / (log1["num_frames"] * self.epochs)
        log1["grad_norm"]   = numpy.mean(log_grad_norms1)

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
