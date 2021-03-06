import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import babyai.rl
from babyai.rl.utils.supervised_losses import required_heads


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871

class AgentControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=imm_channels, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        return self.conv(x) * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)


class ExpertControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out


class ImageBOWEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, reduce_fn=torch.mean):
        super(ImageBOWEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.reduce_fn = reduce_fn
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = self.reduce_fn(embeddings, dim=1)
        embeddings = torch.transpose(torch.transpose(embeddings, 1, 3), 2, 3)
        return embeddings

class Encoder(nn.Module):
    def __init__(self, embedding_size, enc_dim, num_symbols):
        super().__init__()
        self.lstm = nn.LSTM(num_symbols, enc_dim)

    def forward(self, inputs):
        h, c = self.lstm(inputs)

        msg = h[-1, :, :]
        
        return msg

class Decoder(nn.Module):
    def __init__(self, embedding_size, dec_dim, max_len_msg, num_symbols):
        super().__init__()
        self.lstm   = nn.LSTM(embedding_size, dec_dim)
        self.linear = nn.Linear(dec_dim, num_symbols)
        
        self.embedding_size = embedding_size
        self.max_len_msg    = max_len_msg
        self.num_symbols    = num_symbols

    def forward(self, inputs, training, msg_hard=None, rng_states=None, cuda_rng_states=None):
        batch_size = inputs.size(0)
        
        h, c   = self.lstm(inputs.expand(self.max_len_msg, batch_size, self.embedding_size))
        logits = self.linear(h)

        #device = torch.device("cuda" if logits.is_cuda else "cpu")
        #msg = torch.zeros(self.max_len_msg, batch_size, self.num_symbols, device=device)
        
        #out_rng_states = torch.zeros(batch_size, *torch.get_rng_state().shape, dtype=torch.uint8)
        #if torch.cuda.is_available():
        #    out_cuda_rng_states = torch.zeros(batch_size, *torch.cuda.get_rng_state().shape, dtype=torch.uint8)
        
        #for i in range(batch_size):
            #if rng_states is not None:
            #    torch.set_rng_state(rng_states[i])
            #out_rng_states[i] = torch.get_rng_state()
            #if cuda_rng_states is not None:
            #    torch.cuda.set_rng_state(cuda_rng_states[i])
            #if torch.cuda.is_available():
            #    out_cuda_rng_states[i] = torch.cuda.get_rng_state()
            #for j in range(self.max_len_msg):
            #    msg[j,i,:] = nn.functional.gumbel_softmax(logits[j,i,:].unsqueeze(0)).squeeze() ### NOTE
        
        #for i in range(self.max_len_msg):
        #    msg[i,:,:] = nn.functional.gumbel_softmax(logits[i,:,:]) ### NOTE
        
        msg = self.gumbel_softmax(logits, training, msg_hard=msg_hard)
        
        #if torch.cuda.is_available():
        #    return logits, msg, out_rng_states, out_cuda_rng_states
        #else:
        #    return logits, msg, out_rng_states

        return logits, msg

    def gumbel_softmax(self, logits, training, tau=1.0, msg_hard=None):
        device = torch.device("cuda" if logits.is_cuda else "cpu")
        
        if training:
            # Here, Gumbel sample is taken:
            msg_dists = RelaxedOneHotCategorical(tau, logits=logits)
            msg       = msg_dists.rsample()
            
            if msg_hard is None:
                msg_hard = torch.zeros_like(msg, device=device)
                msg_hard.scatter_(-1, torch.argmax(msg, dim=-1, keepdim=True), 1.0)
            
            # detach() detaches the output from the computation graph, so no gradient will be backprop'ed along this variable
            msg = (msg_hard - msg).detach() + msg
        
        else:
            if msg_hard is None:
                msg = torch.zeros_like(logits, device=self.device)
                msg.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
            else:
                msg = msg_hard
        
        return msg

class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128, enc_dim=128, dec_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False, arch="cnn1",
                 max_len_msg=16, num_symbols=2, aux_info=None):
        super().__init__()

        # Decide which components are enabled
        self.use_instr   = use_instr
        self.use_memory  = use_memory
        self.arch        = arch
        self.lang_model  = lang_model
        self.aux_info    = aux_info
        self.image_dim   = image_dim
        self.memory_dim  = memory_dim
        self.instr_dim   = instr_dim
        self.enc_dim     = enc_dim
        self.dec_dim     = dec_dim
        self.max_len_msg = max_len_msg
        self.num_symbols = num_symbols

        self.obs_space = obs_space

        if arch == "cnn1":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=image_dim, kernel_size=(2, 2)),
                nn.ReLU()
            )
        elif arch == "cnn2":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2, ceil_mode=True),
                nn.Conv2d(in_channels=16, out_channels=image_dim, kernel_size=(3, 3)),
                nn.ReLU()
            )
        elif arch == "filmcnn":
            if not self.use_instr:
                raise ValueError("FiLM architecture can be used when instructions are enabled")

            self.image_conv_1 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )
            self.image_conv_2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(2, 2)),
                nn.ReLU()
            )
        elif arch.startswith("expert_filmcnn"):
            if not self.use_instr:
                raise ValueError("FiLM architecture can be used when instructions are enabled")

            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )
            self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        elif arch == 'embcnn1':
            self.image_conv = nn.Sequential(
                ImageBOWEmbedding(obs_space["image"], embedding_dim=16, padding_idx=0, reduce_fn=torch.mean),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=image_dim, kernel_size=(3, 3)),
                nn.ReLU()
            )
        else:
            raise ValueError("Incorrect architecture name: {}".format(arch))

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model in ['gru', 'conv', 'bigru', 'attgru']:
                self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
                if self.lang_model in ['gru', 'bigru', 'attgru']:
                    gru_dim = self.instr_dim
                    if self.lang_model in ['bigru', 'attgru']:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim, gru_dim, batch_first=True,
                        bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    self.final_instr_dim = self.instr_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList([
                        nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            elif self.lang_model == 'bow':
                hidden_units = [obs_space["instr"], self.instr_dim, self.instr_dim]
                layers = []
                for n_in, n_out in zip(hidden_units, hidden_units[1:]):
                    layers.append(nn.Linear(n_in, n_out))
                    layers.append(nn.ReLU())
                self.instr_bow = nn.Sequential(*layers)
                self.final_instr_dim = instr_dim

            if self.lang_model == 'attgru':
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_instr and arch != "filmcnn" and not arch.startswith("expert_filmcnn"):
            self.embedding_size += self.final_instr_dim

        if arch == "filmcnn":
            self.controller_1 = AgentControllerFiLM(
                in_features=self.final_instr_dim, out_features=64,
                in_channels=3, imm_channels=16)
            self.controller_2 = AgentControllerFiLM(
                in_features=self.final_instr_dim,
                out_features=64, in_channels=32, imm_channels=32)

        if arch.startswith("expert_filmcnn"):
            if arch == "expert_filmcnn":
                num_module = 2
            else:
                num_module = int(arch[(arch.rfind('_') + 1):])
            self.controllers = []
            for ni in range(num_module):
                if ni < num_module-1:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim+self.enc_dim,
                        out_features=128, in_channels=128, imm_channels=128)
                else:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim+self.enc_dim, out_features=self.image_dim,
                        in_channels=128, imm_channels=128)
                self.controllers.append(mod)
                self.add_module('FiLM_Controler_' + str(ni), mod)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Define encoder
        self.encoder = Encoder(self.embedding_size, self.enc_dim, self.num_symbols)
        
        # Define decoder
        self.decoder = Decoder(self.embedding_size, self.dec_dim, self.max_len_msg, self.num_symbols)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, instr_embedding=None, msg=None, msg_out=None, rng_states=None, cuda_rng_states=None):
        if self.use_instr and instr_embedding is None:
            instr_embedding = self._get_instr_embedding(obs.instr)
        if self.use_instr and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            mask = (obs.instr != 0).float()
            instr_embedding = instr_embedding[:, :mask.shape[1]]
            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)
        
        if msg is None:
            device = torch.device("cuda" if obs.instr.is_cuda else "cpu")
            msg = torch.zeros(self.max_len_msg, obs.image.size(0), self.num_symbols, device=device)
        
        msg_embedding = self.encoder(msg)
        
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        
        if self.arch == "filmcnn":
            x = self.controller_1(x, instr_embedding)
            x = self.image_conv_1(x)
            x = self.controller_2(x, instr_embedding)
            x = self.image_conv_2(x)
        elif self.arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            for controler in self.controllers:
                x = controler(x, torch.cat((instr_embedding, msg_embedding), dim=-1))
            x = F.relu(self.film_pool(x))
        else:
            x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_instr and self.arch != "filmcnn" and not self.arch.startswith("expert_filmcnn"):
            embedding = torch.cat((embedding, instr_embedding), dim=1)

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        #if torch.cuda.is_available():
        #    logits, message, rng_states, cuda_rng_states = self.decoder(embedding, rng_states, cuda_rng_states)
        #else:
        #    logits, message, rng_states                  = self.decoder(embedding, rng_states, cuda_rng_states)
        
        logits, message = self.decoder(embedding, self.training, msg_out)

        dists_speaker = Categorical(logits=F.log_softmax(logits, dim=2))
        
        #if torch.cuda.is_available():
        #    return {'dist': dist, 'value': value, 'memory': memory, 'message': message, 'dists_speaker': dists_speaker, 'rng_states': rng_states, 'cuda_rng_states': cuda_rng_states, 'extra_predictions': extra_predictions}
        #else:
        #    return {'dist': dist, 'value': value, 'memory': memory, 'message': message, 'dists_speaker': dists_speaker, 'rng_states': rng_states, 'extra_predictions': extra_predictions}
        
        return {'dist': dist, 'value': value, 'memory': memory, 'message': message, 'dists_speaker': dists_speaker, 'extra_predictions': extra_predictions}

    def _get_instr_embedding(self, instr):
        if self.lang_model == 'gru':
            _, hidden = self.instr_rnn(self.word_embedding(instr))
            return hidden[-1]

        elif self.lang_model in ['bigru', 'attgru']:
            lengths = (instr != 0).sum(1).long()
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            if outputs.shape[1] < masks.shape[1]:
                masks = masks[:, :(outputs.shape[1]-masks.shape[1])] 
                # the packing truncated the original length 
                # so we need to change mask to fit it

            return outputs if self.lang_model == 'attgru' else final_states

        elif self.lang_model == 'conv':
            inputs = self.word_embedding(instr).unsqueeze(1)  # (B,1,T,D)
            inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.instr_convs]
            inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]

            return torch.cat(inputs, 1)

        elif self.lang_model == 'bow':
            device = torch.device("cuda" if instr.is_cuda else "cpu")
            input_dim = self.obs_space["instr"]
            input = torch.zeros((instr.size(0), input_dim), device=device)
            idx = torch.arange(instr.size(0), dtype=torch.int64)
            input[idx.unsqueeze(1), instr] = 1.
            return self.instr_bow(input)
        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))

    def speaker_log_prob(self, dists_speaker, msg):
        return dists_speaker.log_prob(msg.argmax(dim=2)).mean()

    def speaker_entropy(self, dists_speaker):
        return dists_speaker.entropy().mean()
