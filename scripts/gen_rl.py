#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import os
import logging
import csv
import json
import gym
import time
import datetime
import torch
import numpy as np
import subprocess

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model import ACModel
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent

# three functions that should be in babyai/utils/model.py, but that doesn't seem to work
def get_optimizer_path(model_name, n):
    return os.path.join(utils.get_model_dir(model_name), "optimizer" + str(n) + ".pt")


def load_optimizer(model_name, n, raise_not_found=True):
    path = get_optimizer_path(model_name, n)
    try:
        if not torch.cuda.is_available():
            optimizer = torch.load(path, map_location='cpu')
        else:
            optimizer = torch.load(path)
        return optimizer
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No optimizer found at {}".format(path))


def save_optimizer(optimizer, model_name, n):
    path = get_optimizer_path(model_name, n)
    utils.create_folders_if_necessary(path)
    torch.save(optimizer, path)


# Parse arguments
parser = ArgumentParser()
parser.add_argument("--algo", default='ppo',
                    help="algorithm to use (default: ppo)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--reward-scale", type=float, default=20.,
                    help="Reward scale multiplier")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--ppo-epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--len-message", type=int, default=16,
                    help="lengths of messages (default: 16)")
parser.add_argument("--num-symbols", type=int, default=2,
                    help="number of symbols (default: 2)")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of decoder/encoder layers (default: 1)")
parser.add_argument("--enc-dim", type=int, default=128,
                    help="dimensionality of the encoder LSTM")
parser.add_argument("--dec-dim", type=int, default=128,
                    help="dimensionality of the decoder LSTM")
parser.add_argument("--no-comm", action="store_true", default=False,
                    help="don't use communication")
parser.add_argument("--all-angles", action="store_true", default=False,
                    help="let the sender observe the environment from all angles")
parser.add_argument("--disc-comm", action="store_true", default=False,
                    help="use discrete instead of continuous communication")
parser.add_argument("--disc-comm-rl", action="store_true", default=False,
                    help="use discrete instead of continuous communication (RL)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="use argmax for messages instead of sampling")
parser.add_argument("--tau-init", type=float, default=1.0,
                    help="initial Gumbel temperature (default: 1.0)")
parser.add_argument("--n", type=int, default=64,
                    help="period of sender messages (default: 64)")
parser.add_argument("--ignorant-scout", action="store_true", default=False,
                    help="blinds the sender to the instruction")
parser.add_argument("--gen", type=int, default=1000,
                    help="minimum number of frames to generate (default: 1000)")
args = parser.parse_args()

utils.seed(args.seed)

# Generate environments
envs = []
for i in range(args.procs):
    env = gym.make(args.env)
    env.seed(100 * args.seed + i)
    envs.append(env)

# Define model name
suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
instr = args.instr_arch if args.instr_arch else "noinstr"
mem = "mem" if not args.no_mem else "nomem"
model_name_parts = {
    'env': args.env,
    'algo': args.algo,
    'arch': args.arch,
    'instr': instr,
    'mem': mem,
    'seed': args.seed,
    'info': '',
    'coef': '',
    'suffix': suffix}
default_model_name = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**model_name_parts)
if args.pretrained_model:
    default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
args.model = args.model.format(**model_name_parts) if args.model else default_model_name

utils.configure_logging(args.model)
logger0 = logging.getLogger(__name__ + "0")
logger1 = logging.getLogger(__name__ + "1")

# Define obss preprocessor
if 'emb' in args.arch:
    obss_preprocessor = utils.IntObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
else:
    obss_preprocessor = utils.ObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)

loading_flag = True

# Define actor-critic model
acmodel0 = utils.load_model(args.model, 0, raise_not_found=False)
acmodel1 = utils.load_model(args.model, 1, raise_not_found=False)
if acmodel0 is None:
    loading_flag = False
    if args.pretrained_model:
        acmodel0 = utils.load_model(args.pretrained_model, 0, raise_not_found=True)
    else:
        #torch.manual_seed(args.seed)
        acmodel0 = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                           args.image_dim, args.memory_dim, args.instr_dim, args.enc_dim, args.dec_dim,
                           not args.no_instr, args.instr_arch, not args.no_mem, args.arch,
                           args.len_message, args.num_symbols, args.num_layers, args.all_angles, args.disc_comm, args.disc_comm_rl, args.argmax, args.tau_init)
if acmodel1 is None:
    loading_flag = False
    if args.pretrained_model:
        acmodel1 = utils.load_model(args.pretrained_model, 1, raise_not_found=True)
    else:
        #torch.manual_seed(args.seed)
        acmodel1 = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                           args.image_dim, args.memory_dim, args.instr_dim, args.enc_dim, args.dec_dim,
                           not args.no_instr, args.instr_arch, not args.no_mem, args.arch,
                           args.len_message, args.num_symbols, args.num_layers, False, args.disc_comm, args.disc_comm_rl, args.argmax, args.tau_init)

obss_preprocessor.vocab.save()
utils.save_model(acmodel0, args.model, 0)
utils.save_model(acmodel1, args.model, 1)

if torch.cuda.is_available():
    acmodel0.cuda()
    acmodel1.cuda()

# Define actor-critic algo

reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
if args.algo == "ppo":
    algo = babyai.rl.PPOAlgo(envs, acmodel0, acmodel1, args.n, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                              args.gae_lambda,
                              args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                              args.optim_eps, args.clip_eps, args.ppo_epochs, args.batch_size, obss_preprocessor,
                              reshape_reward, not args.no_comm, args.ignorant_scout)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

if loading_flag:
    algo.optimizer0.load_state_dict(load_optimizer(args.model, 0, raise_not_found=False).state_dict())
    algo.optimizer1.load_state_dict(load_optimizer(args.model, 1, raise_not_found=False).state_dict())
    
    #algo.optimizer0 = load_optimizer(args.model, 0, raise_not_found=False)
    #algo.optimizer1 = load_optimizer(args.model, 1, raise_not_found=False)
else:
    save_optimizer(algo.optimizer0, args.model, 0)
    save_optimizer(algo.optimizer1, args.model, 1)

# When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
# Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
# the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

utils.seed(args.seed)

# Restore training status

status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
if os.path.exists(status_path):
    with open(status_path, 'r') as src:
        status = json.load(src)
else:
    status = {'i': 0,
              'num_episodes': 0,
              'num_frames':   0,
              'num_frames0':  0,
              'num_frames1':  0,
}

# Define loggers and Tensorboard writer and CSV writers

header = (["update", "episodes", "frames", "FPS", "duration"]
          + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["success_rate"]
          + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])
if args.tb:
    from tensorboardX import SummaryWriter

    writer0 = SummaryWriter(utils.get_log_dir(args.model))
    writer1 = SummaryWriter(utils.get_log_dir(args.model))
csv_path0 = os.path.join(utils.get_log_dir(args.model), 'log0.csv')
first_created0 = not os.path.exists(csv_path0)
# we don't buffer data going in the csv log, cause we assume
# that one update will take much longer that one write to the log
csv_writer0 = csv.writer(open(csv_path0, 'a', 1))
if first_created0:
    csv_writer0.writerow(header)

csv_path1 = os.path.join(utils.get_log_dir(args.model), 'log1.csv')
first_created1 = not os.path.exists(csv_path1)
# we don't buffer data going in the csv log, cause we assume
# that one update will take much longer that one write to the log
csv_writer1 = csv.writer(open(csv_path1, 'a', 1))
if first_created1:
    csv_writer1.writerow(header)

# Log code state, command, availability of CUDA and models

babyai_code = list(babyai.__path__)[0]
try:
    last_commit = subprocess.check_output(
        'cd {}; git log -n1'.format(babyai_code), shell=True).decode('utf-8')
    logger0.info('LAST COMMIT INFO:')
    logger0.info(last_commit)
    logger1.info('LAST COMMIT INFO:')
    logger1.info(last_commit)
except subprocess.CalledProcessError:
    logger0.info('Could not figure out the last commit')
    logger1.info('Could not figure out the last commit')
try:
    diff = subprocess.check_output(
        'cd {}; git diff'.format(babyai_code), shell=True).decode('utf-8')
    if diff:
        logger0.info('GIT DIFF:')
        logger0.info(diff)
        logger1.info('GIT DIFF:')
        logger1.info(diff)
except subprocess.CalledProcessError:
    logger0.info('Could not figure out the last commit')
    logger1.info('Could not figure out the last commit')
logger0.info('COMMAND LINE ARGS:')
logger0.info(args)
logger0.info("CUDA available: {}".format(torch.cuda.is_available()))
logger0.info(acmodel0)
logger1.info('COMMAND LINE ARGS:')
logger1.info(args)
logger1.info("CUDA available: {}".format(torch.cuda.is_available()))
logger1.info(acmodel1)

# Train models

total_start_time  = time.time()
best_success_rate = 0
test_env_name     = args.env
while status['num_frames'] < args.frames:
    # Update parameters
    
    update_start_time = time.time()
    logs0, logs1      = algo.update_parameters()
    update_end_time   = time.time()

    status['num_frames']   += logs0["num_frames"] + logs1["num_frames"]
    status['num_frames0']  += logs0["num_frames"]
    status['num_frames1']  += logs1["num_frames"]
    status['num_episodes'] += logs0['episodes_done']
    status['i'] += 1

    # Print logs

    if status['i'] % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)
        fps0                = logs0["num_frames"] / (update_end_time - update_start_time)
        fps1                = logs1["num_frames"] / (update_end_time - update_start_time)
        duration            = datetime.timedelta(seconds=total_ellapsed_time)
        
        return_per_episode0     = utils.synthesize(logs0["return_per_episode"])
        success_per_episode0    = utils.synthesize(
            [1 if r > 0 else 0 for r in logs0["return_per_episode"]])
        
        return_per_episode1     = utils.synthesize(logs1["return_per_episode"])
        success_per_episode1    = utils.synthesize(
            [1 if r > 0 else 0 for r in logs1["return_per_episode"]])
            
        num_frames_per_episode0 = utils.synthesize(logs0["num_frames_per_episode"])
        
        num_frames_per_episode1 = utils.synthesize(logs1["num_frames_per_episode"])

        data0 = [status['i'], status['num_episodes'], status['num_frames0'],
                 fps0, total_ellapsed_time,
                 *return_per_episode0.values(),
                 success_per_episode0['mean'],
                 *num_frames_per_episode0.values(),
                 logs0["entropy"], logs0["value"], logs0["policy_loss"], logs0["value_loss"],
                 logs0["loss"], logs0["grad_norm"]]
        
        data1 = [status['i'], status['num_episodes'], status['num_frames1'],
                 fps1, total_ellapsed_time,
                 *return_per_episode1.values(),
                 success_per_episode1['mean'],
                 *num_frames_per_episode1.values(),
                 logs1["entropy"], logs1["value"], logs1["policy_loss"], logs1["value_loss"],
                 logs1["loss"], logs1["grad_norm"]]

        format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                      "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

        logger0.info(format_str.format(*data0))
        if args.tb:
            assert len(header) == len(data0)
            for key, value in zip(header, data0):
                writer0.add_scalar(key, float(value), status['num_frames'])
    
        format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                      "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")
    
        logger1.info(format_str.format(*data1))
        if args.tb:
            assert len(header) == len(data1)
            for key, value in zip(header, data1):
                writer1.add_scalar(key, float(value), status['num_frames'])

    if args.gen - 1 < len(algo.buffer):
        np.save('data.npy', algo.buffer)
        break