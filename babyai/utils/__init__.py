import os
import random
import numpy
import torch
from babyai.utils.agent import load_agent, ModelAgent
from babyai.utils.demos import (
    load_demos, save_demos, synthesize_demos, get_demos_path)
from babyai.utils.format import ObssPreprocessor, IntObssPreprocessor, get_vocab_path
from babyai.utils.log import (
    get_log_path, get_log_dir, synthesize, configure_logging)
from babyai.utils.model import get_model_dir, load_model, save_model


def storage_dir():
    # defines the storage directory to be in the root (Same level as babyai folder)
    return os.environ.get("BABYAI_STORAGE", '.')


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not(os.path.isdir(dirname)):
        os.makedirs(dirname)


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_seed():
    if torch.cuda.is_available():
        return (random.getstate(), numpy.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
    else:
        return (random.getstate(), numpy.random.get_state(), torch.get_rng_state())

def set_seed(state):
    random.setstate(state[0])
    numpy.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state[3])
