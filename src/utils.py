import random

import numpy as np
import torch
import yaml


def set_seed(seed):
    # Same as pl.seed_everything(seed) + Trainer(deterministic=True) in 3D-RETR
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # Pass to DataLoader(worker_init_fn=...) so each worker has a fixed seed
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed):
    # Pass to DataLoader(generator=...) so the shuffle order is fixed
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
