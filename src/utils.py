import json
import os
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


def print_epoch(log, epochs):
    print(f"Epoch {log['epoch']}/{epochs}  loss {log['train_loss']:.4f}  val CD {log['val_cd']:.5f}  "
          f"val F {log['val_fscore']:.4f}  lr {log['lr']:.1e}  {log['time_s']:.0f}s")


def finished_run(out_dir, epochs):
    # If the run already trained all epochs (e.g. the notebook is run again), print its log and return True
    last_path = os.path.join(out_dir, "last.pt")
    if not os.path.exists(last_path) or torch.load(last_path, map_location="cpu")["epoch"] + 1 < epochs:
        return False
    print("Training already finished, log:")
    for line in open(os.path.join(out_dir, "log.jsonl")):
        print_epoch(json.loads(line), epochs)
    return True
