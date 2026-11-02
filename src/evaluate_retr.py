import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset.shapenet import CLASS_NAMES, VoxelDataset
from models.retr import RETR
from train_retr import validate
from utils import get_device, load_config, set_seed


# Baseline 2: test IoU per class (view 00 of each object, as the official eval.py mode="first"),
# number of parameters, inference time and GPU memory
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--name", default="test_results", help="name of the output json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    data_cfg = cfg["data"]

    model = RETR(cfg["model"], pretrained=False).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print("Checkpoint epoch:", ckpt["epoch"] + 1)

    dataset = VoxelDataset(data_cfg, args.split)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=data_cfg["num_workers"])
    results = validate(model, loader, device, cfg["model"]["threshold"])
    print(f"{'class':<10}{'IoU':>8}")
    for key, r in results.items():
        print(f"{CLASS_NAMES.get(key, key):<10}{r['iou']:>8.4f}")

    # Model size (every parameter is trained)
    results["params_total"] = sum(p.numel() for p in model.parameters())

    # Inference time (image -> voxels), batch 1, 200 images after 10 warm-up images
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad():
        for i in range(210):
            image = dataset[i][0].unsqueeze(0).to(device)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(image)
            torch.cuda.synchronize()
            if i >= 10:
                times.append(time.time() - t0)
    results["time_ms"] = 1000 * sum(times) / len(times)
    results["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(f"Params: {results['params_total'] / 1e6:.2f}M")
    print(f"Inference: {results['time_ms']:.1f} ms/image, GPU memory {results['gpu_mem_gb']:.2f} GB")

    with open(os.path.join(cfg["output_dir"], f"{args.name}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
