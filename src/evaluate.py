import argparse
import json
import os
import time

import torch

from dataset.shapenet import CLASS_NAMES, ImageDataset, load_cache
from metrics import evaluate_head
from models.rgb2point import build_model
from utils import get_device, load_config, set_seed

# Layer names in the official checkpoint (model.py of RGB2point) -> names in our PointHead
OFFICIAL_NAMES = {
    "aggregator.": "aggregator.",
    "point_cloud_generator.self_attention.": "attention.",
    "point_cloud_generator.linear_layers.": "mlp.",
}


def load_head(head, ckpt_path, official):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not official:
        head.load_state_dict(ckpt["head"])
        print("Checkpoint epoch:", ckpt["epoch"] + 1)
        return
    state = {}
    for key, value in ckpt["model"].items():
        for old, new in OFFICIAL_NAMES.items():
            if key.startswith(old):
                state[new + key[len(old):]] = value
    head.load_state_dict(state)
    print("Loaded official RGB2Point weights")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--official", action="store_true", help="ckpt is the authors' pc1024_three.pth")
    parser.add_argument("--name", default="test_results", help="name of the output json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    data_cfg = cfg["data"]

    model = build_model(cfg["model"], data_cfg["num_points"]).to(device).eval()
    load_head(model.head, args.ckpt, args.official)

    # Accuracy on cached features (same ViT output as running the full model)
    test_data = load_cache(data_cfg, args.split, device)
    results = evaluate_head(model.head, test_data, with_emd=True)
    del test_data
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"{'class':<10}{'CD x100':>10}{'EMD x100':>10}{'F@0.01':>10}")
    for key, r in results.items():
        print(f"{CLASS_NAMES.get(key, key):<10}{r['cd'] * 100:>10.3f}{r['emd'] * 100:>10.3f}{r['fscore']:>10.4f}")

    # Model size
    results["params_total"] = sum(p.numel() for p in model.parameters())
    results["params_trainable"] = sum(p.numel() for p in model.head.parameters())

    # Inference time of the full model (image -> points), batch 1, 200 images
    dataset = ImageDataset(data_cfg, args.split)
    times = []
    with torch.no_grad():
        for i in range(210):
            image = dataset[i][0].unsqueeze(0).to(device)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(image)
            torch.cuda.synchronize()
            if i >= 10:  # skip warm-up
                times.append(time.time() - t0)
    results["time_ms"] = 1000 * sum(times) / len(times)
    results["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(f"Params: {results['params_total'] / 1e6:.1f}M total, {results['params_trainable'] / 1e6:.1f}M trainable")
    print(f"Inference: {results['time_ms']:.1f} ms/image, GPU memory {results['gpu_mem_gb']:.2f} GB")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], f"{args.name}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
