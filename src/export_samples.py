import argparse
import io
import os

import numpy as np
import torch
from PIL import Image

from dataset.shapenet import CLASS_IDS, image_path, load_points, points_path, read_binvox, read_split, voxel_path
from utils import load_config

# Run once on Colab: a few test objects per class for the "Test set" tab of the demo.
# Each sample: rendered view 00 (as official test), ground truth points (3D-LMNet) and voxels (R2N2).
# Saved as one small file next to the demo weights (same Hugging Face repo).
SAMPLES_PER_CLASS = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retr_s.yaml")  # has image, voxel and point paths
    parser.add_argument("--out", default="outputs/demo/samples.pt")
    args = parser.parse_args()
    data_cfg = load_config(args.config)["data"]
    rng = np.random.default_rng(0)
    samples = []
    for name in CLASS_IDS:
        objects = [(c, o) for c, o in read_split(data_cfg["split"], [name], "test")
                   if os.path.exists(image_path(data_cfg, c, o, 0)) and os.path.exists(voxel_path(data_cfg, c, o))
                   and os.path.exists(points_path(data_cfg, c, o))]
        for i in sorted(rng.choice(len(objects), SAMPLES_PER_CLASS, replace=False)):
            class_id, obj = objects[i]
            buffer = io.BytesIO()
            Image.open(image_path(data_cfg, class_id, obj, 0)).save(buffer, format="PNG")  # RGBA render
            samples.append({
                "class": name,
                "object": obj,
                "image": buffer.getvalue(),
                "points": load_points(data_cfg, [(class_id, obj)])[0],  # (1024, 3), centered as in training
                "voxels": torch.from_numpy(read_binvox(voxel_path(data_cfg, class_id, obj))).bool(),
            })
    torch.save(samples, args.out)
    print(f"{len(samples)} samples, {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
