import argparse
import itertools
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.shapenet import TRANSFORM, load_points, read_split
from evaluate import load_head, load_vit
from metrics import chamfer_fscore
from models.rgb2point import build_model
from utils import get_device, load_config, set_seed

# Real-image test: Pix3D chairs (the only Pix3D class among our 3 classes).
# Ground truth point clouds from 3D-LMNet (pix3d_pointclouds.zip), one per CAD model.
FILL = 0.7  # the object takes 70% of the image side, as in the ShapeNet renderings


def all_rotations():
    # The 24 rotations that map the x / y / z axes onto each other (det = +1)
    rotations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            r = np.zeros((3, 3), dtype=np.float32)
            for row in range(3):
                r[row, perm[row]] = signs[row]
            if np.linalg.det(r) > 0:
                rotations.append(r)
    return rotations


def choose_rotation(pix3d_points, shapenet_points):
    # Pix3D models use another axis convention than our ShapeNet data. Choose the rotation that
    # makes Pix3D chairs closest to ShapeNet chairs (ground truth only, no model involved).
    # 256 of the 1024 points are enough to compare the axes, and much faster.
    pix3d_points, shapenet_points = pix3d_points[:, ::4], shapenet_points[:, ::4]
    best, best_cd = None, float("inf")
    for r in all_rotations():
        rotated = pix3d_points @ torch.tensor(r, device=pix3d_points.device).T
        cd = []
        for p in rotated:
            d, _ = chamfer_fscore(p.expand(len(shapenet_points), -1, -1), shapenet_points)
            cd.append(d.min().item())  # nearest ShapeNet chair
        if np.mean(cd) < best_cd:
            best, best_cd = r, np.mean(cd)
    print("Chosen rotation (Pix3D -> ShapeNet axes):\n", best, "\nmean CD to nearest ShapeNet chair:", round(best_cd * 100, 3))
    return torch.tensor(best)


def load_image(pix3d_root, item, background):
    # Keep only the object (mask), put it on the same background as the renderings,
    # and crop a square around it so it fills the image like in the renderings.
    image = Image.open(os.path.join(pix3d_root, item["img"])).convert("RGB")
    mask = Image.open(os.path.join(pix3d_root, item["mask"])).convert("L")
    canvas = Image.new("RGB", image.size, background)
    canvas.paste(image, mask=mask)
    x0, y0, x1, y1 = item["bbox"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = int(max(x1 - x0, y1 - y0) / 2 / FILL)
    # Add a border of background colour, so the square crop never leaves the image
    padded = Image.new("RGB", (image.width + 2 * half, image.height + 2 * half), background)
    padded.paste(canvas, (half, half))
    cx, cy = int(cx) + half, int(cy) + half
    return padded.crop((cx - half, cy - half, cx + half, cy + half))


class Pix3DImages(Dataset):
    # Lets the DataLoader prepare the (large) Pix3D photos in several workers
    def __init__(self, pix3d_root, items, background):
        self.pix3d_root, self.items, self.background = pix3d_root, items, background

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return TRANSFORM(load_image(self.pix3d_root, self.items[idx], self.background)), idx


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--pix3d_root", default="data/raw/pix3d")
    parser.add_argument("--points_root", default="data/raw/pix3d_pointclouds/model/chair")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    data_cfg = cfg["data"]

    # Images: chairs, not truncated, not occluded, with a ground truth point cloud
    items = json.load(open(os.path.join(args.pix3d_root, "pix3d.json")))
    items = [it for it in items if it["category"] == "chair" and not it["truncated"] and not it["occluded"]]
    gt_path = lambda it: os.path.join(args.points_root, it["model"].split("/")[2], "pcl_1024.npy")
    items = [it for it in items if os.path.exists(gt_path(it))]
    print("Pix3D chair images:", len(items))

    # Ground truth: centered (as ShapeNet), then rotated to the ShapeNet axes
    models = sorted({gt_path(it) for it in items})
    points = torch.from_numpy(np.stack([np.load(p) for p in models])).float()
    points = points - points.mean(1, keepdim=True)
    shapenet_chairs = load_points(data_cfg, read_split(data_cfg["split"], ["chair"], "train")[:300])
    rotation = choose_rotation(points[:50].to(device), shapenet_chairs.to(device))
    points = points.to(device) @ rotation.to(device).T
    model_index = {p: i for i, p in enumerate(models)}

    model = build_model(cfg["model"], data_cfg["num_points"]).to(device).eval()
    load_head(model.head, args.ckpt, args.official)
    load_vit(model.vit, args.ckpt)

    # Background colour of the renderings after convert("RGB") (transparent pixels)
    first = read_split(data_cfg["split"], ["chair"], "test")[0]
    render = np.array(Image.open(os.path.join(data_cfg["image_root"], first[0], first[1], "rendering", "00.png")))
    background = tuple(int(v) for v in render[0, 0, :3])

    gt_index = torch.tensor([model_index[gt_path(it)] for it in items])
    loader = DataLoader(Pix3DImages(args.pix3d_root, items, background), batch_size=64, num_workers=8)
    cds, fs = [], []
    for images, idx in tqdm(loader):
        gt = points[gt_index[idx].to(device)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(images.to(device))
        cd, f = chamfer_fscore(pred.float(), gt)
        cds.append(cd.cpu())
        fs.append(f.cpu())
    results = {"chair": {"cd": torch.cat(cds).mean().item(), "fscore": torch.cat(fs).mean().item()},
               "num_images": len(items)}
    print(f"Pix3D chair: CD x100 {results['chair']['cd'] * 100:.3f}  F@0.01 {results['chair']['fscore']:.4f}")
    with open(os.path.join(cfg["output_dir"], "pix3d.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
