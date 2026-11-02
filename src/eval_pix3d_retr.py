import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cross_eval import choose_axes, iou, points_to_voxels, voxels_to_points
from dataset.shapenet import load_points, read_binvox, read_split, retr_transform, voxel_path
from eval_pix3d import choose_rotation, load_image
from metrics import chamfer_fscore
from models.retr import RETR
from utils import get_device, load_config, set_seed

# Real-image test of baseline 2 on Pix3D chairs (same images and ground truth as baseline 1, eval_pix3d.py).
# Predicted voxels -> 1024 surface points -> CD / F-score. IoU against the voxelized ground truth points
# (approximate: Pix3D has no R2N2 voxels).


class Pix3DImages(Dataset):
    def __init__(self, pix3d_root, items, background, transform):
        self.pix3d_root, self.items, self.background, self.transform = pix3d_root, items, background, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # Square crop (object = 70% of the side), resized to 137 x 137 like the renderings, then the 3D-RETR transform
        image = load_image(self.pix3d_root, self.items[idx], self.background).resize((137, 137))
        return self.transform(image), idx


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pix3d_root", default="data/raw/pix3d")
    parser.add_argument("--points_root", default="data/raw/pix3d_pointclouds/model/chair")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    data_cfg = cfg["data"]

    items = json.load(open(os.path.join(args.pix3d_root, "pix3d.json")))
    items = [it for it in items if it["category"] == "chair" and not it["truncated"] and not it["occluded"]]
    gt_path = lambda it: os.path.join(args.points_root, it["model"].split("/")[2], "pcl_1024.npy")
    items = [it for it in items if os.path.exists(gt_path(it))]
    print("Pix3D chair images:", len(items))

    # Ground truth points in the ShapeNet axes (same as baseline 1)
    models = sorted({gt_path(it) for it in items})
    points = torch.from_numpy(np.stack([np.load(p) for p in models])).float()
    points = points - points.mean(1, keepdim=True)
    train_chairs = read_split(data_cfg["split"], ["chair"], "train")[:300]
    rotation = choose_rotation(points[:50].to(device), load_points(data_cfg, train_chairs).to(device))
    points = (points.to(device) @ rotation.to(device).T).cpu()
    model_index = {p: i for i, p in enumerate(models)}

    # Axes from the voxel grid to the point clouds (ShapeNet ground truth only)
    chairs = read_split(data_cfg["split"], ["chair"], "train")[:200]
    gt_voxels = torch.stack([torch.from_numpy(read_binvox(voxel_path(data_cfg, *o)) > 0.5) for o in chairs])
    axes = choose_axes(gt_voxels, load_points(data_cfg, chairs))

    model = RETR(cfg["model"], pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(os.path.join(cfg["output_dir"], "best.pt"), map_location="cpu")["model"])

    background = (data_cfg["background"],) * 3
    gt_index = torch.tensor([model_index[gt_path(it)] for it in items])
    gt_voxels = [points_to_voxels(p, axes) for p in points]
    loader = DataLoader(Pix3DImages(args.pix3d_root, items, background, retr_transform(data_cfg)),
                        batch_size=64, num_workers=8)
    cds, fs, ious = [], [], []
    for images, idx in tqdm(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = (torch.sigmoid(model(images.to(device)).float()) > 0.5).cpu()
        pred_points = torch.stack([voxels_to_points(v, axes) for v in pred])
        cd, f = chamfer_fscore(pred_points, points[gt_index[idx]])
        cds.append(cd)
        fs.append(f)
        ious += [iou(v, gt_voxels[k]) for v, k in zip(pred, gt_index[idx].tolist())]
    results = {"chair": {"cd": torch.cat(cds).mean().item(), "fscore": torch.cat(fs).mean().item(),
                         "iou": float(np.mean(ious))}, "num_images": len(items)}
    r = results["chair"]
    print(f"Pix3D chair: CD x100 {r['cd'] * 100:.3f}  F@0.01 {r['fscore']:.4f}  IoU {r['iou']:.4f}")
    with open(os.path.join(cfg["output_dir"], "pix3d.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
