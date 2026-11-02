import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.shapenet import CLASS_NAMES, VoxelDataset, view_angles
from models.retr import RETR
from utils import get_device, load_config, set_seed

THRESHOLDS = [0.2, 0.3, 0.4, 0.5]


# Error analysis of baseline 2: run the model on EVERY view of the objects (not only view 00)
# and save one row per image: IoU at several thresholds, precision / recall of the occupied voxels,
# number of occupied voxels, viewpoint.
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    data_cfg = cfg["data"]

    model = RETR(cfg["model"], pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])

    dataset = VoxelDataset(data_cfg, args.split, all_views=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=data_cfg["num_workers"])

    rows = []
    i = 0
    for images, voxels, _ in tqdm(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prob = torch.sigmoid(model(images.to(device)).float()).flatten(1)
        gt = voxels.to(device).flatten(1) > 0.5
        ious = []
        for t in THRESHOLDS:
            pred = prob > t
            ious.append((pred & gt).sum(1) / (pred | gt).sum(1))
        pred = prob > 0.5
        inter = (pred & gt).sum(1).float()
        precision = inter / pred.sum(1).clamp(min=1)  # predicted voxels that are correct
        recall = inter / gt.sum(1)                     # true voxels that are found
        for k in range(len(images)):
            o, view = dataset.samples[i]
            class_id, obj = dataset.objects[o]
            azimuth, elevation = view_angles(data_cfg, class_id, obj)[view]
            rows.append([CLASS_NAMES[class_id], obj, view, azimuth, elevation]
                        + [iou[k].item() for iou in ious]
                        + [precision[k].item(), recall[k].item(), gt[k].sum().item(), pred[k].sum().item()])
            i += 1

    path = os.path.join(cfg["output_dir"], f"analysis_{args.split}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "object", "view", "azimuth", "elevation"]
                        + [f"iou_{t}" for t in THRESHOLDS]
                        + ["precision", "recall", "gt_voxels", "pred_voxels"])
        writer.writerows(rows)
    print("Saved", len(rows), "rows to", path)


if __name__ == "__main__":
    main()
