import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.shapenet import CLASS_NAMES, ImageDataset, load_points, view_angles
from evaluate import load_head
from metrics import chamfer_fscore
from models.rgb2point import build_model
from utils import get_device, load_config, set_seed


# Error analysis: run the full model on EVERY view of the test objects (not only view 00)
# and save one row per image, so we can group the errors by class, object and viewpoint.
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    data_cfg = cfg["data"]

    model = build_model(cfg["model"], data_cfg["num_points"]).to(device).eval()
    load_head(model.head, args.ckpt, args.official)

    dataset = ImageDataset(data_cfg, args.split, all_views=True)
    points = load_points(data_cfg, dataset.objects).to(device)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=data_cfg["num_workers"])

    rows = []
    i = 0
    for images, obj_idx in tqdm(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(images.to(device))
        cd, fscore = chamfer_fscore(pred.float(), points[obj_idx.to(device)])
        for k in range(len(images)):
            o, view = dataset.samples[i]
            class_id, obj = dataset.objects[o]
            azimuth, elevation = view_angles(data_cfg, class_id, obj)[view]
            rows.append([CLASS_NAMES[class_id], obj, view, azimuth, elevation, cd[k].item(), fscore[k].item()])
            i += 1

    path = os.path.join(cfg["output_dir"], f"analysis_{args.split}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "object", "view", "azimuth", "elevation", "cd", "fscore"])
        writer.writerows(rows)
    print("Saved", len(rows), "rows to", path)


if __name__ == "__main__":
    main()
