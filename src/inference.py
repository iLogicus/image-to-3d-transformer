import argparse

import torch
import trimesh
from PIL import Image

from dataset.shapenet import TRANSFORM
from models.rgb2point import build_model
from utils import get_device, load_config


def load_model(cfg, ckpt_path, device):
    model = build_model(cfg["model"], cfg["data"]["num_points"]).to(device).eval()
    model.head.load_state_dict(torch.load(ckpt_path, map_location=device)["head"])
    return model


@torch.no_grad()
def predict(image, model, device):
    # PIL image -> point cloud (N, 3) numpy array
    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    return model(x)[0].float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rgb2point.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="outputs/result.ply")
    args = parser.parse_args()

    device = get_device()
    model = load_model(load_config(args.config), args.ckpt, device)
    points = predict(Image.open(args.image), model, device)
    trimesh.PointCloud(points).export(args.out)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
