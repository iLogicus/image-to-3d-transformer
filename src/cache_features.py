import argparse
import os

import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.shapenet import ImageDataset, feature_path
from models.rgb2point import image_tokens
from utils import get_device, load_config, set_seed


# The ViT is frozen and there is no augmentation, so its output for each image never changes.
# We compute it once and save it; training then only runs the small head (much faster).
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    data_cfg = cfg["data"]
    os.makedirs(data_cfg["feature_dir"], exist_ok=True)

    vit = timm.create_model(cfg["model"]["backbone"], pretrained=True, num_classes=0).to(device).eval()

    for stage in ["train", "val", "test"]:
        if os.path.exists(feature_path(data_cfg, stage)):
            print(f"[{stage}] already cached")
            continue
        dataset = ImageDataset(data_cfg, stage)
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=data_cfg["num_workers"])
        features, obj_index = None, []
        start = 0
        for images, idx in tqdm(loader, desc=stage):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feat = image_tokens(vit, images.to(device), cfg["model"]["tokens"]).half().cpu()
            if features is None:
                # V1 features are large (about 18 GB for train), so fill one tensor instead of a list
                features = torch.empty((len(dataset),) + feat.shape[1:], dtype=torch.float16)
            features[start:start + len(feat)] = feat
            start += len(feat)
            obj_index.append(idx)
        torch.save({
            "features": features,
            "obj_index": torch.cat(obj_index),
            "objects": dataset.objects,
        }, feature_path(data_cfg, stage))


if __name__ == "__main__":
    main()
