import argparse
import os

import torch

from dataset.shapenet import TRANSFORM
from evaluate import load_head, load_vit
from models.retr import load_trained
from models.rgb2point import build_model
from utils import load_config

# Run once on Colab after training: one small weight file per demo model (model only, fp16, no optimizer),
# with everything the demo needs to rebuild it (config, image normalization). Upload outputs/demo/ to the
# Hugging Face Hub so the Docker image / Space can download it.
DEMO_RUNS = {
    "b1_v3": "configs/v3_finetune.yaml",
    "rgb2point": "configs/rgb2point.yaml",
    "b2_v4": "configs/retr_v4_augment.yaml",
    "retr_s": "configs/retr_s.yaml",
}


def make_info(kind, model, model_cfg, mean, std, num_points=None, threshold=0.5):
    # Same format is read by demo.build_from_file
    return {
        "kind": kind,
        "model_cfg": model_cfg,
        "num_points": num_points,
        "mean": list(mean),
        "std": list(std),
        "threshold": threshold,
        "state": {k: v.half() for k, v in model.state_dict().items()},
    }


def export(name, cfg):
    if "voxel_root" in cfg["data"]:  # baseline 2
        model = load_trained(cfg, "cpu")
        return make_info("voxels", model, cfg["model"], cfg["data"]["mean"], cfg["data"]["std"],
                         threshold=cfg["model"]["threshold"])
    ckpt = os.path.join(cfg["output_dir"], "best.pt")
    model = build_model(cfg["model"], cfg["data"]["num_points"]).eval()  # pretrained ViT (frozen runs)
    load_head(model.head, ckpt, official=False)
    load_vit(model.vit, ckpt)  # V3: fine-tuned ViT
    norm = TRANSFORM.transforms[-1]  # ImageNet mean / std, as training
    return make_info("points", model, cfg["model"], norm.mean, norm.std, num_points=cfg["data"]["num_points"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/demo")
    parser.add_argument("--repo", default=None, help="Hugging Face model repo (user/name) to upload the files to")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for name, config in DEMO_RUNS.items():
        path = os.path.join(args.out_dir, name + ".pt")
        torch.save(export(name, load_config(config)), path)
        print(f"{name}: {os.path.getsize(path) / 1e6:.0f} MB")
    if args.repo:  # needs a write token: huggingface-cli login, or HF_TOKEN in the Colab secrets
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, exist_ok=True)
        api.upload_folder(folder_path=args.out_dir, repo_id=args.repo)
        print("Uploaded to https://huggingface.co/" + args.repo)


if __name__ == "__main__":
    main()
