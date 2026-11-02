import argparse
import os

import torch
import trimesh
from PIL import Image

from dataset.shapenet import TRANSFORM
from models.rgb2point import build_model


def load_model(cfg, ckpt_path, device):
    model = build_model(cfg["model"], cfg["data"]["num_points"]).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.head.load_state_dict(ckpt["head"])
    if "vit" in ckpt:  # V3: fine-tuned ViT
        model.vit.load_state_dict(ckpt["vit"])
    return model


@torch.no_grad()
def predict(image, model, device):
    # PIL image -> point cloud (N, 3) numpy array
    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    return model(x)[0].float().cpu().numpy()


def main():
    # Command line reconstruction with the demo weights (downloaded from the Hugging Face Hub, no Colab needed):
    # python src/inference.py --image chair.jpg --model b1_v3   ->  outputs/inference/chair_b1_v3.ply (+ .glb)
    from demo import MODELS, reconstruct, save_result

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="b1_v3", choices=list(MODELS))
    parser.add_argument("--keep_background", action="store_true", help="do not remove the background")
    parser.add_argument("--out_dir", default="outputs/inference")
    args = parser.parse_args()

    result = reconstruct(Image.open(args.image), args.model, not args.keep_background)
    stem = os.path.splitext(os.path.basename(args.image))[0] + "_" + args.model
    viewer, download = save_result(result, args.out_dir, stem)
    print(f"{MODELS[args.model]}: {result['note']}, {result['time_ms']:.0f} ms")
    print("Saved", download, "and", viewer)


if __name__ == "__main__":
    main()
