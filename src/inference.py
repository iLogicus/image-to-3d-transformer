import argparse

from utils import get_device


def predict(image, model, device):
    # TODO: preprocess image -> model -> point cloud (N, 3)
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="outputs/result.glb")
    args = parser.parse_args()

    device = get_device()
    # TODO: load model from ckpt, run predict(), save with trimesh (.glb for Gradio, .ply for download)


if __name__ == "__main__":
    main()
