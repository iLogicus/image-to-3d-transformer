import argparse

from utils import get_device, load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    # TODO: load model + test set, compute metrics, inference time, number of params


if __name__ == "__main__":
    main()
