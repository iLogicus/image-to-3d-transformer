import argparse

from utils import get_device, load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    print("Device:", device)

    # TODO: build dataset, model, optimizer; training loop; save best checkpoint by val metric


if __name__ == "__main__":
    main()
