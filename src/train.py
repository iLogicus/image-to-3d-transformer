import argparse
import json
import os
import time

import torch

from dataset.shapenet import load_cache
from metrics import chamfer_loss, evaluate_head
from models.rgb2point import build_head
from utils import get_device, load_config, make_generator, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override config (e.g. 1 to time an epoch)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    print("Device:", device)
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]
    epochs = args.epochs or train_cfg["epochs"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Training uses cached ViT features (ViT is frozen), so only the head is trained
    train_data = load_cache(data_cfg, "train", device)
    val_data = load_cache(data_cfg, "val", device)
    head = build_head(model_cfg, train_data["features"].shape[-1], data_cfg["num_points"]).to(device)
    print("Trainable params:", sum(p.numel() for p in head.parameters()))

    optimizer = torch.optim.Adam(head.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"], fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=5, threshold=0.01, min_lr=1e-5)
    generator = make_generator(cfg["seed"])

    # Resume from the last epoch if the session was disconnected
    start_epoch, best_cd = 0, float("inf")
    last_path = os.path.join(out_dir, "last.pt")
    if os.path.exists(last_path):
        ckpt = torch.load(last_path)
        head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        generator.set_state(ckpt["generator"])
        start_epoch, best_cd = ckpt["epoch"] + 1, ckpt["best_cd"]
        print(f"Resumed from epoch {start_epoch}")

    batch_size = data_cfg["batch_size"]
    num_train = len(train_data["features"])
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        head.train()
        total_loss, steps = 0.0, 0
        perm = torch.randperm(num_train, generator=generator)
        for i in range(0, num_train - batch_size + 1, batch_size):  # drop_last=True
            idx = perm[i:i + batch_size]
            feat = train_data["features"][idx].to(device).float()
            gt = train_data["points"][train_data["obj_index"][idx].to(device)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = head(feat)
            loss = chamfer_loss(pred, gt) * train_cfg["loss_scale"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        # Validation (view 00), choose the checkpoint by val CD (lower is better)
        val = evaluate_head(head, val_data)
        val_cd = val["mean"]["cd"]
        scheduler.step(val_cd)
        log = {
            "epoch": epoch + 1,
            "train_loss": total_loss / steps,
            "val_cd": val_cd,
            "val_fscore": val["mean"]["fscore"],
            "val_per_class": val,
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": time.time() - t0,
        }
        with open(os.path.join(out_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(log) + "\n")
        print(f"Epoch {epoch + 1}/{epochs}  loss {log['train_loss']:.4f}  val CD {val_cd:.5f}  "
              f"val F {log['val_fscore']:.4f}  lr {log['lr']:.1e}  {log['time_s']:.0f}s")

        if val_cd < best_cd:
            best_cd = val_cd
            torch.save({"head": head.state_dict(), "epoch": epoch, "val": val}, os.path.join(out_dir, "best.pt"))
        torch.save({
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "generator": generator.get_state(),
            "epoch": epoch,
            "best_cd": best_cd,
        }, last_path)


if __name__ == "__main__":
    main()
