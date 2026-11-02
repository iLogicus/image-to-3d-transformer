import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset.shapenet import VoxelDataset
from metrics import dice_loss, voxel_iou
from models.retr import RETR
from utils import finished_run, get_device, load_config, make_generator, print_epoch, seed_worker, set_seed


# Baseline 2: 3D-RETR-S. The whole model (DeiT encoder + decoder) is trained end to end with Dice loss,
# AdamW (lr 1e-4, weight decay 1e-2), constant lr, batch 16 (as the official code).
@torch.no_grad()
def validate(model, loader, device, threshold):
    # Mean IoU per class, then the mean over classes (as get_class_iou in the official code)
    model.eval()
    iou_all, class_all = [], []
    for images, voxels, obj_idx in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images.to(device))
        iou_all.append(voxel_iou(logits, voxels.to(device), threshold).cpu())
        class_all += [loader.dataset.objects[i][0] for i in obj_idx.tolist()]
    iou_all = torch.cat(iou_all)
    results = {}
    for class_id in sorted(set(class_all)):
        mask = torch.tensor([c == class_id for c in class_all])
        results[class_id] = {"iou": iou_all[mask].mean().item()}
    results["mean"] = {"iou": sum(r["iou"] for r in results.values()) / len(results)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override config (e.g. 1 to time an epoch)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]
    epochs = args.epochs or train_cfg["epochs"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    if finished_run(out_dir, epochs):
        return

    train_set = VoxelDataset(data_cfg, "train", augment=data_cfg["augment"])
    val_set = VoxelDataset(data_cfg, "val")
    train_loader = DataLoader(train_set, batch_size=data_cfg["batch_size"], shuffle=True,
                              num_workers=data_cfg["num_workers"], worker_init_fn=seed_worker,
                              generator=make_generator(cfg["seed"]), pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=data_cfg["num_workers"])

    model = RETR(model_cfg).to(device)
    print("Params:", sum(p.numel() for p in model.parameters()))
    # Encoder and decoder can have different learning rates (baseline: both 1e-4, V1: encoder 1e-5)
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": train_cfg["encoder_lr"]},
        {"params": decoder_params, "lr": train_cfg["lr"]},
    ], weight_decay=train_cfg["weight_decay"], fused=True)
    # Baseline / V1 / V2: constant lr (official). V3: cosine decay to 0 over the epochs.
    if train_cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)

    # Resume from the last epoch if the session was disconnected
    start_epoch, best_iou = 0, 0.0
    last_path = os.path.join(out_dir, "last.pt")
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        train_loader.generator.set_state(ckpt["generator"])
        start_epoch, best_iou = ckpt["epoch"] + 1, ckpt["best_iou"]
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        total_loss, steps = 0.0, 0
        for images, voxels, _ in train_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images.to(device, non_blocking=True))
            loss = dice_loss(logits, voxels.to(device, non_blocking=True))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        scheduler.step()

        val = validate(model, val_loader, device, model_cfg["threshold"])
        val_iou = val["mean"]["iou"]
        log = {
            "epoch": epoch + 1,
            "train_loss": total_loss / steps,
            "val_iou": val_iou,
            "val_per_class": val,
            "lr": optimizer.param_groups[1]["lr"],
            "time_s": time.time() - t0,
        }
        with open(os.path.join(out_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(log) + "\n")
        print_epoch(log, epochs)

        # Keep the checkpoint with the best val IoU (official: ModelCheckpoint(monitor="val_iou_mean"))
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, os.path.join(out_dir, "best.pt"))
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "generator": train_loader.generator.get_state(),
            "epoch": epoch,
            "best_iou": best_iou,
        }, last_path)


if __name__ == "__main__":
    main()
