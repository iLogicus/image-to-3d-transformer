import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset.shapenet import ImageDataset, load_points
from metrics import chamfer_fscore, chamfer_loss
from models.rgb2point import build_model, unfreeze_last_blocks
from utils import finished_run, get_device, load_config, make_generator, print_epoch, seed_worker, set_seed


# V3: start from V2 (frozen DINOv2 + trained decoder), then train the last ViT blocks together
# with the decoder on augmented images. The ViT changes, so features cannot be cached:
# every step reads the images and runs the full model.
@torch.no_grad()
def validate(model, loader, points, device):
    model.eval()
    cd_all, f_all, class_all = [], [], []
    for images, obj_idx in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(images.to(device))
        cd, f = chamfer_fscore(pred.float(), points[obj_idx.to(device)])
        cd_all.append(cd.cpu())
        f_all.append(f.cpu())
        class_all += [loader.dataset.objects[i][0] for i in obj_idx.tolist()]
    cd_all, f_all = torch.cat(cd_all), torch.cat(f_all)
    results = {}
    for class_id in sorted(set(class_all)):
        mask = torch.tensor([c == class_id for c in class_all])
        results[class_id] = {"cd": cd_all[mask].mean().item(), "fscore": f_all[mask].mean().item()}
    results["mean"] = {k: sum(r[k] for r in results.values()) / len(results) for k in ["cd", "fscore"]}
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

    train_set = ImageDataset(data_cfg, "train", augment=True)
    val_set = ImageDataset(data_cfg, "val")
    train_loader = DataLoader(train_set, batch_size=data_cfg["batch_size"], shuffle=True, drop_last=True,
                              num_workers=data_cfg["num_workers"], worker_init_fn=seed_worker,
                              generator=make_generator(cfg["seed"]), pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=data_cfg["num_workers"])
    train_points = load_points(data_cfg, train_set.objects).to(device)
    val_points = load_points(data_cfg, val_set.objects).to(device)

    # Start from the V2 decoder, then unfreeze the last ViT blocks
    model = build_model(model_cfg, data_cfg["num_points"]).to(device)
    model.head.load_state_dict(torch.load(train_cfg["init_head"], map_location=device)["head"])
    vit_params = unfreeze_last_blocks(model.vit, model_cfg["finetune_blocks"])
    print("Trainable params:", sum(p.numel() for p in vit_params) + sum(p.numel() for p in model.head.parameters()))

    # Small learning rate for the pretrained ViT blocks, larger for the decoder
    optimizer = torch.optim.Adam([
        {"params": vit_params, "lr": train_cfg["vit_lr"]},
        {"params": model.head.parameters(), "lr": train_cfg["lr"]},
    ], fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    # Resume from the last epoch if the session was disconnected
    start_epoch, best_cd = 0, float("inf")
    last_path = os.path.join(out_dir, "last.pt")
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu")
        model.head.load_state_dict(ckpt["head"])
        model.vit.load_state_dict(ckpt["vit"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        train_loader.generator.set_state(ckpt["generator"])
        start_epoch, best_cd = ckpt["epoch"] + 1, ckpt["best_cd"]
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        total_loss, steps = 0.0, 0
        for images, obj_idx in train_loader:
            gt = train_points[obj_idx.to(device)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(images.to(device, non_blocking=True))
            loss = chamfer_loss(pred, gt) * train_cfg["loss_scale"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vit_params + list(model.head.parameters()), train_cfg["grad_clip"])
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        scheduler.step()

        val = validate(model, val_loader, val_points, device)
        val_cd = val["mean"]["cd"]
        log = {
            "epoch": epoch + 1,
            "train_loss": total_loss / steps,
            "val_cd": val_cd,
            "val_fscore": val["mean"]["fscore"],
            "val_per_class": val,
            "lr": optimizer.param_groups[1]["lr"],
            "time_s": time.time() - t0,
        }
        with open(os.path.join(out_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(log) + "\n")
        print_epoch(log, epochs)

        if val_cd < best_cd:
            best_cd = val_cd
            torch.save({"head": model.head.state_dict(), "vit": model.vit.state_dict(), "epoch": epoch, "val": val},
                       os.path.join(out_dir, "best.pt"))
        torch.save({
            "head": model.head.state_dict(),
            "vit": model.vit.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "generator": train_loader.generator.get_state(),
            "epoch": epoch,
            "best_cd": best_cd,
        }, last_path)


if __name__ == "__main__":
    main()
