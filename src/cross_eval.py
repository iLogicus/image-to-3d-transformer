import argparse
import itertools
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

from dataset.shapenet import CLASS_NAMES, VoxelDataset, load_cache, load_points, read_binvox, voxel_path
from evaluate import load_head
from metrics import chamfer_fscore
from models.retr import RETR
from models.rgb2point import build_head
from utils import get_device, load_config, set_seed

# Compare the two baselines with the same metrics:
# - voxels (baseline 2) -> 1024 points on the voxel surface -> CD / F-score against the 3D-LMNet ground truth points
# - points (baseline 1) -> filled 32^3 voxels -> IoU against the R2N2 ground truth voxels
# No information of the ground truth object is used to convert a prediction (only one axis rotation for all objects).


def surface_points(voxels, n=1024, seed=0):
    # voxels: (32, 32, 32) bool. Surface voxels = occupied voxels with an empty neighbour.
    # Sample n random points inside the surface voxels (grid units, 0..32).
    v = voxels.float()[None, None]
    inside = 1 - F.max_pool3d(1 - v, 3, stride=1, padding=1)  # occupied and all 26 neighbours occupied
    idx = torch.nonzero((v - inside)[0, 0] > 0).float()
    if len(idx) == 0:  # empty prediction: one point at the centre
        idx = torch.full((1, 3), 15.5)
    g = torch.Generator().manual_seed(seed)
    pick = torch.randint(len(idx), (n,), generator=g)
    return idx[pick] + torch.rand(n, 3, generator=g)


def normalize(points):
    # Same normalization as ShapeNet models: bounding box diagonal = 1, then centered at the mean
    diag = (points.max(0).values - points.min(0).values).norm()
    points = points / diag
    return points - points.mean(0)


def all_axes():
    # The 48 matrices that permute the axes and flip their signs (rotations and mirrors)
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            r = np.zeros((3, 3), dtype=np.float32)
            for row in range(3):
                r[row, perm[row]] = signs[row]
            out.append(torch.tensor(r))
    return out


def voxels_to_points(voxels, axes):
    # (32, 32, 32) bool -> (1024, 3) points in the axes of the 3D-LMNet ground truth
    return normalize(surface_points(voxels)) @ axes.T


def points_to_voxels(points, axes):
    # (N, 3) points -> (32, 32, 32) bool, as R2N2 voxels: the grid fits the bounding box (longest side = 32),
    # points mark their voxel, small gaps are closed, then the inside is filled
    q = points.cpu() @ axes  # back to the voxel axes
    low = q.min(0).values
    side = (q.max(0).values - low).max()
    idx = ((q - low) / side * 32).long().clamp(0, 31).numpy()
    grid = np.zeros((32, 32, 32), dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    grid = ndimage.binary_closing(grid, iterations=1, border_value=0) | grid
    return torch.from_numpy(ndimage.binary_fill_holes(grid))


def iou(a, b):
    return ((a & b).sum() / (a | b).sum()).item()


def choose_axes(gt_voxels, gt_points):
    # Choose the axis matrix that makes ground truth voxels closest to ground truth points (same objects)
    best, best_cd = None, float("inf")
    surface = torch.stack([normalize(surface_points(v)) for v in gt_voxels])
    for r in all_axes():
        cd, _ = chamfer_fscore(surface @ r.T, gt_points)
        if cd.mean() < best_cd:
            best, best_cd = r, cd.mean().item()
    print("Chosen axes (voxel -> point cloud):\n", best.numpy(), "\nGT voxels vs GT points CD x100:", round(best_cd * 100, 3))
    return best


def per_class(values, class_ids):
    out = {}
    for c in sorted(set(class_ids)):
        out[c] = float(np.mean([v for v, k in zip(values, class_ids) if k == c]))
    out["mean"] = float(np.mean([out[c] for c in out]))
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retr_configs", nargs="+", required=True, help="baseline 2 runs (best.pt in output_dir)")
    parser.add_argument("--rgb2point_configs", nargs="+", required=True, help="baseline 1 runs (frozen ViT cache)")
    parser.add_argument("--out", default="outputs/baseline2/cross_eval.json")
    args = parser.parse_args()

    set_seed(0)
    device = get_device()
    retr_cfg = load_config(args.retr_configs[0])
    b1_cfg = load_config(args.rgb2point_configs[0])

    # Test objects of baseline 1 (view 00), their ground truth points and voxels
    test = load_cache(b1_cfg["data"], "test", "cpu")
    objects = [tuple(o) for o in test["objects"]]
    class_ids = [c for c, _ in objects]
    gt_points = test["points"]
    gt_voxels = torch.stack([torch.from_numpy(read_binvox(voxel_path(retr_cfg["data"], *o)) > 0.5) for o in objects])
    axes = choose_axes(gt_voxels[::10], gt_points[::10])

    results = {}
    # Best possible scores of each conversion (ground truth converted, compared to ground truth)
    gt_as_points = torch.stack([voxels_to_points(v, axes) for v in gt_voxels])
    cd, f = chamfer_fscore(gt_as_points, gt_points)
    gt_as_voxels = [points_to_voxels(p, axes) for p in gt_points]
    results["GT voxels / GT points"] = {
        "cd": per_class(cd.tolist(), class_ids), "fscore": per_class(f.tolist(), class_ids),
        "iou": per_class([iou(a, b) for a, b in zip(gt_as_voxels, gt_voxels)], class_ids)}

    # Baseline 1 runs: predicted points; IoU after converting to voxels
    for path in args.rgb2point_configs:
        cfg = load_config(path)
        cache = load_cache(cfg["data"], "test", "cpu")
        assert [tuple(o) for o in cache["objects"]] == objects
        keep = cache["obj_index"]
        head = build_head(cfg["model"], cache["features"].shape[-1], cfg["data"]["num_points"]).to(device)
        load_head(head, os.path.join(cfg["output_dir"], "best.pt"), False)
        head.eval()
        pred = torch.cat([head(cache["features"][i:i + 256].to(device).float()).cpu()
                          for i in range(0, len(keep), 256)])
        cd, f = chamfer_fscore(pred, gt_points[keep])
        ious = [iou(points_to_voxels(p, axes), gt_voxels[k]) for p, k in zip(pred, keep.tolist())]
        ids = [class_ids[k] for k in keep.tolist()]
        results[cfg["model"]["name"]] = {"cd": per_class(cd.tolist(), ids), "fscore": per_class(f.tolist(), ids),
                                         "iou": per_class(ious, ids)}
        print(cfg["model"]["name"], {k: round(v["mean"], 4) for k, v in results[cfg["model"]["name"]].items()})

    # Baseline 2 runs: predicted voxels; CD / F-score after converting to points
    for path in args.retr_configs:
        cfg = load_config(path)
        model = RETR(cfg["model"], pretrained=False).to(device).eval()
        model.load_state_dict(torch.load(os.path.join(cfg["output_dir"], "best.pt"), map_location="cpu")["model"])
        dataset = VoxelDataset(cfg["data"], "test")
        index = {o: i for i, o in enumerate(dataset.objects)}
        rows = [index[o] for o in objects]  # same objects and order as baseline 1
        pred = []
        for images, _, _ in DataLoader(torch.utils.data.Subset(dataset, rows), batch_size=64, num_workers=8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred.append((torch.sigmoid(model(images.to(device)).float()) > 0.5).cpu())
        pred = torch.cat(pred)
        points = torch.stack([voxels_to_points(v, axes) for v in pred])
        cd, f = chamfer_fscore(points, gt_points)
        results[cfg["model"]["name"]] = {"cd": per_class(cd.tolist(), class_ids), "fscore": per_class(f.tolist(), class_ids),
                                         "iou": per_class([iou(a, b) for a, b in zip(pred, gt_voxels)], class_ids)}
        print(cfg["model"]["name"], {k: round(v["mean"], 4) for k, v in results[cfg["model"]["name"]].items()})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
