import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def nearest_dist(pred, gt):
    # For each point, distance to the nearest point in the other cloud
    d = torch.cdist(pred.float(), gt.float(), compute_mode="donot_use_mm_for_euclid_dist")
    return d.min(2).values, d.min(1).values  # pred -> gt, gt -> pred


def chamfer_loss(pred, gt):
    # Same as chamferdist.ChamferDistance(bidirectional=True) in official train.py:
    # squared distance, sum over points, mean over batch
    d1, d2 = nearest_dist(pred, gt)
    return (d1.pow(2).sum(1) + d2.pow(2).sum(1)).mean()


def chamfer_fscore(pred, gt, threshold=0.01):
    # Per-sample metrics. CD as official test code: L2 distance, mean of each direction, added.
    d1, d2 = nearest_dist(pred, gt)
    cd = d1.mean(1) + d2.mean(1)
    precision = (d1 < threshold).float().mean(1)
    recall = (d2 < threshold).float().mean(1)
    fscore = 2 * precision * recall / (precision + recall)
    fscore = torch.nan_to_num(fscore, nan=0.0)
    return cd, fscore


def emd(pred, gt):
    # Earth Mover's Distance: best one-to-one matching of the points (Hungarian algorithm),
    # then the mean L2 distance of matched pairs. Exact but slow, so only used in evaluate.py.
    d = torch.cdist(pred.float(), gt.float()).cpu().numpy()
    out = []
    for cost in d:
        rows, cols = linear_sum_assignment(cost)
        out.append(cost[rows, cols].mean())
    return torch.tensor(np.array(out))


@torch.no_grad()
def evaluate_head(head, cache, batch_size=256, with_emd=False):
    # Per-sample metrics, then the mean per class and the mean over classes (as the paper)
    head.eval()
    metrics = {"cd": [], "fscore": [], "emd": []}
    n = len(cache["features"])
    for i in range(0, n, batch_size):
        pred = head(cache["features"][i:i + batch_size])
        gt = cache["points"][cache["obj_index"][i:i + batch_size]]
        cd, f = chamfer_fscore(pred, gt)
        metrics["cd"].append(cd.cpu())
        metrics["fscore"].append(f.cpu())
        if with_emd:
            metrics["emd"].append(emd(pred, gt))
    names = ["cd", "fscore", "emd"] if with_emd else ["cd", "fscore"]
    metrics = {k: torch.cat(metrics[k]) for k in names}
    obj_class = [cache["class_ids"][j] for j in cache["obj_index"].tolist()]
    results = {}
    for class_id in sorted(set(obj_class)):
        mask = torch.tensor([c == class_id for c in obj_class])
        results[class_id] = {k: metrics[k][mask].mean().item() for k in names}
    results["mean"] = {k: sum(r[k] for r in results.values()) / len(results) for k in names}
    return results
