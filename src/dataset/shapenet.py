import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_IDS = {"airplane": "02691156", "car": "02958343", "chair": "03001627"}
CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}

# Same preprocessing as official RGB2Point train.py
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def read_split(split_path, classes, stage):
    # List of (class_id, object_id) from 3D-RETR ShapeNet.json
    with open(split_path) as f:
        split = json.load(f)
    ids = [CLASS_IDS[c] for c in classes]
    objects = []
    for entry in split:
        if entry["taxonomy_id"] in ids:
            for obj in entry[stage]:
                objects.append((entry["taxonomy_id"], obj))
    return objects


def image_path(data_cfg, class_id, obj, view):
    return os.path.join(data_cfg["image_root"], class_id, obj, "rendering", f"{view:02d}.png")


def points_path(data_cfg, class_id, obj):
    return os.path.join(data_cfg["points_root"], class_id, obj, "pointcloud_1024.npy")


def load_points(data_cfg, objects):
    # Ground truth point clouds, moved to the center (no scaling, as official code)
    points = []
    for class_id, obj in objects:
        pc = np.load(points_path(data_cfg, class_id, obj)).astype(np.float32)
        points.append(pc - pc.mean(axis=0))
    return torch.from_numpy(np.stack(points))


def train_views(data_cfg, class_id, obj):
    # All rendered views of the object: 24 for car/airplane, 10 for chair (HF mirror has only 00-09)
    folder = os.path.dirname(image_path(data_cfg, class_id, obj, 0))
    if not os.path.isdir(folder):
        return []
    return range(len([f for f in os.listdir(folder) if f.endswith(".png")]))


class ImageDataset(Dataset):
    # One sample = one rendered view of one object.
    # Train uses all views, val/test use only view 00 (as official code).
    def __init__(self, data_cfg, stage):
        self.data_cfg = data_cfg
        self.objects = []
        self.samples = []  # (object index, view)
        missing = 0
        for class_id, obj in read_split(data_cfg["split"], data_cfg["classes"], stage):
            views = train_views(data_cfg, class_id, obj) if stage == "train" else [data_cfg["test_view"]]
            ok = len(views) > 0 and os.path.exists(points_path(data_cfg, class_id, obj)) and all(
                os.path.exists(image_path(data_cfg, class_id, obj, v)) for v in views)
            if not ok:
                missing += 1
                continue
            for v in views:
                self.samples.append((len(self.objects), v))
            self.objects.append((class_id, obj))
        print(f"[{stage}] objects: {len(self.objects)}, images: {len(self.samples)}, skipped (missing files): {missing}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_idx, view = self.samples[idx]
        class_id, obj = self.objects[obj_idx]
        image = Image.open(image_path(self.data_cfg, class_id, obj, view)).convert("RGB")
        return TRANSFORM(image), obj_idx


def feature_path(data_cfg, stage):
    return os.path.join(data_cfg["feature_dir"], f"{stage}.pt")


def load_cache(data_cfg, stage, device):
    # Cached ViT features (from cache_features.py) + ground truth points, all on GPU
    cache = torch.load(feature_path(data_cfg, stage))
    return {
        "features": cache["features"].float().to(device),         # (num_images, 768)
        "obj_index": cache["obj_index"].to(device),                # (num_images,)
        "points": load_points(data_cfg, cache["objects"]).to(device),  # (num_objects, 1024, 3)
        "class_ids": [c for c, _ in cache["objects"]],
    }
