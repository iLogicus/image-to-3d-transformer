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

# V3: training augmentation (small shift / zoom, random background, colour). No flip: the ground truth is in the
# object frame, so a mirrored image would not match its point cloud.
# Shift / zoom is done on the transparent image, so new border pixels are transparent too.
TRAIN_SHAPE = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
])
TRAIN_COLOR = transforms.Compose([
    transforms.ColorJitter(0.3, 0.3, 0.3),
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


def view_angles(data_cfg, class_id, obj):
    # (azimuth, elevation) in degrees of each rendered view, from rendering_metadata.txt
    folder = os.path.dirname(image_path(data_cfg, class_id, obj, 0))
    with open(os.path.join(folder, "rendering_metadata.txt")) as f:
        return [(float(line.split()[0]), float(line.split()[1])) for line in f if line.strip()]


class ImageDataset(Dataset):
    # One sample = one rendered view of one object.
    # Train uses all views, val/test use only view 00 (as official code).
    # all_views=True uses every view also for val/test (for the error analysis).
    # augment=True (V3 training): TRAIN_SHAPE, random background colour, TRAIN_COLOR.
    def __init__(self, data_cfg, stage, all_views=False, augment=False):
        self.data_cfg = data_cfg
        self.augment = augment
        self.objects = []
        self.samples = []  # (object index, view)
        missing = 0
        for class_id, obj in read_split(data_cfg["split"], data_cfg["classes"], stage):
            if stage == "train" or all_views:
                views = train_views(data_cfg, class_id, obj)
            else:
                views = [data_cfg["test_view"]]
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
        image = Image.open(image_path(self.data_cfg, class_id, obj, view))
        if not self.augment:
            return TRANSFORM(image.convert("RGB")), obj_idx
        # Shift / zoom, then put the object (transparent PNG) on a background of random colour
        image = TRAIN_SHAPE(image.convert("RGBA"))
        color = tuple(np.random.randint(0, 256, 3).tolist())
        background = Image.new("RGB", image.size, color)
        background.paste(image.convert("RGB"), mask=image.getchannel("A"))
        return TRAIN_COLOR(background), obj_idx


def feature_path(data_cfg, stage):
    return os.path.join(data_cfg["feature_dir"], f"{stage}.pt")


def load_cache(data_cfg, stage, device):
    # Cached ViT features (from cache_features.py) stay on CPU in float16 (V1 train features are
    # about 18 GB); each batch is moved to the GPU. Ground truth points go to the GPU.
    cache = torch.load(feature_path(data_cfg, stage))
    return {
        "features": cache["features"],                             # (num_images, 768) or (num_images, 50, 768)
        "obj_index": cache["obj_index"],                           # (num_images,)
        "points": load_points(data_cfg, cache["objects"]).to(device),  # (num_objects, 1024, 3)
        "class_ids": [c for c, _ in cache["objects"]],
        "objects": cache["objects"],
    }


def drop_objects(cache, exclude_file):
    # Keep only images whose object is NOT listed in exclude_file (lines "class_id/object_id")
    with open(exclude_file) as f:
        exclude = {tuple(line.strip().split("/")) for line in f}
    keep_obj = torch.tensor([tuple(o) not in exclude for o in cache["objects"]])
    keep = keep_obj[cache["obj_index"]]
    cache["features"] = cache["features"][keep]
    cache["obj_index"] = cache["obj_index"][keep]
    print(f"Kept {keep.sum().item()} / {len(keep)} images (dropped objects in {exclude_file})")
    return cache


# ---------------- Baseline 2 (3D-RETR): image -> 32^3 voxels ----------------

def retr_transform(data_cfg):
    # Same as 3D-RETR train.py: CenterCrop((224, 224), (128, 128)) = crop the middle 128 x 128 of the 137 x 137
    # render, resize to 224. Baseline: x * 2 - 1 (mean 0.5, std 0.5). V1 (DINOv2): ImageNet mean / std.
    return transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg["mean"], data_cfg["std"]),
    ])


# V3 (baseline 2): training augmentation, same idea as V3 of baseline 1. Shift / zoom on the transparent image
# (so new border pixels are transparent too), then a background of random colour, then colour jitter. No flip.
RETR_SHAPE = transforms.Compose([
    transforms.CenterCrop(128),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
])
RETR_COLOR = transforms.ColorJitter(0.3, 0.3, 0.3)


def voxel_path(data_cfg, class_id, obj):
    return os.path.join(data_cfg["voxel_root"], class_id, obj, "model.binvox")


def read_binvox(path):
    # binvox file: text header ending with "data", then run-length pairs (value, count) of bytes.
    # The axes are swapped (0, 2, 1) as read_as_3d_array(fix_coords=True) used by 3D-RETR.
    with open(path, "rb") as f:
        while f.readline().strip() != b"data":
            pass
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    values, counts = raw[0::2], raw[1::2]
    voxels = np.repeat(values, counts).reshape(32, 32, 32)
    return np.transpose(voxels, (0, 2, 1)).astype(np.float32)


class VoxelDataset(Dataset):
    # One sample = one rendered view + the 32^3 voxel grid of the object.
    # Train: one_view_per_object=True is the official 3D-RETR epoch (one random view of each object),
    # False uses every view in each epoch (as baseline 1). Val/test: only view 00 (official mode="first").
    # augment=True (V3 training): RETR_SHAPE, random background colour, RETR_COLOR.
    def __init__(self, data_cfg, stage, all_views=False, augment=False):
        self.data_cfg = data_cfg
        self.augment = augment
        self.transform = retr_transform(data_cfg)
        self.random_view = stage == "train" and data_cfg["one_view_per_object"]
        self.objects = []
        self.views = []    # all views of each object
        self.samples = []  # (object index, view)
        missing = 0
        for class_id, obj in read_split(data_cfg["split"], data_cfg["classes"], stage):
            if stage == "train" or all_views:
                views = list(train_views(data_cfg, class_id, obj))
            else:
                views = [data_cfg["test_view"]]
            ok = len(views) > 0 and os.path.exists(voxel_path(data_cfg, class_id, obj)) and all(
                os.path.exists(image_path(data_cfg, class_id, obj, v)) for v in views)
            if not ok:
                missing += 1
                continue
            if self.random_view:
                self.samples.append((len(self.objects), None))  # the view is chosen in __getitem__
            else:
                for v in views:
                    self.samples.append((len(self.objects), v))
            self.objects.append((class_id, obj))
            self.views.append(views)
        print(f"[{stage}] objects: {len(self.objects)}, samples: {len(self.samples)}, skipped (missing files): {missing}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_idx, view = self.samples[idx]
        class_id, obj = self.objects[obj_idx]
        if view is None:
            view = int(np.random.choice(self.views[obj_idx]))
        # Put the transparent render on a plain background (official command: black, --background default 0 0 0)
        rgba = Image.open(image_path(self.data_cfg, class_id, obj, view)).convert("RGBA")
        voxels = torch.from_numpy(read_binvox(voxel_path(self.data_cfg, class_id, obj)))
        if self.augment:
            rgba = RETR_SHAPE(rgba)  # 128 x 128, so the CenterCrop(128) of self.transform does nothing
            color = tuple(np.random.randint(0, 256, 3).tolist())
        else:
            color = (self.data_cfg["background"],) * 3
        image = Image.new("RGB", rgba.size, color)
        image.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
        if self.augment:
            image = RETR_COLOR(image)
        return self.transform(image), voxels, obj_idx
