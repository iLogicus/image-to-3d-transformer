import io
import os
import time

import numpy as np
import torch
import trimesh
from PIL import Image
from rembg import new_session, remove
from torchvision import transforms

from metrics import chamfer_fscore
from models.retr import RETR
from models.rgb2point import build_model

# Shared code of the Gradio demo (app/gradio_app.py) and the FastAPI server (api/main.py):
# photo -> object mask -> square crop -> model -> point cloud or voxels -> files for the 3D viewer and download.

# Demo models. Weight files are made on Colab by src/export_demo.py.
MODELS = {
    "b1_v3": "Baseline 1 V3: DINOv2 + query decoder (point cloud)",
    "rgb2point": "Baseline 1: RGB2Point (point cloud)",
    "b2_v4": "Baseline 2 V4: DINOv2 + augmentation (voxels)",
    "retr_s": "Baseline 2: 3D-RETR-S (voxels)",
}
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "weights")
WEIGHTS_REPO = os.environ.get("WEIGHTS_REPO")  # Hugging Face model repo with the weight files (for Docker / Spaces)

FILL = 0.7  # the object takes 70% of the image side, as in the ShapeNet renderings (same as eval_pix3d.py)
BACKGROUND = (0, 0, 0)  # transparent pixels of the renderings are black after convert("RGB")

# Our data has -x up. Rotation to the usual viewer axes (y up). det = +1, so no mirror.
TO_VIEWER = torch.tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=torch.float32)
# Voxel grid axes -> point cloud axes, chosen with ground truth in cross_eval.py
VOXEL_AXES = torch.tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=torch.float32)

_models = {}
_session = None
_samples = None


def weight_file(name):
    # Local file first (weights/<name>.pt), else download it from the Hugging Face Hub
    path = os.path.join(WEIGHTS_DIR, name + ".pt")
    if os.path.exists(path):
        return path
    if WEIGHTS_REPO is None:
        raise FileNotFoundError(f"{path} not found and WEIGHTS_REPO is not set")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(WEIGHTS_REPO, name + ".pt")


def build_from_file(info):
    # info: dict saved by export_demo.py (model config, image normalization, weights)
    if info["kind"] == "points":
        model = build_model(info["model_cfg"], info["num_points"], pretrained=False)
    else:
        model = RETR(info["model_cfg"], pretrained=False)
    state = {k: v.float() for k, v in info["state"].items()}  # weights are stored in fp16
    model.load_state_dict(state)
    return model.eval()


def load_demo_model(name):
    # Each model is loaded once and kept in memory
    if name not in _models:
        info = torch.load(weight_file(name), map_location="cpu")
        model = build_from_file(info)
        # Points: Resize(224) as RGB2Point. Voxels: crop the middle 128 of the 137 x 137 render, as 3D-RETR.
        steps = [] if info["kind"] == "points" else [transforms.Resize((137, 137)), transforms.CenterCrop(128)]
        transform = transforms.Compose(steps + [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(info["mean"], info["std"]),
        ])
        _models[name] = (info["kind"], model, transform, info.get("threshold", 0.5))
    return _models[name]


def segment(image):
    # Object mask with rembg (U^2-Net salient object detection, any class). Returns a PIL "L" image.
    global _session
    if _session is None:
        _session = new_session("u2net")
    mask = remove(image, session=_session, only_mask=True)
    return mask.point(lambda v: 255 if v > 127 else 0)


def prepare(image, remove_background=True):
    # Make a photo look like a ShapeNet rendering: object only, square crop, object fills 70%.
    # Returns (RGBA image 224 x 224 with the mask as alpha, note for the user). Running it again on its
    # output gives the same image (the mask is read from the alpha channel).
    image = image.convert("RGBA")
    if np.array(image.getchannel("A")).min() < 255:  # transparent PNG: the mask is already given
        mask = image.getchannel("A")
        note = "mask from the PNG transparency"
    elif remove_background:
        mask = segment(image.convert("RGB"))
        note = "background removed (rembg U²-Net)"
    else:
        mask = Image.new("L", image.size, 255)
        note = "background kept"
    if mask.getbbox() is None:  # nothing found: keep the whole image
        mask = Image.new("L", image.size, 255)
        note = "no object found, background kept"
    image.putalpha(mask)
    x0, y0, x1, y1 = mask.getbbox()
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = max(int(max(x1 - x0, y1 - y0) / 2 / FILL), 1)
    # Add a transparent border, so the square crop never leaves the image
    padded = Image.new("RGBA", (image.width + 2 * half, image.height + 2 * half), (0, 0, 0, 0))
    padded.paste(image, (half, half))
    cx, cy = cx + half, cy + half
    return padded.crop((cx - half, cy - half, cx + half, cy + half)).resize((224, 224)), note


def on_background(rgba):
    # Object on the black background of the renderings, as the models were trained
    image = Image.new("RGB", rgba.size, BACKGROUND)
    image.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
    return image


def voxel_size(voxels):
    # Side of one voxel after scaling the shape so its bounding box diagonal is 1, as the ShapeNet point clouds
    idx = torch.nonzero(voxels).float()
    if len(idx) == 0:
        return 1 / 32
    return 1 / (idx.max(0).values - idx.min(0).values + 1).norm().item()


def voxel_points(voxels):
    # Centers of the occupied voxels in the point cloud axes, same scale as the point clouds, centered
    idx = torch.nonzero(voxels).float()
    if len(idx) == 0:
        return torch.zeros(1, 3)
    points = idx * voxel_size(voxels) @ VOXEL_AXES.T
    return points - (points.max(0).values + points.min(0).values) / 2


@torch.no_grad()
def predict(prepared, name, render=False):
    # Prepared RGBA image (output of prepare) -> result dict: kind, points (viewer axes), voxels (bool 32^3 or None), time
    # render=True: a ShapeNet test rendering, given to the model exactly as in our evaluation
    # (baseline 1: convert("RGB") as RGB2Point, baseline 2: on the black background as 3D-RETR)
    kind, model, transform, threshold = load_demo_model(name)
    image = prepared.convert("RGB") if render and kind == "points" else on_background(prepared)
    start = time.time()
    out = model(transform(image).unsqueeze(0))[0]
    ms = (time.time() - start) * 1000
    if kind == "points":
        voxels = None
        points = out
    else:
        voxels = torch.sigmoid(out) > threshold
        points = voxel_points(voxels)
    return {"kind": kind, "points": (points @ TO_VIEWER.T).numpy(), "voxels": voxels, "time_ms": ms}


def reconstruct(image, name, remove_background=True):
    # Any PIL image -> prepare -> predict. The result also has the preprocessing note.
    prepared, note = prepare(image, remove_background)
    result = predict(prepared, name)
    result["note"] = note
    return result


def load_samples():
    # Test set samples made by export_samples.py: render of view 00, ground truth points and voxels
    global _samples
    if _samples is None:
        _samples = torch.load(weight_file("samples"))
        for sample in _samples:
            sample["picture"] = Image.open(io.BytesIO(sample["image"])).convert("RGBA")
    return _samples


def ground_truth(sample):
    # Ground truth as two results, so save_result can make their viewer files
    return ({"kind": "points", "points": (sample["points"].float() @ TO_VIEWER.T).numpy(), "voxels": None},
            {"kind": "voxels", "points": None, "voxels": sample["voxels"]})


def score(result, sample):
    # Same metrics as our evaluation: CD x100 and F-score@0.01 for points, IoU for voxels
    if result["kind"] == "points":
        pred = torch.tensor(result["points"]) @ TO_VIEWER  # back to the data axes (TO_VIEWER is a rotation)
        cd, f = chamfer_fscore(pred[None], sample["points"].float()[None])
        return f"CD x100 {cd.item() * 100:.2f}, F@0.01 {f.item():.3f}"
    pred, gt = result["voxels"], sample["voxels"]
    return f"IoU {((pred & gt).sum() / (pred | gt).sum()).item():.3f}"


def height_colors(points):
    # Colour each point by its height (y), dark blue (low) to light blue (high), as the demo theme
    t = (points[:, 1] - points[:, 1].min()) / (np.ptp(points[:, 1]) + 1e-8)
    low, high = np.array([24, 56, 99]), np.array([120, 160, 215])
    rgb = low + t[:, None] * (high - low)
    return np.hstack([rgb, np.full((len(points), 1), 255)]).astype(np.uint8)


def points_mesh(points, radius=0.007):
    # The 3D viewer shows meshes better than raw points: one small sphere per point
    ball = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    n, nv = len(points), len(ball.vertices)
    vertices = (ball.vertices[None] + points[:, None]).reshape(-1, 3)
    faces = (ball.faces[None] + nv * np.arange(n)[:, None, None]).reshape(-1, 3)
    colors = np.repeat(height_colors(points), nv, axis=0)
    return trimesh.Trimesh(vertices, faces, vertex_colors=colors, process=False)


def voxel_mesh(voxels):
    # One cube per occupied voxel, in the same axes and scale as voxel_points
    idx = torch.nonzero(voxels).float()
    if len(idx) == 0:
        return trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    centers = voxel_points(voxels) @ TO_VIEWER.T
    box = trimesh.creation.box(extents=(voxel_size(voxels),) * 3)
    n, nv = len(centers), len(box.vertices)
    vertices = (box.vertices[None] + centers.numpy()[:, None]).reshape(-1, 3)
    faces = (box.faces[None] + nv * np.arange(n)[:, None, None]).reshape(-1, 3)
    colors = np.repeat(height_colors(centers.numpy()), nv, axis=0)
    return trimesh.Trimesh(vertices, faces, vertex_colors=colors, process=False)


def save_result(result, out_dir, stem):
    # Write the viewer file (.glb) and the download file (.ply points or .obj voxel mesh). Returns both paths.
    os.makedirs(out_dir, exist_ok=True)
    viewer = os.path.join(out_dir, stem + ".glb")
    if result["kind"] == "points":
        points_mesh(result["points"]).export(viewer)
        download = os.path.join(out_dir, stem + ".ply")
        trimesh.PointCloud(result["points"], colors=height_colors(result["points"])).export(download)
    else:
        mesh = voxel_mesh(result["voxels"])
        mesh.export(viewer)
        download = os.path.join(out_dir, stem + ".obj")
        mesh.export(download)
    return viewer, download
