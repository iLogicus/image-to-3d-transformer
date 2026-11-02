import io
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

# Run from the repo root: python -m pytest tests
# The real weights are too large for CI, so tiny random models with the same code are saved as demo weights.
os.environ["WEIGHTS_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, "src")
sys.path.insert(0, "app")
sys.path.insert(0, "api")

from demo import MODELS, OUTPUTS, prepare, reconstruct, save_result
from export_demo import make_info
from models.retr import RETR
from models.rgb2point import build_model

TINY_POINTS = {"backbone": "vit_tiny_patch16_224", "tokens": "pooled", "head": "query", "num_queries": 16,
               "dim": 32, "num_layers": 1, "num_heads": 2}
TINY_VOXELS = {"encoder": "vit_tiny_patch16_224", "encoder_dropout": 0.0, "decoder_depth": 1, "decoder_heads": 2,
               "decoder_dim": 192, "decoder_dropout": 0.0, "patch_num": 4, "cnn_dim": 8, "threshold": 0.5}


def setup_module():
    torch.manual_seed(0)
    for name in MODELS:
        if OUTPUTS[name] == "points":
            model = build_model(TINY_POINTS, 64, pretrained=False)
            info = make_info("points", model, TINY_POINTS, [0.5] * 3, [0.5] * 3, num_points=64)
        else:
            model = RETR(TINY_VOXELS, pretrained=False)
            info = make_info("voxels", model, TINY_VOXELS, [0.5] * 3, [0.5] * 3)
        torch.save(info, os.path.join(os.environ["WEIGHTS_DIR"], name + ".pt"))
    # Two fake test set samples, same format as export_samples.py
    buffer = io.BytesIO()
    chair_png().save(buffer, format="PNG")
    voxels = torch.zeros(32, 32, 32, dtype=torch.bool)
    voxels[8:24, 8:24, 8:24] = True
    sample = {"class": "chair", "object": "abc", "image": buffer.getvalue(), "points": torch.rand(1024, 3) - 0.5,
              "voxels": voxels}
    torch.save([sample, sample], os.path.join(os.environ["WEIGHTS_DIR"], "samples.pt"))


def chair_png():
    # Transparent PNG with a red square: the mask comes from the alpha channel (no segmentation download)
    image = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    image.paste((200, 30, 30, 255), (100, 50, 160, 150))
    return image


def test_prepare_crops_object():
    prepared, note = prepare(chair_png())
    assert prepared.size == (224, 224)
    assert "transparency" in note
    alpha = np.array(prepared.getchannel("A"))
    # The object (100 px high) fills 70% of the crop, centered
    rows = np.nonzero(alpha.max(1) > 127)[0]
    assert abs((rows.max() - rows.min() + 1) / 224 - 0.7) < 0.03
    assert alpha[0, 0] == 0
    # Preparing the prepared image again changes nothing
    again, _ = prepare(prepared)
    assert np.abs(np.array(again, dtype=int) - np.array(prepared, dtype=int)).mean() < 2


def test_prepare_without_background_removal():
    prepared, note = prepare(Image.new("RGB", (120, 80), (255, 255, 255)), remove_background=False)
    assert prepared.size == (224, 224) and note == "background kept"


@pytest.mark.parametrize("name", list(MODELS))
def test_reconstruct_and_files(name):
    result = reconstruct(chair_png(), name)
    assert result["points"].ndim == 2 and result["points"].shape[1] == 3
    if result["kind"] == "points":
        assert len(result["points"]) == 64
    else:
        assert result["voxels"].shape == (32, 32, 32)
    viewer, download = save_result(result, tempfile.mkdtemp(), name)
    assert os.path.getsize(viewer) > 0 and os.path.getsize(download) > 0
    assert download.endswith(".ply" if result["kind"] == "points" else ".obj")


def test_api():
    from typing import get_args

    from fastapi.testclient import TestClient
    from main import ModelName, app

    assert set(get_args(ModelName)) == set(MODELS)  # API model list in sync with demo.MODELS
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert [m["name"] for m in client.get("/models").json()] == list(MODELS)
    assert client.get("/openapi.json").json()["info"]["title"] == "Single-image 3D reconstruction API"
    # Only the main event is listed on the Gradio "Use via API" page
    assert list(client.get("/gradio_api/info").json()["named_endpoints"]) == ["/reconstruct"]

    buffer = io.BytesIO()
    chair_png().save(buffer, format="PNG")
    files = {"image": ("chair.png", buffer.getvalue(), "image/png")}
    r = client.post("/predict?model=b1_v3", files=files)
    assert r.status_code == 200 and len(r.json()["points"]) == 64 and r.json()["label"] == MODELS["b1_v3"]
    r = client.post("/predict?model=b2_v4&format=file", files=files)
    assert r.status_code == 200 and r.content.startswith(b"#")  # .obj text file
    assert client.post("/predict?model=nope", files=files).status_code == 422  # not in the model list
    bad = {"image": ("x.png", b"not an image", "image/png")}
    assert client.post("/predict", files=bad).status_code == 400
    assert client.get("/").status_code == 200  # Gradio page


def test_build_photos():
    from build_photos import build, evaluate

    raw, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    for obj in ["a", "b"]:
        for view in ["1", "2"]:
            os.makedirs(os.path.join(raw, "chair", obj), exist_ok=True)
            chair_png().save(os.path.join(raw, "chair", obj, view + ".png"))
    build(raw, out)
    result = os.path.join(out, "consistency.json")
    evaluate(out, result)
    r = json.load(open(result))
    assert set(r) == set(MODELS) and r["b1_v3"]["chair"]["cd_same_object"] is not None


def test_sample_tab():
    from gradio_app import SAMPLES, run

    assert len(SAMPLES) == 2
    out = run("test", None, True, 1, "b1_v3", "b2_v4")
    assert os.path.getsize(out[1]) > 0 and os.path.getsize(out[3]) > 0  # predictions
    assert os.path.getsize(out[5]) > 0 and os.path.getsize(out[6]) > 0  # ground truth
    assert "CD x100" in out[-1] and "IoU" in out[-1]
    out = run("upload", chair_png(), True, None, "rgb2point", "retr_s")
    assert out[5] is None and "No ground truth" in out[-1]
