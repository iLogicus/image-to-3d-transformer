import io
import os
import sys
import tempfile
from typing import Literal

import gradio as gr
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

sys.path.insert(0, "src")  # run from repo root: python api/main.py
sys.path.insert(0, "app")

from demo import MODELS, OUTPUTS, reconstruct, save_result
from gradio_app import CSS, THEME, demo

# REST API + the Gradio demo in one server:
#   GET  /health    server is up
#   GET  /models    demo models
#   POST /predict   image -> JSON points (or a .ply / .obj file with format=file)
#   /docs           interactive API documentation (Swagger)
#   /               Gradio demo
DESCRIPTION = """
REST API of the **Single-image 3D reconstruction** demo (airplane, car, chair). The interactive demo is at [/](/).

| Model | Name in the demo | Output |
|---|---|---|
""" + "\n".join(f"| `{name}` | {label} | {OUTPUTS[name]} |" for name, label in MODELS.items()) + """

`POST /predict` removes the background (rembg U²-Net), crops the object like the ShapeNet renderings and runs the
chosen model on CPU. Points are in viewer axes (y up), scaled so the bounding box diagonal is about 1.
"""

app = FastAPI(title="Single-image 3D reconstruction API", version="1.0", description=DESCRIPTION)

ModelName = Literal["b1_v3", "rgb2point", "b2_v4", "retr_s"]  # same names as demo.MODELS (dropdown in /docs)


class Health(BaseModel):
    status: str = Field(examples=["ok"])


class ModelInfo(BaseModel):
    name: str = Field(description="name used by the API", examples=["b1_v3"])
    label: str = Field(description="name shown in the demo", examples=[MODELS["b1_v3"]])
    output: Literal["points", "voxels"]


class Prediction(BaseModel):
    model: str = Field(examples=["b1_v3"])
    label: str = Field(examples=[MODELS["b1_v3"]])
    output: Literal["points", "voxels"]
    preprocessing: str = Field(examples=["background removed (rembg U²-Net)"])
    time_ms: float = Field(description="model time on CPU", examples=[95.2])
    num_voxels: int | None = Field(None, description="occupied voxels of the 32³ grid (voxel models only)")
    points: list[list[float]] = Field(description="1024 points, or the centers of the occupied voxels",
                                      examples=[[[0.12, -0.31, 0.05], [0.10, 0.27, -0.08]]])


@app.get("/health", tags=["status"], summary="Server status")
def health() -> Health:
    return Health(status="ok")


@app.get("/models", tags=["status"], summary="Available models")
def models() -> list[ModelInfo]:
    return [ModelInfo(name=name, label=label, output=OUTPUTS[name]) for name, label in MODELS.items()]


@app.post("/predict", tags=["reconstruction"], summary="Reconstruct one image", response_model=Prediction,
          responses={200: {"content": {"application/octet-stream": {}},
                           "description": "JSON prediction, or a .ply (points) / .obj (voxels) file with format=file"},
                     400: {"description": "The file is not an image"},
                     503: {"description": "The model weights could not be loaded"}})
def predict(image: UploadFile = File(..., description="photo or rendering of an airplane, car or chair"),
            model: ModelName = Query("b1_v3", description="model name, see GET /models"),
            remove_background: bool = Query(True, description="cut the object out with rembg U²-Net "
                                                              "(not used for transparent PNG files)"),
            format: Literal["json", "file"] = Query("json", description="json, or file to download .ply / .obj")):
    try:
        picture = Image.open(io.BytesIO(image.file.read()))
    except UnidentifiedImageError:
        raise HTTPException(400, "the file is not an image")
    try:
        result = reconstruct(picture, model, remove_background)
    except OSError as e:  # weights missing or not downloadable
        raise HTTPException(503, f"model weights could not be loaded: {e}")
    if format == "file":
        _, download = save_result(result, tempfile.mkdtemp(), model)
        return FileResponse(download, filename=os.path.basename(download))
    return Prediction(
        model=model,
        label=MODELS[model],
        output=result["kind"],
        preprocessing=result["note"],
        time_ms=round(result["time_ms"], 1),
        num_voxels=int(result["voxels"].sum()) if result["kind"] == "voxels" else None,
        points=result["points"].astype(float).round(4).tolist(),
    )


# Gradio last, so the API routes above are matched first
app = gr.mount_gradio_app(app, demo, path="/", theme=THEME, css=CSS)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
