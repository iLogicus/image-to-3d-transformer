import io
import os
import sys
import tempfile

import gradio as gr
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, "src")  # run from repo root: python api/main.py
sys.path.insert(0, "app")

from demo import MODELS, reconstruct, save_result
from gradio_app import CSS, THEME, demo

# REST API + the Gradio demo in one server:
#   GET  /health            server is up
#   GET  /models            demo models
#   POST /predict           image -> JSON points (or a .ply / .obj file with format=file)
#   /docs                   interactive API documentation (Swagger)
#   /                       Gradio demo
app = FastAPI(title="2D -> 3D reconstruction API", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return MODELS


@app.post("/predict")
def predict(image: UploadFile = File(...),
            model: str = Query("b1_v3", description="one of GET /models"),
            remove_background: bool = Query(True),
            format: str = Query("json", pattern="^(json|file)$")):
    if model not in MODELS:
        raise HTTPException(400, f"unknown model {model}, choose one of {list(MODELS)}")
    try:
        picture = Image.open(io.BytesIO(image.file.read()))
    except UnidentifiedImageError:
        raise HTTPException(400, "the file is not an image")
    try:
        result = reconstruct(picture, model, remove_background)
    except FileNotFoundError as e:  # weights of this model are missing
        raise HTTPException(503, str(e))
    if format == "file":
        _, download = save_result(result, tempfile.mkdtemp(), model)
        return FileResponse(download, filename=os.path.basename(download))
    response = {
        "model": model,
        "kind": result["kind"],
        "preprocessing": result["note"],
        "time_ms": round(result["time_ms"], 1),
        "points": result["points"].astype(float).round(4).tolist(),  # viewer axes, y up
    }
    if result["kind"] == "voxels":
        response["num_voxels"] = int(result["voxels"].sum())
    return response


# Gradio last, so the API routes above are matched first
app = gr.mount_gradio_app(app, demo, path="/", theme=THEME, css=CSS)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
