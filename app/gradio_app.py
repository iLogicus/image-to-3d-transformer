import sys

import gradio as gr

sys.path.append("src")  # run from repo root: python app/gradio_app.py

from inference import predict


def reconstruct(image):
    # TODO: load model once, call predict(), save .glb (viewer) and .ply (download)
    return None, None


demo = gr.Interface(
    fn=reconstruct,
    inputs=gr.Image(type="pil", label="Ảnh 2D"),
    outputs=[gr.Model3D(label="Mô hình 3D"), gr.File(label="Tải point cloud (.ply)")],
    title="2D → 3D Reconstruction",
)

if __name__ == "__main__":
    demo.launch()
