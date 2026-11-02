import glob
import os
import sys
import tempfile

import gradio as gr

sys.path.insert(0, "src")  # run from repo root: python app/gradio_app.py
sys.path.insert(0, "app")

from demo import MODELS, ground_truth, load_samples, on_background, predict, prepare, save_result, score
from theme import LabTheme

# Short names for the interface (the API keeps the long names of demo.MODELS)
NAMES = {
    "b1_v3": "V3 (ours): DINOv2 + query decoder",
    "rgb2point": "RGB2Point (baseline 1)",
    "b2_v4": "V4 (ours): DINOv2 + augmentation",
    "retr_s": "3D-RETR-S (baseline 2)",
}
POINT_MODELS = [(NAMES[name], name) for name in ["b1_v3", "rgb2point"]]
VOXEL_MODELS = [(NAMES[name], name) for name in ["b2_v4", "retr_s"]]
CAMERA = (35, 70, 2.0)  # 3D viewers: degrees around the vertical axis, degrees from the top, distance
EXAMPLES = sorted(glob.glob("app/examples/*"))  # optional: a few photos of the team
try:
    SAMPLES = load_samples()  # test set renders with ground truth (samples.pt from export_samples.py)
except Exception as e:  # e.g. no samples.pt, no internet, cache folder not writable
    print("Test set tab hidden, samples could not be loaded:", repr(e))
    SAMPLES = []

THEME = LabTheme()
CSS = """
.header h1 { font-size: 1.6rem; font-weight: 600; margin: 0.4rem 0 0.3rem; }
.header p { color: var(--body-text-color-subdued); margin: 0; max-width: 52rem; line-height: 1.5; }
.pipeline { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; margin-top: 0.9rem;
            font-family: var(--font-mono); font-size: 0.75rem; color: var(--body-text-color-subdued); }
.pipeline span { border: 1px solid var(--border-color-primary); border-radius: 4px; padding: 0.15rem 0.5rem;
                 background: var(--block-background-fill); color: var(--body-text-color); }
.step { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
        color: var(--color-accent); margin: 0.5rem 0 -0.4rem; }
.dark .step, .dark button[role="tab"][aria-selected="true"] { color: var(--primary-300) !important; }
.metrics table, .about table { width: 100%; border-collapse: collapse; font-size: 0.85rem; border: none !important; }
.metrics th, .about th { text-align: left; font-weight: 500; color: var(--body-text-color-subdued);
                         border: none !important; border-bottom: 1px solid var(--border-color-primary) !important;
                         padding: 0.45rem 0.5rem; background: none !important; }
.metrics td, .about td { padding: 0.45rem 0.5rem; border: none !important;
                         border-bottom: 1px solid var(--border-color-primary) !important; background: none !important; }
.metrics table, .metrics tr, .about table, .about tr { border-left: 0 !important; border-right: 0 !important; }
.metrics th.num, .about th.num { text-align: right; }
.metrics td.num, .about td.num { font-family: var(--font-mono); text-align: right; white-space: nowrap; }
.metrics .note, .about .note { color: var(--body-text-color-subdued); font-size: 0.8rem; margin-top: 0.5rem; }
"""

HEADER = """
<div class="header">
  <h1>Single-image 3D reconstruction</h1>
  <p>One image of an airplane, car or chair is turned into a 3D point cloud (1024 points) and a 32&sup3; voxel
  grid. Each output comes from a reproduced baseline and from our improved version, all trained on ShapeNet.</p>
  <div class="pipeline"><span>image</span>&rarr;<span>background removal</span>&rarr;<span>ViT encoder</span>
  &rarr;<span>Transformer decoder</span>&rarr;<span>point cloud / voxels</span></div>
</div>
"""

# Numbers of our report (test set of the 3D-RETR split, view 00; Pix3D: real chair photos)
ABOUT = """
<div class="about">
<table>
<tr><th>Model</th><th>Output</th><th class="num">Parameters</th><th class="num">ShapeNet test</th>
<th class="num">Pix3D chairs</th></tr>
<tr><td>RGB2Point (baseline 1)</td><td>points</td><td class="num">175 M</td><td class="num">CD 3.99, F 0.230</td>
<td class="num">CD 6.55, F 0.080</td></tr>
<tr><td>Baseline 1 V3: DINOv2 + query decoder</td><td>points</td><td class="num">90 M</td>
<td class="num">CD 3.42, F 0.273</td><td class="num">CD 5.49, F 0.111</td></tr>
<tr><td>3D-RETR-S (baseline 2)</td><td>voxels</td><td class="num">10 M</td><td class="num">IoU 0.678</td>
<td class="num">IoU 0.293</td></tr>
<tr><td>Baseline 2 V4: DINOv2 + augmentation</td><td>voxels</td><td class="num">27 M</td><td class="num">IoU 0.664</td>
<td class="num">IoU 0.327</td></tr>
</table>
<p class="note">CD = Chamfer distance x100 (lower is better), F = F-score@0.01, IoU of 32&sup3; voxels (higher is
better); mean over airplane, car and chair. V4 is shown instead of V3 (IoU 0.692 on ShapeNet) because it is the
best on real photos. Uploaded photos have no ground truth, so only test set images show metrics.</p>
</div>
"""


def metrics_table(rows, note):
    # rows: (model, output, time in ms, metric text)
    lines = "".join(f"<tr><td>{m}</td><td>{o}</td><td class='num'>{t:.0f} ms</td><td class='num'>{s}</td></tr>"
                    for m, o, t, s in rows)
    return (f"<div class='metrics'><table><tr><th>Model</th><th>Output</th><th class='num'>CPU time</th>"
            f"<th class='num'>Metric</th></tr>{lines}</table><p class='note'>{note}</p></div>")


def run(mode, image, remove_background, index, point_model, voxel_model):
    if mode == "test":
        if index is None:
            raise gr.Error("Choose a test image first")
        sample = SAMPLES[index]
        # Test renderings are given to the models as in our evaluation (no background removal, no crop)
        prepared, render = sample["picture"], True
        note = f"Test set: {sample['class']} {sample['object']}, view 00. Metrics against the ground truth."
    else:
        if image is None:
            raise gr.Error("Upload an image first")
        sample, render = None, False
        prepared, note = prepare(image, remove_background)  # background removed once, for both models
        note = f"Preprocessing: {note}. No ground truth for uploaded images."
    out_dir = tempfile.mkdtemp()
    files, rows = [], []
    for name in [point_model, voxel_model]:
        result = predict(prepared, name, render)
        viewer, download = save_result(result, out_dir, name)
        files += [viewer, gr.DownloadButton(value=download, interactive=True)]
        output = "1024 points" if result["kind"] == "points" else f"{int(result['voxels'].sum())} voxels"
        rows.append((NAMES[name], output, result["time_ms"], score(result, sample) if sample else "-"))
    if sample is None:
        gt = [None, None]
    else:
        gt_points, gt_voxels = ground_truth(sample)
        gt = [save_result(gt_points, out_dir, "gt_points")[0], save_result(gt_voxels, out_dir, "gt_voxels")[0]]
    return [on_background(prepared)] + files + gt + [metrics_table(rows, note)]


def select_sample(evt: gr.SelectData):
    return evt.index


with gr.Blocks(title="Single-image 3D reconstruction") as demo:
    mode = gr.State("upload")
    sample_index = gr.State(None)
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=340):
            gr.HTML("<div class='step'>1 &middot; Input</div>")
            with gr.Tabs():
                with gr.Tab("Upload image") as upload_tab:
                    image = gr.Image(type="pil", image_mode="RGBA", label="Image", height=260)
                    remove_background = gr.Checkbox(True, label="Remove background (rembg U²-Net)")
                    if EXAMPLES:
                        gr.Examples(EXAMPLES, inputs=image)
                with gr.Tab("Test set", visible=bool(SAMPLES)) as test_tab:
                    gallery = gr.Gallery([(s["picture"], s["class"]) for s in SAMPLES], columns=6, height=300,
                                         allow_preview=False, object_fit="contain", show_label=False)
            gr.HTML("<div class='step'>2 &middot; Models</div>")
            # Radio buttons: only 2 models per output, both visible at once
            point_model = gr.Radio(POINT_MODELS, value=POINT_MODELS[0][1], label="Point cloud model")
            voxel_model = gr.Radio(VOXEL_MODELS, value=VOXEL_MODELS[0][1], label="Voxel model")
            button = gr.Button("Reconstruct", variant="primary", size="lg")
            model_input = gr.Image(label="Model input", height=180, interactive=False)
        with gr.Column(scale=7):
            gr.HTML("<div class='step'>3 &middot; Result</div>")
            with gr.Row():
                with gr.Column():
                    point_view = gr.Model3D(label="Point cloud", height=380, clear_color=(0, 0, 0, 0), camera_position=CAMERA)
                    point_download = gr.DownloadButton("Download .ply", interactive=False, size="sm")
                with gr.Column():
                    voxel_view = gr.Model3D(label="Voxels 32³", height=380, clear_color=(0, 0, 0, 0), camera_position=CAMERA)
                    voxel_download = gr.DownloadButton("Download .obj", interactive=False, size="sm")
            # Always visible: a 3D viewer created inside a hidden row stays blank when the row is shown
            with gr.Row():
                gt_point_view = gr.Model3D(label="Ground truth points (test set only)", height=260, min_width=280, clear_color=(0, 0, 0, 0), camera_position=CAMERA)
                gt_voxel_view = gr.Model3D(label="Ground truth voxels (test set only)", height=260, min_width=280, clear_color=(0, 0, 0, 0), camera_position=CAMERA)
            metrics = gr.HTML("<div class='metrics'><p class='note'>Results appear here.</p></div>")
    with gr.Accordion("About the models", open=False):
        gr.HTML(ABOUT)

    upload_tab.select(lambda: "upload", None, mode)
    test_tab.select(lambda: "test", None, mode)
    gallery.select(select_sample, None, sample_index)
    button.click(run, [mode, image, remove_background, sample_index, point_model, voxel_model],
                 [model_input, point_view, point_download, voxel_view, voxel_download, gt_point_view,
                  gt_voxel_view, metrics])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), theme=THEME, css=CSS)
