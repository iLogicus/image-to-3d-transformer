import argparse
import itertools
import json
import os

import numpy as np
import torch
from PIL import Image

from demo import MODELS, predict, prepare

# Our own real-photo dataset: the team takes phone photos of real airplanes (models / toys), cars and chairs,
# several views of each object, stored as data/photos/<class>/<object>/<view>.jpg.
# build: remove the background and crop like the ShapeNet renderings -> data/processed/photos (+ manifest.json)
# eval:  no 3D ground truth, so we check consistency: two photos of the same object should give closer shapes
#        (CD between views) than photos of two different objects of the same class (CD between objects).


def build(raw_dir, out_dir):
    manifest = []
    for class_name in sorted(os.listdir(raw_dir)):
        for obj in sorted(os.listdir(os.path.join(raw_dir, class_name))):
            folder = os.path.join(raw_dir, class_name, obj)
            for view in sorted(os.listdir(folder)):
                image = Image.open(os.path.join(folder, view))
                prepared, note = prepare(image)
                path = os.path.join(out_dir, class_name, obj, os.path.splitext(view)[0] + ".png")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                prepared.save(path)
                # Share of the crop covered by the object (about 0.3-0.5 if the mask is good)
                area = float((np.array(prepared.getchannel("A")) > 127).mean())
                manifest.append({"class": class_name, "object": obj, "view": view, "file": path,
                                 "source_size": image.size, "object_area": round(area, 3), "note": note})
                print(path, note, "area", round(area, 3))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{len(manifest)} images, {len({(m['class'], m['object']) for m in manifest})} objects")


def chamfer(a, b):
    # Mean squared distance to the nearest point, both directions (same definition as metrics.py)
    d = torch.cdist(a, b) ** 2
    return (d.min(1).values.mean() + d.min(0).values.mean()).item()


def evaluate(out_dir, result_path):
    manifest = json.load(open(os.path.join(out_dir, "manifest.json")))
    results = {}
    for name in MODELS:
        # Points of every photo, scaled so the bounding box diagonal is 1 (models output different scales)
        preds = []
        for m in manifest:
            p = torch.tensor(predict(Image.open(m["file"]).convert("RGBA"), name)["points"])
            preds.append((p - p.mean(0)) / (p.max(0).values - p.min(0).values).norm())
        results[name] = {}
        for class_name in sorted({m["class"] for m in manifest}):
            idx = [i for i, m in enumerate(manifest) if m["class"] == class_name]
            same, other = [], []
            for i, j in itertools.combinations(idx, 2):
                cd = chamfer(preds[i], preds[j])
                (same if manifest[i]["object"] == manifest[j]["object"] else other).append(cd)
            results[name][class_name] = {
                "cd_same_object": float(np.mean(same)) if same else None,
                "cd_other_object": float(np.mean(other)) if other else None,
            }
            r = results[name][class_name]
            if same and other:
                r["ratio"] = r["cd_other_object"] / r["cd_same_object"]  # larger = more specific to the object
            print(f"{name:10s} {class_name:9s} " + "  ".join(f"{k} {v:.4f}" for k, v in r.items() if v is not None))
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["build", "eval"])
    parser.add_argument("--raw_dir", default="data/photos")
    parser.add_argument("--out_dir", default="data/processed/photos")
    parser.add_argument("--result", default="outputs/photos/consistency.json")
    args = parser.parse_args()
    if args.step == "build":
        build(args.raw_dir, args.out_dir)
    else:
        evaluate(args.out_dir, args.result)


if __name__ == "__main__":
    main()
