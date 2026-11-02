import argparse
import itertools
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from demo import MODELS, predict, prepare

# Our own real-photo dataset: the team takes phone photos of real airplanes (models / toys), cars and chairs,
# several views of each object, stored as data/photos/<class>/<object>/<view>.jpg.
# build: remove the background and crop like the ShapeNet renderings -> data/processed/photos (+ manifest.json,
#        with the date and camera from the photo's EXIF as proof of where the photos come from)
# sheet: one figure with every original photo and its preprocessed version (for the report)
# eval:  no 3D ground truth, so we check consistency: two photos of the same object should give closer shapes
#        (CD between views) than photos of two different objects of the same class (CD between objects).


def camera_info(image):
    # Date taken and phone model from the EXIF data (empty if the photo has none)
    exif = image.getexif()
    date = exif.get_ifd(0x8769).get(36867) or exif.get(306)  # DateTimeOriginal, else DateTime
    camera = " ".join(str(exif.get(tag, "")) for tag in [271, 272]).strip()  # Make, Model
    return date, camera


def build(raw_dir, out_dir):
    manifest = []
    for class_name in sorted(os.listdir(raw_dir)):
        for obj in sorted(os.listdir(os.path.join(raw_dir, class_name))):
            folder = os.path.join(raw_dir, class_name, obj)
            for view in sorted(os.listdir(folder)):
                image = Image.open(os.path.join(folder, view))
                date, camera = camera_info(image)
                image = ImageOps.exif_transpose(image)  # phone photos store the rotation in EXIF
                prepared, note = prepare(image)
                path = os.path.join(out_dir, class_name, obj, os.path.splitext(view)[0] + ".png")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                prepared.save(path)
                # Share of the crop covered by the object (about 0.3-0.5 if the mask is good)
                area = float((np.array(prepared.getchannel("A")) > 127).mean())
                manifest.append({"class": class_name, "object": obj, "view": view, "file": path,
                                 "source_size": image.size, "date_taken": date, "camera": camera,
                                 "object_area": round(area, 3), "note": note})
                print(path, note, "area", round(area, 3))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    for class_name in sorted({m["class"] for m in manifest}):
        rows = [m for m in manifest if m["class"] == class_name]
        print(class_name, len({m["object"] for m in rows}), "objects,", len(rows), "photos")
    print(f"{len(manifest)} images, {len({(m['class'], m['object']) for m in manifest})} objects")


def sheet(raw_dir, out_dir, sheet_path, size=160):
    # One row per object: original photo and preprocessed image (on white) for every view
    manifest = json.load(open(os.path.join(out_dir, "manifest.json")))
    objects = sorted({(m["class"], m["object"]) for m in manifest})
    views = max(len([m for m in manifest if (m["class"], m["object"]) == o]) for o in objects)
    figure = Image.new("RGB", (2 * views * size, len(objects) * (size + 20)), "white")
    draw = ImageDraw.Draw(figure)
    for row, (class_name, obj) in enumerate(objects):
        y = row * (size + 20)
        draw.text((4, y + 4), f"{class_name} / {obj}", fill="black")
        items = [m for m in manifest if (m["class"], m["object"]) == (class_name, obj)]
        for col, m in enumerate(items):
            original = ImageOps.exif_transpose(Image.open(os.path.join(raw_dir, class_name, obj, m["view"])))
            original = ImageOps.pad(original.convert("RGB"), (size, size), color="white")
            prepared = Image.open(m["file"]).convert("RGBA").resize((size, size))
            white = Image.new("RGB", (size, size), "white")
            white.paste(prepared, mask=prepared.getchannel("A"))
            figure.paste(original, (2 * col * size, y + 20))
            figure.paste(white, ((2 * col + 1) * size, y + 20))
    os.makedirs(os.path.dirname(sheet_path), exist_ok=True)
    figure.save(sheet_path)
    print("Saved", sheet_path)


def chamfer(a, b):
    # Mean distance to the nearest point, both directions (same definition as chamfer_fscore in metrics.py)
    d = torch.cdist(a, b)
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
    parser.add_argument("step", choices=["build", "eval", "sheet"])
    parser.add_argument("--raw_dir", default="data/photos")
    parser.add_argument("--out_dir", default="data/processed/photos")
    parser.add_argument("--result", default="outputs/photos/consistency.json")
    parser.add_argument("--sheet", default="outputs/photos/sheet.png")
    args = parser.parse_args()
    if args.step == "build":
        build(args.raw_dir, args.out_dir)
    elif args.step == "eval":
        evaluate(args.out_dir, args.result)
    else:
        sheet(args.raw_dir, args.out_dir, args.sheet)


if __name__ == "__main__":
    main()
