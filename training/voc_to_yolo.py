#!/usr/bin/env python3
"""
Convert Pascal-VOC XML annotations (e.g. RDD2022 India) to a YOLO dataset.

By default keeps ONLY potholes (RDD code D40) and drops crack classes.
Output is a Roboflow-style dataset folder (with data.yaml) that
merge_datasets.py can pick up directly.

Usage:
  python voc_to_yolo.py --src /content/raw_rdd/India --out /content/raw/rdd2022_india
  # keep other codes too, e.g. cracks mapped to their own names:
  python voc_to_yolo.py --src ... --out ... --keep D40:pothole D00:crack
"""
import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with images and .xml files (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", nargs="+", default=["D40:pothole"],
                    help="VOC_NAME:yolo_name pairs to keep (default D40:pothole)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--keep-empty", action="store_true", help="keep images with no kept objects")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keep = dict(k.split(":", 1) for k in args.keep)
    class_names = list(dict.fromkeys(keep.values()))
    cid = {n: i for i, n in enumerate(class_names)}

    src, out = Path(args.src), Path(args.out)
    images = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT}
    xmls = sorted(src.rglob("*.xml"))
    if not xmls:
        sys.exit(f"No .xml files found under {src}")

    items, missing_img = [], 0
    for x in xmls:
        img = images.get(x.stem)
        if img is None:
            missing_img += 1
            continue
        root = ET.parse(x).getroot()
        size = root.find("size")
        W, H = float(size.findtext("width")), float(size.findtext("height"))
        if W <= 0 or H <= 0:
            continue
        lines = []
        for obj in root.iter("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in keep:
                continue
            bb = obj.find("bndbox")
            x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
            x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
            x1, x2 = max(0.0, min(x1, x2)), min(W, max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(H, max(y1, y2))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            lines.append(
                f"{cid[keep[name]]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
            )
        if lines or args.keep_empty:
            items.append((img, lines))

    random.seed(args.seed)
    random.shuffle(items)
    n_val = max(1, int(len(items) * args.val_frac)) if items else 0
    for split in ("train", "valid"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
    for i, (img, lines) in enumerate(items):
        split = "valid" if i < n_val else "train"
        shutil.copy2(img, out / split / "images" / img.name)
        (out / split / "labels" / (img.stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    (out / "data.yaml").write_text(
        yaml.safe_dump({"names": class_names, "nc": len(class_names)}), encoding="utf-8"
    )
    print(f"XML files: {len(xmls)}  (no matching image: {missing_img})")
    print(f"Wrote {len(items)} images ({n_val} val) to {out}")
    print("NOTE: RDD frames come from driving sequences, so val is not fully independent of train.")


if __name__ == "__main__":
    main()
