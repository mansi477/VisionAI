#!/usr/bin/env python3
"""
Draw boxes + class names on random images of a YOLO dataset so you can
eyeball whether the merge/class mapping is correct BEFORE training.
A wrong mapping silently ruins a model, so always run this once.

Usage:
  python preview_labels.py --data /content/datasets/visionai/data.yaml --n 24 --out /content/preview
"""
import argparse
import random
from pathlib import Path

import cv2
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="preview")
    ap.add_argument("--only", default=None, help="only show images containing this class name")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text())
    root = Path(cfg["path"])
    names = cfg["names"]
    img_dir, lab_dir = root / cfg[args.split], root / "labels" / args.split
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imgs = sorted(img_dir.glob("*"))
    if args.only:
        want = [k for k, v in names.items() if v == args.only]
        if not want:
            raise SystemExit(f"class '{args.only}' not in data.yaml")
        imgs = [p for p in imgs
                if any(l.split()[0] == str(want[0])
                       for l in (lab_dir / (p.stem + ".txt")).read_text().splitlines() if l.split())]
    random.shuffle(imgs)
    for p in imgs[: args.n]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        for line in (lab_dir / (p.stem + ".txt")).read_text().splitlines():
            c, xc, yc, bw, bh = line.split()
            x1, y1 = int((float(xc) - float(bw) / 2) * w), int((float(yc) - float(bh) / 2) * h)
            x2, y2 = int((float(xc) + float(bw) / 2) * w), int((float(yc) + float(bh) / 2) * h)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(im, str(names[int(c)]), (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        cv2.imwrite(str(out / p.name), im)
    print(f"Wrote previews to {out}")


if __name__ == "__main__":
    main()
