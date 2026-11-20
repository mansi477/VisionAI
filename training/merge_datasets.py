#!/usr/bin/env python3
"""
Merge several YOLO-format datasets into ONE dataset that uses the VisionAI
master class list (see class_map.yaml).

Works with both common layouts:
  Roboflow export :  <ds>/train/images, <ds>/train/labels, <ds>/valid/..., <ds>/test/...
  Ultralytics     :  <ds>/images/train, <ds>/labels/train, ...

Every dataset folder must contain a data.yaml with its class `names`
(Roboflow exports include one; if a dataset such as DriveIndia does not,
create it yourself using the class order from that dataset's own docs).

Usage:
  python merge_datasets.py --raw /content/raw --out /content/datasets/visionai --clean
"""
import argparse
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": ["train"], "val": ["val", "valid", "validation"], "test": ["test"]}


def norm(name):
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def load_class_map(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    master = list(cfg["master_classes"])
    lookup = {norm(m): m for m in master}
    for m, aliases in (cfg.get("aliases") or {}).items():
        if m not in master:
            sys.exit(f"class_map.yaml: alias group '{m}' is not in master_classes")
        for a in aliases or []:
            lookup[norm(a)] = m
    overrides = {}
    for ds, mapping in (cfg.get("overrides") or {}).items():
        overrides[ds] = {}
        for src, dst in (mapping or {}).items():
            if dst is not None and dst not in master:
                sys.exit(f"class_map.yaml: override target '{dst}' is not in master_classes")
            overrides[ds][norm(src)] = dst
    return master, lookup, overrides


def read_names(ds_dir):
    for fn in ("data.yaml", "dataset.yaml"):
        f = ds_dir / fn
        if f.exists():
            names = yaml.safe_load(f.read_text(encoding="utf-8")).get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names, key=lambda x: int(x))]
            return list(names)
    return None


def find_split(ds_dir, split):
    for s in SPLITS[split]:
        img, lab = ds_dir / s / "images", ds_dir / s / "labels"
        if img.is_dir():
            return img, lab
        img, lab = ds_dir / "images" / s, ds_dir / "labels" / s
        if img.is_dir():
            return img, lab
    return None, None


def convert_line(parts):
    """Return (class_id, [xc, yc, w, h]) or None. Polygons are turned into boxes."""
    try:
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if len(vals) == 4:
        box = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        box = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    else:
        return None
    xc, yc, w, h = [min(max(v, 0.0), 1.0) for v in box]
    if w <= 0 or h <= 0:
        return None
    return cls, [xc, yc, w, h]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder containing one sub-folder per dataset")
    ap.add_argument("--out", required=True, help="output dataset folder")
    ap.add_argument("--map", default=str(Path(__file__).with_name("class_map.yaml")))
    ap.add_argument("--clean", action="store_true", help="delete --out first if it exists")
    ap.add_argument("--keep-empty", action="store_true",
                    help="also keep images that end up with no mapped objects (as background)")
    ap.add_argument("--auto-val", type=float, default=0.1,
                    help="if no dataset has a val split, move this fraction of train to val")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    master, lookup, overrides = load_class_map(args.map)
    master_id = {m: i for i, m in enumerate(master)}

    roots = sorted({p.parent for p in raw.rglob("data.yaml")})
    roots = [r for r in roots if out not in r.parents and r != out]
    if not roots:
        sys.exit(f"No dataset folders (with a data.yaml) found under {raw}")

    if out.exists():
        if args.clean:
            shutil.rmtree(out)
        elif any(out.iterdir()):
            sys.exit(f"{out} already exists and is not empty. Use --clean to overwrite.")
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_counts = Counter()
    kept_by_split = Counter()
    used_prefixes = set()

    print(f"Found {len(roots)} dataset(s):\n")
    for root in roots:
        ds = root.name
        prefix = re.sub(r"[^A-Za-z0-9_]+", "_", ds)
        if prefix in used_prefixes:
            prefix += f"_{len(used_prefixes)}"
        used_prefixes.add(prefix)

        names = read_names(root)
        if not names:
            print(f"[{ds}] SKIPPED: no class names in data.yaml\n")
            continue

        ds_over = overrides.get(ds, {})
        id_to_master = {}
        for i, n in enumerate(names):
            key = norm(n)
            if key in ds_over:
                id_to_master[i] = ds_over[key]
            else:
                id_to_master[i] = lookup.get(key)

        seen, kept = 0, 0
        unmapped = Counter()
        for split in SPLITS:
            img_dir, lab_dir = find_split(root, split)
            if img_dir is None:
                continue
            for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
                seen += 1
                lab = lab_dir / (img.stem + ".txt")
                lines_out = []
                if lab.exists():
                    for line in lab.read_text().splitlines():
                        parts = line.split()
                        if not parts:
                            continue
                        conv = convert_line(parts)
                        if conv is None or conv[0] >= len(names):
                            continue
                        cid, box = conv
                        target = id_to_master.get(cid)
                        if target is None:
                            unmapped[names[cid]] += 1
                            continue
                        lines_out.append(
                            f"{master_id[target]} " + " ".join(f"{v:.6f}" for v in box)
                        )
                        class_counts[target] += 1
                if not lines_out and not args.keep_empty:
                    continue
                new_stem = f"{prefix}__{img.stem}"
                shutil.copy2(img, out / "images" / split / (new_stem + img.suffix.lower()))
                (out / "labels" / split / (new_stem + ".txt")).write_text(
                    "\n".join(lines_out) + ("\n" if lines_out else "")
                )
                kept += 1
                kept_by_split[split] += 1

        print(f"[{ds}] images seen: {seen}, kept: {kept}")
        if unmapped:
            top = ", ".join(f"{n} ({c})" for n, c in unmapped.most_common())
            print(f"    dropped (not in class_map.yaml): {top}")
        print()

    # No validation images anywhere -> carve some out of train.
    if kept_by_split["val"] == 0 and kept_by_split["train"] > 0 and args.auto_val > 0:
        random.seed(args.seed)
        imgs = sorted((out / "images" / "train").iterdir())
        pick = random.sample(imgs, max(1, int(len(imgs) * args.auto_val)))
        for img in pick:
            lab = out / "labels" / "train" / (img.stem + ".txt")
            shutil.move(str(img), out / "images" / "val" / img.name)
            shutil.move(str(lab), out / "labels" / "val" / lab.name)
        kept_by_split["val"] = len(pick)
        kept_by_split["train"] -= len(pick)
        print(f"No val split found: moved {len(pick)} train images to val.\n")

    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: m for i, m in enumerate(master)},
    }
    if kept_by_split["test"] > 0:
        data["test"] = "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    print("=== Final dataset ===")
    print(f"train: {kept_by_split['train']}  val: {kept_by_split['val']}  test: {kept_by_split['test']}")
    print("\ninstances per class:")
    for m in master:
        n = class_counts[m]
        flag = "   <-- very few examples; add more data" if n < 100 else ""
        print(f"  {master_id[m]:>2} {m:<22}{n:>8}{flag}")
    print(f"\nWrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
