# Training the VisionAI hazard model

Fine-tunes your `yolov8n.pt` on Indian road hazards (potholes, stray animals,
construction barriers, open drains/manholes, auto-rickshaws, ...) while keeping
person and vehicle detection.

## Quick start (Google Colab, free T4 GPU)

1. **Download datasets** as zips into `MyDrive/VisionAI/raw_datasets/`
   - Roboflow Universe sets: open the dataset page > *Download Dataset* > format **YOLOv8** > *download zip*.
   - RDD2022 India (potholes): `RDD2022_India.zip` from https://github.com/sekilab/RoadDamageDetector
   - DriveIndia (best single source, **but request-only**, see below): https://tihan.iith.ac.in/TiAND.html
   - More sources are listed under **Datasets** below.
2. **Upload this `training/` folder** to `MyDrive/VisionAI/training/`.
3. Open **`train_colab.ipynb`** in Colab, set *Runtime > T4 GPU*, edit the SETTINGS cell, run all cells.
4. **Review the merge report and label previews** before training (steps 3 and 4 in the notebook).
5. Take the results from `MyDrive/VisionAI/runs/<run>/weights/`:
   - `best.pt` for the laptop script (`src/test.py`)
   - `best_ncnn_model/` for the Raspberry Pi (`src/rpi_test.py`), exported at `PI_IMGSZ` (default 320). Keep `imgsz=` in `rpi_test.py` equal to it.

## Datasets

| Need | Source |
|------|--------|
| General Indian roads + potholes, animals, cones, barriers (YOLO format) | DriveIndia: https://tihan.iith.ac.in/TiAND.html |
| Potholes (Pascal-VOC XML) | RDD2022 India: https://github.com/sekilab/RoadDamageDetector |
| Animals, barricades, manholes | Roboflow: https://universe.roboflow.com/indian-road-dataset/indian-roads-detection |
| Animals, barricades, potholes | Roboflow: https://universe.roboflow.com/vedant-madankar/cow_detection-paz2h |
| Cattle on roads | Roboflow: https://universe.roboflow.com/object-detection-naauz/cowsonroad |
| Construction barriers, cones, standing water | Roboflow: https://universe.roboflow.com/safebd/collect-road-data |
| Animals, construction boards | Roboflow: https://universe.roboflow.com/potholedetection-ceiz9/road_obstacle_detection |
| Open manholes / drains | Roboflow: https://universe.roboflow.com/aibased-solution-for-realtime-detection-of-road-anomalies-d6eay/open-manholes |
| Open manholes, water puddles | Roboflow: https://universe.roboflow.com/coastalseven/project-nfdh6 |
| Curbs, drains, stairs, cones (blind navigation) | Roboflow: https://universe.roboflow.com/stevens-wj2qk/cv_dataset |
| Indian driving scenes (segmentation-oriented) | IDD: http://idd.insaan.iiit.ac.in/ |

Preview labels on Roboflow sets before trusting them; quality varies.

## Files

| File | Purpose |
|------|---------|
| `train_colab.ipynb` | End-to-end notebook: unpack, convert, merge, preview, train, evaluate, export |
| `class_map.yaml` | Master class list + how each dataset's class names map onto it. **Edit this to add/remove classes.** |
| `merge_datasets.py` | Merges many YOLO datasets (Roboflow or Ultralytics layout) into one; reports dropped classes and per-class counts |
| `voc_to_yolo.py` | Converts Pascal-VOC XML (RDD2022) to YOLO, keeping only potholes by default |
| `preview_labels.py` | Draws labels on sample images so you can verify the mapping |
| `train.py` | Same training as the notebook, for a local GPU/CPU |
| `requirements-training.txt` | Packages for local use |

Local use instead of Colab:

```bash
pip install -r requirements-training.txt
python merge_datasets.py --raw raw --out datasets/visionai --clean
python preview_labels.py --data datasets/visionai/data.yaml
python train.py --data datasets/visionai/data.yaml --export-ncnn
```

## Things to know

- **Unmapped classes are dropped, loudly.** The merge report lists every source class
  it ignored. Typical decisions: `bike` (motorcycle or bicycle?), `manhole` (closed covers are
  deliberately ignored; map it only if that dataset's manholes are open), `rider`, `tractor`.
  Add them under `aliases` or per-dataset `overrides` in `class_map.yaml`.
- **DriveIndia is request-based, not an instant download.** The TiHAN portal asks you to fill a
  Google Form (and a signed EULA for the camera data), then sends a download link by email after
  approval. Request it early and train your first model without it. It is also large (about 67,000
  full-HD images), so on free Colab/Drive use a subset.
- **DriveIndia class order.** The merge script reads class names from each dataset's
  `data.yaml`. If DriveIndia ships without one, create it yourself using the class order from
  the dataset's own documentation, then run `preview_labels.py` to confirm the names land on the
  right objects. Do not guess the order.
- **Rare classes will be weakest** (potholes, drains, animals). Add your own images from the
  glasses camera at walking height; the public sets are mostly dash-cam footage.
- **Licences.** DriveIndia is for academic, non-commercial use; Roboflow datasets each have
  their own licence. Fine for a college project; re-check before commercialising.

## After training: update your scripts

1. `MODEL_PATH = "best.pt"` (or the NCNN folder on the Pi).
2. Add widths for the new class names, exactly as spelled in `class_map.yaml`, otherwise they
   fall back to `DEFAULT_OBJECT_WIDTH`:

```python
OBJECT_WIDTHS.update({          # rough starting guesses: measure and calibrate
    "auto_rickshaw": 1.4, "cart": 1.0, "animal": 0.5, "pothole": 0.5,
    "open_manhole_drain": 0.6, "barricade": 1.0, "traffic_cone": 0.3,
    "construction_vehicle": 2.5, "speed_bump": 3.0, "water_puddle": 1.0,
})
```

Distance from box width is only meaningful for things with a fairly fixed size (people, cars,
cones). For potholes, puddles and speed bumps it is unreliable, which is exactly what the
smart stick's ground-level ultrasonic sensor is for.

## Testing status

`merge_datasets.py`, `voc_to_yolo.py` and `preview_labels.py` were tested on synthetic datasets
mimicking the Roboflow, Ultralytics and RDD-XML layouts. `train.py` and the notebook's training,
evaluation and export cells could not be run where this kit was written (no GPU/Ultralytics there),
so expect to fix small things on first run.
