"""
Step 1: Download the dataset and turn it into a face-crop classification dataset.

Dataset: "Face Mask Detection" by Larxel (andrewmvd) on Kaggle
  - 853 images with Pascal VOC XML boxes, 3 classes
  - License: CC0 1.0 Public Domain (confirm on the dataset page before redistributing)
  https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

Usage:
  python prepare_data.py                       # auto-download via kagglehub
  python prepare_data.py --raw path/to/folder  # folder with images/ and annotations/
"""
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]


def get_raw_dir(arg):
    if arg:
        return Path(arg)
    import kagglehub
    return Path(kagglehub.dataset_download("andrewmvd/face-mask-detection"))


def find_dir(root: Path, name: str) -> Path:
    for p in root.rglob(name):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Could not find '{name}/' inside {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default="data/faces")
    ap.add_argument("--pad", type=float, default=0.15, help="context padding around each box")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = get_raw_dir(args.raw)
    img_dir, ann_dir = find_dir(raw, "images"), find_dir(raw, "annotations")
    out = Path(args.out)

    # Split by IMAGE (not by face) so faces from one photo never leak across splits.
    xmls = sorted(ann_dir.glob("*.xml"))
    random.Random(args.seed).shuffle(xmls)
    n = len(xmls)
    splits = {
        "train": xmls[: int(0.8 * n)],
        "val": xmls[int(0.8 * n): int(0.9 * n)],
        "test": xmls[int(0.9 * n):],
    }

    counts = {}
    for split, files in splits.items():
        for c in CLASSES:
            (out / split / c).mkdir(parents=True, exist_ok=True)
        for xml_path in files:
            root = ET.parse(xml_path).getroot()
            img_path = img_dir / root.findtext("filename")
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            for i, obj in enumerate(root.findall("object")):
                label = obj.findtext("name")
                if label not in CLASSES:
                    continue
                bb = obj.find("bndbox")
                x1, y1, x2, y2 = (int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
                w, h = x2 - x1, y2 - y1
                if w < 12 or h < 12:  # skip tiny, unusable faces
                    continue
                px, py = int(w * args.pad), int(h * args.pad)
                crop = img.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)))
                crop.save(out / split / label / f"{xml_path.stem}_{i}.jpg", quality=95)
                counts[(split, label)] = counts.get((split, label), 0) + 1

    print("\nCrops per split/class:")
    for split in splits:
        print(f"  {split}: " + ", ".join(f"{c}={counts.get((split, c), 0)}" for c in CLASSES))
    print(f"\nDone. Dataset written to {out}/")


if __name__ == "__main__":
    main()
