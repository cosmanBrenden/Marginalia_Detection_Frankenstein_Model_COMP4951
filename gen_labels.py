import argparse
import csv
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main():
    # Get cli args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_csv", required=True)
    args = parser.parse_args()

    # Ensure output dir exists to write to
    os.makedirs(args.output_dir, exist_ok=True)

    # Read boxes: {image_id: [(xmin,ymin,xmax,ymax), ...]}
    boxes_by_id = {}

    df = pd.read_csv(args.label_csv)
    for i, row in df.iterrows():
        img_id = int(row["number"])
        xmin = row["xmin_scaled"]
        xmax = row["xmax_scaled"]
        ymin = row["ymin_scaled"]
        ymax = row["ymax_scaled"]
        boxes_by_id.setdefault(img_id, []).append((xmin, ymin, xmax, ymax))

    print(f"Loaded {len(boxes_by_id)} images with boxes")

    # Process each image
    for img_id, boxes in tqdm(boxes_by_id.items()):
        # Find image file (PNG or JPG)
        img_path = None
        for ext in _IMG_EXTS:
            p = Path(args.input_dir) / f"{img_id}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            print(f"Missing image {img_id}")
            continue

        # Get image size (should be 500x500, but read it)
        with Image.open(img_path) as im:
            w, h = im.size

        # Create mask (0 = background)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw boxes (1 = marginalia)
        for xmin, ymin, xmax, ymax in boxes:
            # Clamp to image boundaries
            x1 = max(0, xmin)
            y1 = max(0, ymin)
            x2 = min(w, xmax + 1)   # +1 because Python slice end is exclusive
            y2 = min(h, ymax + 1)
            # if x1 < x2 and y1 < y2:
            mask[int(y1):int(y2), int(x1):int(x2)] = 1

        # Save as indexed PNG (black/white)
        out_img = Image.fromarray(mask, mode="P")
        out_img.putpalette([0,0,0, 255,255,255] + [0,0,0]*254)
        out_img.save(Path(args.output_dir) / f"{img_id}.png")

    print("Done")

if __name__ == "__main__":
    main()