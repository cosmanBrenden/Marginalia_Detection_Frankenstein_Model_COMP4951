import os
import argparse
import warnings
# warnings.filterwarnings("ignore")

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# Get cli args
parser = argparse.ArgumentParser(description="Test/Predict Script")
parser.add_argument("--ckpt-path", type=str, required=True,
    help="Path to checkpoint (.pth) saved by train.py")
parser.add_argument("--test-img-dir", type=str, required=True,
    help="Directory containing test PNG images")
parser.add_argument("--csv", type=str, default=None,
    help="CSV with ground-truth boxes (set to '' to skip IoU)")
parser.add_argument("--score-thresh", type=float, default=0.1,
    help="Minimum detection score to keep (default: 0.1)")
parser.add_argument("--nms-thresh", type=float, default=0.5,
    help="IoU threshold for NMS (default: 0.5)")
parser.add_argument("--results-dir", type=str, default="./results/",
    help="Where to write annotated images")
parser.add_argument("--no-visualize", action="store_true",
    help="Skip writing annotated images")
parser.add_argument("--num-classes", type=int, default=2,
    help="Number of classes incl. background (default: 2)")
args = parser.parse_args()


def preprocessing(image_id: str, path: str) -> torch.Tensor:
    """Read a PNG from disk and return a C*H*W float32 tensor in [0, 1]."""
    img = cv.imread(os.path.join(path, f"{image_id}.png"))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}{image_id}.png")
    img = img / 255.0
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

def generate_data(image_list, box_df, image_path):
    """Compiles test data."""
    data = []
    for fname in image_list:
        if not (fname.endswith(".png") or fname.endswith(".jpg") or fname.endswith(".jpeg")):
            continue
        image_id = fname[:-4]

        # Ground-truth boxes (may be absent for pure test images)
        sub_df = box_df[box_df["number"] == int(image_id)] if box_df is not None else pd.DataFrame()
        n_box = len(sub_df)
        coords = []
        for i in range(n_box):
            r = sub_df.iloc[i]
            coords.append(torch.tensor(
                [int(r["xmin_scaled"]), int(r["ymin_scaled"]),
                 int(r["xmax_scaled"]), int(r["ymax_scaled"])],
                dtype=torch.float32))

        if n_box > 1:
            box_tensor = torch.stack(coords, dim=0)
        elif n_box == 1:
            box_tensor = coords[0].view(1, 4)
        else:
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)

        data.append({
            "data":     preprocessing(image_id, image_path),
            "boxes":    box_tensor,
            "labels":   torch.ones(n_box, dtype=torch.int64),
            "image_id": image_id,
        })
    return data


class MarginaliaDataset(Dataset):
    """Wrapper object for test data"""
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample = self.data[index]
        target = {"boxes": sample["boxes"], "labels": sample["labels"]}
        return sample["data"], target, sample["image_id"]

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    """Collates batch tensors"""
    return tuple(zip(*batch))


def build_model(num_classes: int = 2) -> torchvision.models.detection.FasterRCNN:
    """Construct the same Faster R-CNN head used in train.py."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> dict:
    """
    Load a checkpoint saved by train.py.

    The checkpoint is a dict with keys:
        "model"          model state_dict
        "optimizer"      optimizer state_dict
        "epoch"          last completed epoch
        "previous_best"  best IoU seen so far
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # train.py always wraps weights under "model"
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        meta = {k: v for k, v in ckpt.items() if k != "model"}
    else:
        # Bare state_dict fallback (just in case)
        state_dict = ckpt
        meta = {}

    # Strip accidental "module." prefix from DataParallel/DDP wrapping
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "", 1)] = v

    model.load_state_dict(cleaned)
    return meta


# IoU helpers

def bbox_iou(boxA, boxB) -> float:
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))
    iW, iH = xB - xA + 1, yB - yA + 1
    if iW <= 0 or iH <= 0:
        return -1.0
    inter = iW * iH
    aA = (float(boxA[2]) - float(boxA[0]) + 1) * (float(boxA[3]) - float(boxA[1]) + 1)
    aB = (float(boxB[2]) - float(boxB[0]) + 1) * (float(boxB[3]) - float(boxB[1]) + 1)
    return inter / float(aA + aB - inter)


def match_multiple_boxes(boxes_target, boxes_predicted) -> float:
    """Mean best-match IoU across all ground-truth boxes."""
    total = 0.0
    for i in range(len(boxes_target)):
        max_iou = 0.0
        for j in range(len(boxes_predicted)):
            try:
                curr = bbox_iou(boxes_target[i], boxes_predicted[j])
                if curr > max_iou:
                    max_iou = curr
            except IndexError:
                pass
        total += max_iou
    denom = max(len(boxes_target), len(boxes_predicted))
    return total / denom if denom > 0 else 0.0


# Visualization

def visualize_prediction_and_target(image_id: str, target_boxes, predicted_boxes, image_dir: str, results_dir: str) -> None:
    """Draws ground-truth (blue) and predicted (red) boxes onto the image."""
    img_path = os.path.join(image_dir, f"{image_id}.png")
    image = cv.imread(img_path)
    if image is None:
        print(f"  [warn] Could not read {img_path} for visualization.")
        return
    image = np.asarray(image).copy()

    if target_boxes is not None and len(target_boxes) > 0:
        tb = target_boxes.cpu().numpy() if isinstance(target_boxes, torch.Tensor) else np.array(target_boxes)
        for box in tb:
            cv.rectangle(image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 0, 255), 1) # Red = ground truth, BGR

    if predicted_boxes is not None and len(predicted_boxes) > 0:
        pb = predicted_boxes.cpu().numpy() if isinstance(predicted_boxes, torch.Tensor) else np.array(predicted_boxes)
        for box in pb:
            cv.rectangle(image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 0, 0), 1) # Blue = prediction, BGR

    os.makedirs(results_dir, exist_ok=True)
    cv.imwrite(os.path.join(results_dir, f"prediction_{image_id}.png"), image)


# Evaluation

def evaluate_and_visualize(results: list, box_df, image_dir: str, results_dir: str, visualize: bool = True) -> float:
    """
    Compute mean IoU over all results and (optionally) write annotated images.

    Parameters
    ----------
    results : list of (image_id, predicted_boxes, scores)
    box_df : DataFrame with columns [number, xmin_scaled, ymin_scaled,
        xmax_scaled, ymax_scaled]
        Pass None to skip IoU computation (no GT available).
    """
    if box_df is None:
        print("No ground-truth CSV supplied — skipping IoU computation.")
        if visualize:
            for image_id, pred_boxes, _ in results:
                visualize_prediction_and_target(image_id, None, pred_boxes,
                                                image_dir, results_dir)
        return float("nan")

    iou_list = []
    for image_id, pred_boxes, _ in results:
        sub_df = box_df[box_df["number"] == int(image_id)]
        n_box = len(sub_df)
        if n_box == 0:
            if visualize:
                visualize_prediction_and_target(image_id, None, pred_boxes, image_dir, results_dir)
            continue

        coords = []
        for i in range(n_box):
            r = sub_df.iloc[i]
            coords.append(torch.tensor(
                [int(r["xmin_scaled"]), int(r["ymin_scaled"]),
                 int(r["xmax_scaled"]), int(r["ymax_scaled"])],
                dtype=torch.float32))

        target_boxes = torch.stack(coords) if n_box > 1 else coords[0].view(1, 4)

        iou_mean = match_multiple_boxes(target_boxes, pred_boxes.cpu())
        iou_list.append(iou_mean)

        if visualize:
            visualize_prediction_and_target(image_id, target_boxes, pred_boxes,
                                            image_dir, results_dir)

    if not iou_list:
        return 0.0
    return sum(iou_list) / len(iou_list)


# Main

def main():
    

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Data
    image_list = [f for f in os.listdir(args.test_img_dir) if f.endswith(".png")]
    if not image_list:
        raise RuntimeError(f"No PNG files found in {args.test_img_dir}")

    box_df = None
    if args.csv and os.path.exists(args.csv):
        box_df = pd.read_csv(args.csv)[
            ["number", "xmin_scaled", "ymin_scaled", "xmax_scaled", "ymax_scaled"]]
    else:
        print(f"[info] CSV not found or not supplied — IoU evaluation will be skipped.")

    test_data = generate_data(image_list, box_df, args.test_img_dir)
    test_dataset = MarginaliaDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, pin_memory=True)
    print(f"Loaded {len(test_dataset)} test images.")

    # Model
    model = build_model(num_classes=args.num_classes)
    meta = load_checkpoint(model, args.ckpt_path, device)
    model = model.to(device)
    model.eval()

    if meta:
        print(f"Checkpoint info — epoch: {meta.get('epoch', '?')}, "
              f"best IoU: {meta.get('previous_best', '?'):.4f}")

    # Inference
    results = []
    with torch.no_grad():
        for images, targets, ids in test_loader:
            try:
                images_gpu = [img.to(device) for img in images]
                outputs = model(images_gpu)

                for i, output in enumerate(outputs):
                    boxes = output["boxes"]
                    scores = output["scores"]
                    labels = output["labels"]

                    # NMS — note: score_thresh is applied first so NMS sees only
                    # confident boxes; then low-score boxes are filtered again.
                    keep = torchvision.ops.nms(boxes, scores, args.nms_thresh)
                    boxes = boxes[keep]
                    scores = scores[keep]

                    # Score threshold
                    above = scores >= args.score_thresh
                    boxes = boxes[above]
                    scores = scores[above]

                    results.append((ids[i], boxes, scores))
                    print(f"{ids[i]}: {len(boxes)} detections "
                          f"(scores {scores.cpu().tolist()})")

            except Exception as e:
                print(f"[error] {e}")

    # Evaluation
    mean_iou = evaluate_and_visualize(
        results,
        box_df=box_df,
        image_dir=args.test_img_dir,
        results_dir=args.results_dir,
        visualize=not args.no_visualize,
    )

    if not np.isnan(mean_iou):
        print(f"\nMean IoU over {len(results)} images: {mean_iou:.4f}")
    if not args.no_visualize:
        print(f"Annotated images written to: {args.results_dir}")


if __name__ == "__main__":
    main()