import os
import time
import math
import random
import logging
import argparse
import warnings
import json

# warnings.filterwarnings("ignore")

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Get cli args
parser = argparse.ArgumentParser(description="Semi-supervised Frankenstein Project Marginalia - SemiOVS Training Script")
parser.add_argument("--data-path", type=str, required=True,
    help="Root of in-distribution labeled data. Must contain images/ and labels/ sub-folders.")
parser.add_argument("--proportion-of-labeled", type=float, default=0.5,
    help="(Optional) The proportion of labeled to unlabeled data, e.g. 0.55 -> 55%% of the data is labeled.")
parser.add_argument("--ood-path", type=str, default=None,
    help="(Optional) Root of OOD labeled data, same layout as --data-path. If omitted, OOD branch is disabled.")
parser.add_argument("--save-path", type=str, default="./checkpoints",
    help="Directory for TensorBoard logs and model checkpoints.")
parser.add_argument("--keep-prop", type=float, default=1,
    help="(Optional) Proportion of the dataset to use from the start, e.g. 0.42 -> first 42%% of the dataset.")
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--conf-thresh",type=float, default=0.7,
    help="Pseudo-label confidence threshold.")
parser.add_argument("--no-amp", action="store_true",
    help="Disable automatic mixed precision.")



# Utility


class DictAverageMeter:
    """Running average tracker for a dictionary of scalar values."""

    def __init__(self):
        self.avgs: dict   = {}
        self.counts: dict = {}

    def update(self, d: dict):
        for k, v in d.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            if k not in self.avgs:
                self.avgs[k]   = 0.0
                self.counts[k] = 0
            self.avgs[k]   += val
            self.counts[k] += 1

    def reset(self):
        self.avgs   = {}
        self.counts = {}

    def __str__(self):
        return ", ".join(
            f"{k}: {v / self.counts[k]:.4f}" for k, v in self.avgs.items()
        )



# Data loading  (mask-based labels)


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def generate_data_from_masks(data_root: str) -> list:
    """
    Load (image, label-mask) pairs from:

        data_root/
            images/   <-- input images, any common format
            labels/   <-- grayscale or binary masks, matching filenames

    Each non-zero connected component in the mask becomes one bounding box.
    The mask filename is matched to the image filename stem; extension may
    differ (e.g. image = foo.jpg, mask = foo.png - both are tried).

    Returns
    -------
    list of dicts:
        {
            "data":     FloatTensor  C*H*W  in [0, 1]
            "boxes":    FloatTensor  N*4    (x1, y1, x2, y2)
            "labels":   LongTensor   N      (all 1 - foreground class)
            "image_id": str          filename stem
        }
    """
    images_dir = os.path.join(data_root, "images")
    labels_dir = os.path.join(data_root, "labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images/ sub-folder not found in: {data_root}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"labels/ sub-folder not found in: {data_root}")

    data = []
    for fname in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in _IMG_EXTS:
            continue

        # Image 
        img_bgr = cv.imread(os.path.join(images_dir, fname))
        if img_bgr is None:
            continue
        img_tensor = torch.tensor(
            img_bgr / 255.0, dtype=torch.float32
        ).permute(2, 0, 1)          # C*H*W

        #  Mask  (try same extension first, then common alternatives) 
        mask = None
        for try_ext in [ext, ".png", ".jpg", ".bmp"]:
            candidate = os.path.join(labels_dir, stem + try_ext)
            if os.path.exists(candidate):
                mask = cv.imread(candidate, cv.IMREAD_GRAYSCALE)
                break

        #  Boxes from connected components 
        boxes = []
        if mask is not None:
            binary = (mask > 0).astype(np.uint8)
            n_comp, _, stats, _ = cv.connectedComponentsWithStats(
                binary, connectivity=8)
            for comp_id in range(1, n_comp):            # 0 = background
                x  = int(stats[comp_id, cv.CC_STAT_LEFT])
                y  = int(stats[comp_id, cv.CC_STAT_TOP])
                w  = int(stats[comp_id, cv.CC_STAT_WIDTH])
                h  = int(stats[comp_id, cv.CC_STAT_HEIGHT])
                if w > 0 and h > 0:
                    boxes.append([float(x), float(y),
                                  float(x + w), float(y + h)])

        n_box      = len(boxes)
        box_tensor = (torch.tensor(boxes, dtype=torch.float32)
                      if n_box > 0
                      else torch.zeros((0, 4), dtype=torch.float32))

        data.append({
            "data":     img_tensor,
            "boxes":    box_tensor,
            "labels":   torch.ones(n_box, dtype=torch.int64),
            "image_id": stem,
        })

    return data



# Datasets


class MarginaliaDataset(Dataset):
    """Labeled dataset: returns (image_tensor, target_dict, image_id)."""

    def __init__(self, data):
        self.data      = data
        self.n_samples = len(data)

    def __getitem__(self, index):
        sample = self.data[index]
        target = {"boxes": sample["boxes"], "labels": sample["labels"]}
        return sample["data"], target, sample["image_id"]

    def __len__(self):
        return self.n_samples


class UnlabeledMarginaliaDataset(Dataset):
    """
    Unlabeled dataset for the semi-supervised loop.

    Returns per sample:
        img_w        - weakly augmented (identity)
        img_s1       - strongly augmented view 1
        img_s2       - strongly augmented view 2
        ignore_mask  - zeros (H*W); API parity placeholder
        cutmix_box1  - random crop region [x1,y1,x2,y2] for CutMix pass 1
        cutmix_box2  - random crop region [x1,y1,x2,y2] for CutMix pass 2
    """

    def __init__(self, data, img_size=(500, 500)):
        self.data      = data
        self.img_size  = img_size
        self.n_samples = len(data)

    def __getitem__(self, index):
        img  = self.data[index]["data"]
        img_w   = img
        img_s1  = self._strong_augment(img)
        img_s2  = self._strong_augment(img)
        ignore_mask = torch.zeros(self.img_size, dtype=torch.long)
        cutmix_box1 = self._random_cutmix_box()
        cutmix_box2 = self._random_cutmix_box()
        return img_w, img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def _strong_augment(self, img: torch.Tensor) -> torch.Tensor:
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        aug    = random.choice(["noise", "brightness", "colorjitter"])
        if aug == "noise":
            img_np = noisy(img_np)
        elif aug == "brightness":
            img_np = brightness(img_np)
        else:
            img_np = colorjitter(img_np)
        return torch.tensor(img_np / 255.0, dtype=torch.float32).permute(2, 0, 1)

    def _random_cutmix_box(self) -> torch.Tensor:
        H, W = self.img_size
        lam   = random.uniform(0.3, 0.7)
        cut_h = int(H * math.sqrt(1.0 - lam))
        cut_w = int(W * math.sqrt(1.0 - lam))
        cy = random.randint(0, H)
        cx = random.randint(0, W)
        y1 = max(0, cy - cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        y2 = min(H, y1 + cut_h)
        x2 = min(W, x1 + cut_w)
        return torch.tensor([x1, y1, x2, y2], dtype=torch.long)

    def __len__(self):
        return self.n_samples


class OODMarginaliaDataset(Dataset):
    """
    Out-of-distribution labeled dataset.

    Returns per sample:
        (img_w, target)  - weak-aug image + GT boxes/labels
        img_s1, img_s2   - strongly augmented views
        ignore_mask, cutmix_box1, cutmix_box2
    """

    def __init__(self, data, img_size=(500, 500)):
        self.data      = data
        self._ulb      = UnlabeledMarginaliaDataset(data, img_size)
        self.n_samples = len(data)

    def __getitem__(self, index):
        sample = self.data[index]
        target = {"boxes": sample["boxes"], "labels": sample["labels"]}
        img_w, img_s1, img_s2, ignore_mask, cb1, cb2 = self._ulb[index]
        return (img_w, target), img_s1, img_s2, ignore_mask, cb1, cb2

    def __len__(self):
        return self.n_samples



# Collate functions


def collate_fn_labeled(batch):
    return tuple(zip(*batch))


def collate_fn_unlabeled(batch):
    img_w  = torch.stack([b[0] for b in batch])
    img_s1 = torch.stack([b[1] for b in batch])
    img_s2 = torch.stack([b[2] for b in batch])
    ignore = torch.stack([b[3] for b in batch])
    cb1    = torch.stack([b[4] for b in batch])
    cb2    = torch.stack([b[5] for b in batch])
    return img_w, img_s1, img_s2, ignore, cb1, cb2


def collate_fn_ood(batch):
    imgs    = [b[0][0] for b in batch]
    targets = [b[0][1] for b in batch]
    img_s1  = torch.stack([b[1] for b in batch])
    img_s2  = torch.stack([b[2] for b in batch])
    ignore  = torch.stack([b[3] for b in batch])
    cb1     = torch.stack([b[4] for b in batch])
    cb2     = torch.stack([b[5] for b in batch])
    return (imgs, targets), img_s1, img_s2, ignore, cb1, cb2



# CutMix helpers


def cutmix_img_(img_dst: torch.Tensor,
                img_src: torch.Tensor,
                cutmix_boxes: torch.Tensor) -> None:
    for b in range(img_dst.shape[0]):
        x1, y1, x2, y2 = cutmix_boxes[b].tolist()
        img_dst[b, :, y1:y2, x1:x2] = img_src[b, :, y1:y2, x1:x2]


def cutmix_detection_targets(targets_dst: list,
                              targets_src: list,
                              cutmix_boxes: torch.Tensor,
                              device) -> list:
    def _overlap_ratio(boxes, rx1, ry1, rx2, ry2):
        ix1     = torch.clamp(boxes[:, 0], min=rx1)
        iy1     = torch.clamp(boxes[:, 1], min=ry1)
        ix2     = torch.clamp(boxes[:, 2], max=rx2)
        iy2     = torch.clamp(boxes[:, 3], max=ry2)
        inter   = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
        area    = ((boxes[:, 2] - boxes[:, 0]).clamp(min=1) *
                   (boxes[:, 3] - boxes[:, 1]).clamp(min=1))
        return inter / area

    merged = []
    for t_dst, t_src, box in zip(targets_dst, targets_src, cutmix_boxes):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]

        b_dst = t_dst["boxes"].to(device).float()
        l_dst = t_dst["labels"].to(device)
        b_src = t_src["boxes"].to(device).float()
        l_src = t_src["labels"].to(device)

        if len(b_dst) > 0:
            keep  = _overlap_ratio(b_dst, x1, y1, x2, y2) < 0.5
            b_dst, l_dst = b_dst[keep], l_dst[keep]

        if len(b_src) > 0:
            keep  = _overlap_ratio(b_src, x1, y1, x2, y2) >= 0.5
            b_src, l_src = b_src[keep], l_src[keep]

        if len(b_dst) > 0 and len(b_src) > 0:
            all_boxes  = torch.cat([b_dst, b_src], dim=0)
            all_labels = torch.cat([l_dst, l_src], dim=0)
        elif len(b_dst) > 0:
            all_boxes, all_labels = b_dst, l_dst
        elif len(b_src) > 0:
            all_boxes, all_labels = b_src, l_src
        else:
            all_boxes  = torch.zeros((0, 4), dtype=torch.float32, device=device)
            all_labels = torch.zeros(0,       dtype=torch.int64,   device=device)

        merged.append({"boxes": all_boxes, "labels": all_labels})

    return merged



# Pseudo-label helpers


def predictions_to_targets(predictions: list, conf_thresh: float, device) -> list:
    pseudo = []
    for pred in predictions:
        keep   = pred["scores"] >= conf_thresh
        boxes  = pred["boxes"][keep]
        labels = pred["labels"][keep]
        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32, device=device)
            labels = torch.zeros(0,       dtype=torch.int64,   device=device)
        pseudo.append({"boxes": boxes.to(device), "labels": labels.to(device)})
    return pseudo


def compute_detection_loss(model, images: list, targets: list, device) -> torch.Tensor:
    valid_imgs, valid_tgts = [], []
    for img, tgt in zip(images, targets):
        if len(tgt["boxes"]) > 0:
            valid_imgs.append(img.to(device) if img.device != device else img)
            # valid_tgts.append(tgt)
            valid_tgts.append({k: v.to(device) for k, v in tgt.items()})
    if not valid_imgs:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return sum(model(valid_imgs, valid_tgts).values())



# Data augmentation helpers


def noisy(img, noise_type="gauss"):
    if noise_type == "gauss":
        image = img.copy()
        gauss = np.random.normal(0, 0.7, image.shape).astype("uint8")
        return cv.add(image, gauss)
    image = img.copy()
    prob  = 0.05
    cs    = image.shape[2] if len(image.shape) == 3 else 1
    black = np.array([0]*cs, dtype="uint8") if cs > 1 else 0
    white = np.array([255]*cs, dtype="uint8") if cs > 1 else 255
    probs = np.random.random(image.shape[:2])
    image[probs < prob / 2]     = black
    image[probs > 1 - prob / 2] = white
    return image


def brightness(img, low=0.2, high=0.6):
    value = random.uniform(low, high)
    hsv   = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * value, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value, 0, 255)
    return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)


def colorjitter(img, cj_type="c"):
    if cj_type == "b":
        value = np.random.choice([-50, -40, -30, 30, 40, 50])
        hsv   = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        if value >= 0:
            v[v > 255 - value]  = 255
            v[v <= 255 - value] += value
        else:
            v[v < abs(value)]  = 0
            v[v >= abs(value)] -= abs(value)
        return cv.cvtColor(cv.merge((h, s, v)), cv.COLOR_HSV2BGR)
    elif cj_type == "s":
        value = np.random.choice([-50, -40, -30, 30, 40, 50])
        hsv   = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        if value >= 0:
            s[s > 255 - value]  = 255
            s[s <= 255 - value] += value
        else:
            s[s < abs(value)]  = 0
            s[s >= abs(value)] -= abs(value)
        return cv.cvtColor(cv.merge((h, s, v)), cv.COLOR_HSV2BGR)
    else:
        contrast = random.randint(40, 100)
        dummy    = np.clip(np.int16(img) * (contrast / 127 + 1) - contrast + 10, 0, 255)
        return np.uint8(dummy)



# Evaluation helpers


def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    iW, iH = xB - xA + 1, yB - yA + 1
    if iW <= 0 or iH <= 0:
        return -1.0
    iA = iW * iH
    aA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    aB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return iA / float(aA + aB - iA)


def match_multiple_boxes(boxes_target, boxes_predicted):
    total = 0
    for i in range(len(boxes_target)):
        max_iou = 0
        for j in range(len(boxes_predicted)):
            try:
                curr = bbox_iou(boxes_target[i], boxes_predicted[j])
                if curr > max_iou:
                    max_iou = curr
            except IndexError:
                pass
        total += max_iou
    return total / max(len(boxes_target), len(boxes_predicted))


def evaluate_IOU_score(results) -> float:
    """
    Parameters
    ----------
    results : list of (image_id, predicted_boxes_tensor, gt_boxes_tensor)
        GT boxes come from the in-memory val dataset - no CSV dependency.

    Returns
    -------
    float  - mean IoU across all val images that have >=1 GT box.
    """
    iou_list = []
    for _, predicted_boxes, target_boxes in results:
        if len(target_boxes) == 0:
            continue
        iou_list.append(
            match_multiple_boxes(target_boxes, predicted_boxes.to("cpu")))
    return float(sum(iou_list) / len(iou_list)) if iou_list else 0.0



# Main
if __name__ == "__main__":
    args = parser.parse_args()

    # Ensure save path exists
    os.makedirs(args.save_path, exist_ok=True)

    use_ood = args.ood_path is not None

    cfg = {
        "lr":          args.lr,
        "lr_multi":    10.0,
        "momentum":    0.9,
        "weight_decay": 5e-4,
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "amp":         not args.no_amp,
        "conf_thresh": args.conf_thresh,
        "lambda":      1.0,
        "save_path":   args.save_path,

    }

    os.makedirs(cfg["save_path"], exist_ok=True)

    #  Logging & TensorBoard 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    logger = logging.getLogger("train")
    writer = SummaryWriter(cfg["save_path"])

    rank = 0

    torch.cuda.empty_cache()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"OOD branch: {'enabled  -> ' + args.ood_path if use_ood else 'disabled'}")

    #  In-distribution data
    # Load all labeled in-dist samples, then split 90/10 train/val.
    # The training portion is further split: first half -> labeled loader,
    # second half -> unlabeled loader (labels withheld at the loader level).
    all_indist = generate_data_from_masks(args.data_path)

    random.seed(42)
    random.shuffle(all_indist)

    n = len(all_indist)
    val_split = math.ceil(0.9 * n)
    train_data = all_indist[:val_split]
    val_data = all_indist[val_split:]
    train_data = train_data[:math.ceil(args.keep_prop * len(train_data))]

    split_pt = math.ceil(len(train_data) * args.proportion_of_labeled)
    use_unl = split_pt < len(train_data)
    labeled_data = train_data[:split_pt]      # has GT boxes -> supervised loss
    unlabeled_data = train_data[split_pt:]      # boxes withheld -> pseudo-labels

    logger.info(
        f"In-dist  -  labeled: {len(labeled_data)},  "
        f"unlabeled: {len(unlabeled_data)},  val: {len(val_data)}")

    # OOD data (optional)
    if use_ood:
        ood_data = generate_data_from_masks(args.ood_path)
        logger.info(f"OOD  -  labeled: {len(ood_data)}")

    # DataLoaders
    trainloader_l = DataLoader(
        MarginaliaDataset(labeled_data),
        batch_size=cfg["batch_size"], num_workers=8,
        collate_fn=collate_fn_labeled, pin_memory=True,
        shuffle=True, drop_last=True)

    if use_unl:
        trainloader_u = DataLoader(
            UnlabeledMarginaliaDataset(unlabeled_data),
            batch_size=cfg["batch_size"], num_workers=8,
            collate_fn=collate_fn_unlabeled, pin_memory=True,
            shuffle=True, drop_last=True)

    if use_ood:
        trainloader_u_ood = DataLoader(
            OODMarginaliaDataset(ood_data),
            batch_size=cfg["batch_size"], num_workers=4,
            collate_fn=collate_fn_ood, pin_memory=True,
            shuffle=True, drop_last=True)

    val_dl = DataLoader(
        MarginaliaDataset(val_data),
        batch_size=cfg["batch_size"],
        collate_fn=collate_fn_labeled, pin_memory=True, 
        num_workers=0)

    # Model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model = model.to(device)

    # Optimizer 
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    head_params  = [p for p in model.parameters()
                    if id(p) not in backbone_ids and p.requires_grad]

    optimizer = torch.optim.SGD(
        [{"params": list(model.backbone.parameters()), "lr": cfg["lr"]},
         {"params": head_params,                        "lr": cfg["lr"] * cfg["lr_multi"]}],
        lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"])

    # State
    total_epochs  = cfg["epochs"]
    if use_unl:
        total_iters   = len(trainloader_u) * total_epochs
    else:
        total_iters   = len(trainloader_l) * total_epochs
    conf_thresh   = cfg["conf_thresh"]
    epoch         = -1
    previous_best = 0.0
    ETA           = 0.0

    # Resume
    ckpt_path = os.path.join(cfg["save_path"], "latest.pth")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        epoch         = ck["epoch"]
        previous_best = ck["previous_best"]
        logger.info("************ Loaded checkpoint from epoch %i" % epoch)

    # Load or init loss lists for training and validation loss
    if os.path.exists("./train_losses.json"):
        with open("./train_losses.json", "r") as f:
            loss_per_epoch = list(json.load(f))
    else:
        loss_per_epoch = []

    if os.path.exists("./val_losses.json"):
        with open("./val_losses.json", "r") as f:
            val_loss_per_epoch = list(json.load(f))
    else:
        val_loss_per_epoch = []
 
    # EPOCH LOOP
    for epoch in range(epoch + 1, total_epochs):
        start_time = time.time()
        logger.info(
            "===========> Epoch: {:}, LR: {:.5f}, "
            "Previous best: {:.4f}, ETA: {:.2f}M".format(
                epoch, optimizer.param_groups[0]["lr"], previous_best, ETA / 60))

        log_avg = DictAverageMeter()
        epoch_loss_sum = 0.0
        val_loss_sum = 0.0
        epoch_iters = 0

        # Build the zipped loader for this epoch
        # Two independent references to trainloader_u -> different batches each
        # step for the (main, mix) unlabeled pair, mirroring semiovs.
        if use_ood:
            loader = zip(trainloader_l, trainloader_u, trainloader_u,
                         trainloader_u_ood)
        elif use_unl:
            loader = zip(trainloader_l, trainloader_u, trainloader_u)
        else:
            loader = trainloader_l

        for i, batch in enumerate(loader):
            t0 = time.time()

            if use_ood:
                (img_x, targets_x, _), \
                (img_u_w, img_u_s1, img_u_s2, _, cutmix_box1, cutmix_box2), \
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _, _), \
                ((img_u_w_ood_list, targets_u_ood_raw),
                  img_u_s1_ood, img_u_s2_ood, _, _, _) = batch
            elif use_unl:
                (img_x, targets_x, _), \
                (img_u_w, img_u_s1, img_u_s2, _, cutmix_box1, cutmix_box2), \
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _, _) = batch
            else:
                (img_x, targets_x, _) = batch

            # Move to device 
            img_x     = torch.stack(list(img_x)).to(device)
            targets_x = [{k: v.to(device) for k, v in t.items()} for t in targets_x]

            if use_unl:
                img_u_w      = img_u_w.to(device)
                img_u_s1     = img_u_s1.to(device)
                img_u_s2     = img_u_s2.to(device)
                cutmix_box1  = cutmix_box1.to(device)
                cutmix_box2  = cutmix_box2.to(device)
                img_u_w_mix  = img_u_w_mix.to(device)
                img_u_s1_mix = img_u_s1_mix.to(device)
                img_u_s2_mix = img_u_s2_mix.to(device)

            if use_ood:
                img_u_w_ood   = torch.stack(list(img_u_w_ood_list)).to(device)
                targets_u_ood = [{k: v.to(device) for k, v in t.items()}
                                 for t in targets_u_ood_raw]
                img_u_s1_ood  = img_u_s1_ood.to(device)
                img_u_s2_ood  = img_u_s2_ood.to(device)

            # Pseudo-label generation (eval, no grad)
            model.eval()
            if use_unl:
                with torch.cuda.amp.autocast(enabled=cfg["amp"]):
                    with torch.no_grad():
                        preds_u_w     = model(list(img_u_w))
                        preds_u_w_mix = model(list(img_u_w_mix))

                pseudo_u_w     = predictions_to_targets(preds_u_w,     conf_thresh, device)
                pseudo_u_w_mix = predictions_to_targets(preds_u_w_mix, conf_thresh, device)

            # CutMix images (in-place) 
            if use_ood:
                cutmix_img_(img_u_s1_ood, img_u_s1, cutmix_box1)
                cutmix_img_(img_u_s2_ood, img_u_s2, cutmix_box2)
            if use_unl:
                cutmix_img_(img_u_s1, img_u_s1_mix, cutmix_box1)
                cutmix_img_(img_u_s2, img_u_s2_mix, cutmix_box2)

            # CutMix targets 
            if use_unl:
                pseudo_s1 = cutmix_detection_targets(
                    pseudo_u_w, pseudo_u_w_mix, cutmix_box1, device)
                pseudo_s2 = cutmix_detection_targets(
                    pseudo_u_w, pseudo_u_w_mix, cutmix_box2, device)

            if use_ood:
                pseudo_s1_ood = cutmix_detection_targets(
                    targets_u_ood, pseudo_u_w, cutmix_box1, device)
                pseudo_s2_ood = cutmix_detection_targets(
                    targets_u_ood, pseudo_u_w, cutmix_box2, device)

            # Loss (train mode) 
            model.train()

            with torch.cuda.amp.autocast(enabled=cfg["amp"]):

                # Get supervised loss
                loss_x = compute_detection_loss(model, list(img_x), targets_x, device)
                # Get unsupervised in distribution loss
                if use_unl:
                    loss_u_s1 = compute_detection_loss(model, list(img_u_s1), pseudo_s1, device)
                    loss_u_s2 = compute_detection_loss(model, list(img_u_s2), pseudo_s2, device)
                    loss_u_w_fp = compute_detection_loss(model, list(img_u_w), pseudo_u_w, device)

                    mask_ratio = (sum(1 for t in pseudo_u_w if len(t["boxes"]) > 0)
                        / max(len(pseudo_u_w), 1))
                # Get ood loss
                if use_ood:
                    loss_u_s1_ood   = compute_detection_loss(
                        model, list(img_u_s1_ood), pseudo_s1_ood, device)
                    loss_u_s2_ood   = compute_detection_loss(
                        model, list(img_u_s2_ood), pseudo_s2_ood, device)
                    loss_u_w_fp_ood = compute_detection_loss(
                        model, list(img_u_w_ood),  targets_u_ood, device)

                    loss_ood   = cfg["lambda"] * (
                        loss_u_s1_ood * 0.25 + loss_u_s2_ood * 0.25 + loss_u_w_fp_ood * 0.5)
                    total_loss = (
                        loss_ood + loss_x
                        + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5
                    ) / (2.0 + cfg["lambda"])
                # Calculate loss with unsupervised but no ood
                elif use_unl:
                    # No OOD branch: standard semi-supervised loss only
                    total_loss = (
                        loss_x
                        + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5
                    ) / 2.0
                # Get pure unsupervised loss if no unsupervised training occured
                else:
                    total_loss = loss_x

            # Backward 
            optimizer.zero_grad()
            if cfg["amp"]:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            torch.cuda.empty_cache()

            # Logging 
            log_entry = {
                "iter_time":  time.time() - t0,
                "Total_loss": total_loss,
                "Loss_x":     loss_x,
                
            }
            # Log losses
            if use_unl:
                log_entry["Loss_s"] = (loss_u_s1 + loss_u_s2) / 2.0
                log_entry["Loss_w_fp"] = loss_u_w_fp
                log_entry["Mask_ratio"] = mask_ratio
            if use_ood:
                log_entry["Loss_s_ood"]    = (loss_u_s1_ood + loss_u_s2_ood) / 2.0
                log_entry["Loss_w_fp_ood"] = loss_u_w_fp_ood
            log_avg.update(log_entry)

            # add total loss to epoch loss for average calculation
            epoch_loss_sum += total_loss.item()
            epoch_iters    += 1

            # Poly LR schedule (per-iteration, mirrors semiovs)
            if use_unl:
                iters = epoch * len(trainloader_u) + i
            else:
                iters = epoch * len(trainloader_l) + i
            lr    = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg["lr_multi"]

            for k, v in log_avg.avgs.items():
                writer.add_scalar("train/" + k, v / max(log_avg.counts[k], 1), iters)

            if use_unl:
                if (i % max(len(trainloader_u) // 8, 1) == 0) and rank == 0:
                    logger.info(f"Iters: {i},  " + str(log_avg))
                    log_avg.reset()
            else:
                if (i % max(len(trainloader_l) // 8, 1) == 0) and rank == 0:
                    logger.info(f"Iters: {i},  " + str(log_avg))
                    log_avg.reset()

        # Epoch-end evaluation
        model.eval()
        val_results = []
        val_losses = []
        val_iters = 0
        with torch.no_grad():
            for val_imgs, val_targets, val_ids in val_dl:
                # compute_detection_loss(model, list(img_x), targets_x, device)
                val_imgs = [img.to(device) for img in val_imgs]
                preds = model(val_imgs)
                model.train()
                val_losses.append(compute_detection_loss(model, list(val_imgs), val_targets, device))
                model.eval()
                for img_id, pred, tgt in zip(val_ids, preds, val_targets):
                    val_results.append((img_id, pred["boxes"], tgt["boxes"]))
                val_iters += 1
            val_loss_sum += sum(val_losses)

        epoch_iou = evaluate_IOU_score(val_results) if val_results else 0.0
        logger.info("***** Evaluation ***** >>>> Mean IoU: {:.4f}".format(epoch_iou))
        writer.add_scalar("eval/mean_IoU", epoch_iou, epoch)

        # Checkpoint 
        is_best       = epoch_iou > previous_best
        previous_best = max(epoch_iou, previous_best)
        checkpoint    = {
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "epoch":         epoch,
            "previous_best": previous_best,
        }
        torch.save(checkpoint, os.path.join(cfg["save_path"], "latest.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(cfg["save_path"], "best.pth"))

        mean_epoch_loss = epoch_loss_sum / max(epoch_iters, 1)
        mean_val_loss_per_epoch = val_loss_sum / max(val_iters, 1)

        

        loss_per_epoch.append(float(mean_epoch_loss))
        val_loss_per_epoch.append(float(mean_val_loss_per_epoch))

        with open("./train_losses.json", "w") as f:
            json.dump(loss_per_epoch, f)
        with open("./val_losses.json", "w") as f:
            json.dump(val_loss_per_epoch, f)
        logger.info(f"Epoch {epoch} mean total loss: {mean_epoch_loss:.4f}")

        end_time       = time.time()
        ETA            = (total_epochs - (epoch + 1)) * (end_time - start_time)

    # Loss curve 
    if loss_per_epoch:
        plt.figure()
        plt.plot(range(len(loss_per_epoch)), loss_per_epoch, color="orange")
        plt.plot(range(len(val_loss_per_epoch)), val_loss_per_epoch, color="blue")
        plt.legend(["Training Loss", "Val Loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss per Epoch")
        plt.savefig("loss_per_epoch.png")
        plt.show()

    writer.close()