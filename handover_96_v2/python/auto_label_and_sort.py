"""
auto_label_and_sort.py

Semi-automated labeling: runs your trained model on each photo in a flat
input folder, shows you its prediction (including Joker calls), and you
either press Enter to accept it or type the correct class code to
override. Files get copied into the right class folder automatically.

Also does a train/val split as it sorts (set --val_split 0 to keep
everything in train, e.g. when you'll augment + split afterward with
split_augmented_train_val.py).

Class code format: <rank><suit-letter>, e.g. "AS", "10H", "KD", "JOKER".
Ranks: 2 3 4 5 6 7 8 9 10 J Q K A     Suits: S C H D

Usage:
  python auto_label_and_sort.py \
      --checkpoint checkpoints/best_dualhead_model.pt \
      --input_dir ./my_deck_photos \
      --out_dir ./my_deck_labeled \
      --val_split 0.15
"""

import argparse
import random
import re
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

try:
    import pillow_heif
    pillow_heif.register_heif_opener()  # lets PIL.Image.open() read .heic/.heif directly
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False


IMG_SIZE = 48  # must match train_card_cnn_dualhead.py exactly
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["S", "C", "H", "D"]
BLACK_SUIT_IDX = [0, 1]   # S, C
RED_SUIT_IDX = [2, 3]     # H, D


class DualHeadCardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        feat_dim = 16 * (IMG_SIZE // 4) * (IMG_SIZE // 4)
        self.fc_shared = nn.Linear(feat_dim + 1, 64)
        self.fc_rank = nn.Linear(64, len(RANKS))
        self.fc_suit = nn.Linear(64, len(SUITS))
        self.fc_joker = nn.Linear(64, 2)  # 0=not joker, 1=joker

    def forward(self, x, color_flag):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = torch.cat([x, color_flag.unsqueeze(1)], dim=1)
        shared = F.relu(self.fc_shared(x))
        return self.fc_rank(shared), self.fc_suit(shared), self.fc_joker(shared)


infer_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def compute_color_flag(pil_img: Image.Image) -> float:
    """Corner-cropped, white-balanced chromaticity ratio with a brightness
    floor -- see recognize_card.py for full rationale. White balancing makes
    this robust to a camera colour cast (e.g. the D8M's greenish tint)."""
    img = pil_img.convert("RGB")
    w, h = img.size
    CROP_FRAC = 0.20
    img = img.crop((0, 0, int(w * CROP_FRAC), int(h * CROP_FRAC)))
    img = img.resize((128, 128))
    arr = np.asarray(img, dtype=np.float32)

    # white-patch normalization -- must match recognize_card.py exactly
    flat = arr.reshape(-1, 3)
    white_point = np.maximum(np.percentile(flat, 95, axis=0), 1.0)
    gains = white_point.mean() / white_point
    arr = np.clip(arr * gains[np.newaxis, np.newaxis, :], 0, 255)

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    BRIGHTNESS_FLOOR = 30
    total = r + g + b + BRIGHTNESS_FLOOR
    chromaticity = (r / total) * 255.0
    signal = np.percentile(chromaticity, 98)
    return 1.0 if signal > 99 else 0.0  # augmented 1:2 crops: black max 82.9, red min 115.9


def is_valid_code(code: str) -> bool:
    code = code.strip().upper()
    if code == "JOKER":
        return True
    return bool(re.match(r"^(10|[2-9]|[JQKA])[SCHD]$", code))


@torch.no_grad()
def predict_code(model, pil_img, device):
    color = compute_color_flag(pil_img)
    x = infer_transform(pil_img).unsqueeze(0).to(device)
    color_t = torch.tensor([color], dtype=torch.float32).to(device)
    rank_logits, suit_logits, joker_logits = model(x, color_t)

    if joker_logits.argmax(dim=1).item() == 1:
        return "JOKER"

    rank_idx = rank_logits.argmax(dim=1).item()
    # restrict the 4-way suit head to the pair matching the detected colour
    allowed = RED_SUIT_IDX if color >= 0.5 else BLACK_SUIT_IDX
    probs = suit_logits[0]
    suit_idx = max(allowed, key=lambda i: probs[i].item())
    return f"{RANKS[rank_idx]}{SUITS[suit_idx]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", default="./my_deck_labeled")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadCardCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if HEIC_SUPPORTED:
        exts |= {".heic", ".heif"}
    paths = sorted(p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in exts)

    heic_files = [p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in (".heic", ".heif")]
    if heic_files and not HEIC_SUPPORTED:
        print(f"found {len(heic_files)} .heic file(s) but pillow-heif isn't installed.")
        print("run: pip install pillow-heif\nthen re-run this script.\n")

    print(f"found {len(paths)} images in {args.input_dir}\n")
    print("For each image: press Enter to ACCEPT the prediction, type a code")
    print("(e.g. 'KD', '10H', 'JOKER') to CORRECT it, or 's' to skip the file.\n")

    accepted, corrected, skipped = 0, 0, 0

    for path in paths:
        pil_img = Image.open(path)
        pred_code = predict_code(model, pil_img, device)

        user_in = input(f"{path.name:<30} model guess: {pred_code:<8} > ").strip().upper()

        if user_in == "S":
            skipped += 1
            continue
        elif user_in == "":
            final_code = pred_code
            accepted += 1
        elif is_valid_code(user_in):
            final_code = user_in
            corrected += 1
        else:
            print(f"  '{user_in}' doesn't look like a valid code -- skipping this file")
            skipped += 1
            continue

        split = "val" if random.random() < args.val_split else "train"
        dest_dir = Path(args.out_dir) / split / final_code
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dest_dir / path.name)

    print(f"\ndone. accepted: {accepted}  corrected: {corrected}  skipped: {skipped}")
    print(f"labeled dataset written to {args.out_dir}/train and {args.out_dir}/val")


if __name__ == "__main__":
    main()
