"""
recognize_card.py

Inference script for the trained DualHeadCardCNN checkpoint (rank + suit +
joker heads). Model definition duplicated here from
train_card_cnn_dualhead.py so this script has no cross-file dependency --
if you retrain and change the architecture, update BOTH files.

Three modes:
  1. Single image:  --image <path>        predict one image, print result + confidence
  2. Label-free batch: --predict_dir <dir>  predict every file in a folder, no labels needed
  3. Labeled eval:   --eval_dir <dir>       ImageFolder layout, reports accuracy +
                                             confusion matrix + joker accuracy

Usage:
  python recognize_card.py --checkpoint checkpoints/best_dualhead_model.pt --image path/to/card.jpg
  python recognize_card.py --checkpoint checkpoints/best_dualhead_model.pt --predict_dir ./my_deck_photos
  python recognize_card.py --checkpoint checkpoints/best_dualhead_model.pt --eval_dir ./image_data/val
"""

import argparse
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

try:
    import pillow_heif
    pillow_heif.register_heif_opener()  # lets PIL.Image.open() read .heic/.heif directly
except ImportError:
    pass  # falls back to normal PIL behavior; .heic files will fail to open with a clear error


IMG_SIZE = 48  # must match train_card_cnn_dualhead.py exactly

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

# Absolute 4-way suit encoding (see train_card_cnn_dualhead.py for why this
# replaced the old colour-relative 2-way scheme). The colour flag is applied
# at DECODE time to restrict the choice to the matching pair.
# Colour threshold. RECALIBRATE whenever the crop region changes -- the
# chromaticity distribution moves with it. Use calibrate_color_threshold.py
# and pass --crop_frac 1.0 when measuring an already-cropped corner dataset.
# Colour threshold. RECALIBRATE whenever the crop region changes -- the
# chromaticity distribution moves with it. Use calibrate_color_threshold.py
# and pass --crop_frac 1.0 when measuring an already-cropped corner dataset.
DEFAULT_COLOR_THRESHOLD = 99.0

SUITS = ["S", "C", "H", "D"]
SUIT_NAMES = {"S": "Spade", "C": "Club", "H": "Heart", "D": "Diamond"}
SUIT_TO_IDX = {s: i for i, s in enumerate(SUITS)}
BLACK_SUIT_IDX = [0, 1]   # S, C
RED_SUIT_IDX = [2, 3]     # H, D


# ---------------------------------------------------------------------------
# Model (must match train_card_cnn_dualhead.py exactly)
# ---------------------------------------------------------------------------

class DualHeadCardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        feat_dim = 16 * (IMG_SIZE // 4) * (IMG_SIZE // 4)
        shared_out = 64
        self.fc_shared = nn.Linear(feat_dim + 1, shared_out)
        self.fc_rank = nn.Linear(shared_out, len(RANKS))
        self.fc_suit = nn.Linear(shared_out, len(SUITS))
        self.fc_joker = nn.Linear(shared_out, 2)  # 0=not joker, 1=joker

    def forward(self, x, color_flag):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = torch.cat([x, color_flag.unsqueeze(1)], dim=1)
        shared = F.relu(self.fc_shared(x))
        return self.fc_rank(shared), self.fc_suit(shared), self.fc_joker(shared)


# ---------------------------------------------------------------------------
# Preprocessing (must match training -- no augmentation at inference time)
# ---------------------------------------------------------------------------

infer_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def white_balance(arr: np.ndarray, debug: bool = False) -> np.ndarray:
    """Per-image white-patch normalization: estimates the scene's white point
    from the brightest pixels and rescales each channel so that white reads
    as neutral.

    WHY THIS MATTERS: the D8M camera (and most raw sensor pipelines without
    proper AWB) applies a global color cast -- typically greenish/grey, since
    Bayer sensors have twice as many green photosites and a naive demosaic
    leaves green over-weighted. A cast like that shifts EVERY pixel's
    chromaticity in the same direction, so an absolute threshold calibrated
    on neutrally-balanced phone photos will systematically under-read red on
    D8M frames (red cards falling below the threshold and being called black).

    A playing card gives us an ideal built-in reference: the card face is
    white by definition. Estimating the white point from the brightest ~5%
    of pixels and normalizing against it cancels the cast, so the resulting
    chromaticity is comparable across cameras -- and the threshold stops
    being camera-specific."""
    flat = arr.reshape(-1, 3)
    white_point = np.percentile(flat, 95, axis=0)          # approximates the white card face
    white_point = np.maximum(white_point, 1.0)              # guard against a fully dark crop
    gains = white_point.mean() / white_point                # normalize, preserving overall brightness
    if debug:
        print(f"  [debug] estimated white point R/G/B: "
              f"{white_point[0]:.1f}/{white_point[1]:.1f}/{white_point[2]:.1f}  "
              f"-> gains: {gains[0]:.3f}/{gains[1]:.3f}/{gains[2]:.3f}")
    return np.clip(arr * gains[np.newaxis, np.newaxis, :], 0, 255)


def compute_color_flag(pil_img: Image.Image, debug: bool = False, crop_corner: bool = True,
                        crop_frac: float = 0.20, use_white_balance: bool = True,
                        threshold: float = None) -> float:
    """Python-side stand-in for the FPGA red/black comparator. Returns
    1.0 for red, 0.0 for black.

    Pipeline: optional corner crop -> per-image white balance -> chromaticity
    ratio (R/(R+G+B)) with a brightness floor -> 98th percentile -> threshold.

    If crop_corner=True (default), restricts analysis to a fixed relative
    region of the frame (top-left corner by default) BEFORE computing
    chromaticity -- the corner rank/suit index is always plain ink, never
    decorated, unlike the rest of a face card. This assumes cards are
    reasonably consistently positioned/oriented in frame; if your capture
    setup has different framing (e.g. a webcam close-up, or a D8M mounted at
    a different distance), this crop can miss the card corner entirely.
    Recalibrate crop_frac against your actual setup via
    calibrate_color_threshold.py before trusting it.

    White balancing (use_white_balance=True) makes the metric robust to a
    camera color cast -- see white_balance() for the full rationale."""
    img = pil_img.convert("RGB")

    if crop_corner:
        w, h = img.size
        img = img.crop((0, 0, int(w * crop_frac), int(h * crop_frac)))

    img = img.resize((128, 128))
    arr = np.asarray(img, dtype=np.float32)

    if debug:
        print(f"  [debug] raw channel means R/G/B: {arr[...,0].mean():.1f}/"
              f"{arr[...,1].mean():.1f}/{arr[...,2].mean():.1f}  "
              f"(a large G excess here indicates a camera colour cast)")
    if use_white_balance:
        arr = white_balance(arr, debug=debug)

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    # BRIGHTNESS_FLOOR (not just a divide-by-zero guard): near-black pixels
    # (e.g. anti-aliased edges of black ink against white background) can have
    # tiny, visually meaningless channel noise like R=1,G=0,B=0 -- with a
    # near-zero epsilon, that noise produces a spurious ratio of ~255 ("100%
    # red"), because the denominator is almost zero. A meaningful floor damps
    # this: the same pixel becomes ~8 instead of ~255, while genuinely bright
    # red ink (R=200,G=30,B=30) barely changes (~176 either way). Confirmed
    # via a real misclassified photo (a clean black Ace of Clubs registering
    # chromaticity=255 with no visible red content anywhere in the image).
    BRIGHTNESS_FLOOR = 30
    total = r + g + b + BRIGHTNESS_FLOOR
    chromaticity = (r / total) * 255.0
    signal = np.percentile(chromaticity, 98)

    if debug:
        print(f"  [debug] chromaticity 98th percentile (corner-cropped={crop_corner}): {signal:.2f}  "
              f"(mean was: {chromaticity.mean():.2f})")

    # Threshold 110 was calibrated on the phone scan+camera domains BEFORE white
    # balancing was added. White balancing shifts the distribution, so re-run
    # calibrate_color_threshold.py (on both the old data and any D8M captures)
    # and update this if the separation point has moved.
    # Calibrated on WARPED cards (build_warped_dataset.py output): black tops
    # out at 82.9, red starts at 97.5, so 90 sits in the gap. The previous
    # value of 110 was above the bottom of the red range, which pushed the
    # dimmest red cards into the black class -- visible as Diamond->Club and
    # Heart->Spade errors in the suit confusion matrix.
    thr = DEFAULT_COLOR_THRESHOLD if threshold is None else threshold
    return 1.0 if signal > thr else 0.0


def parse_class_name(name: str):
    """Returns (rank_idx, absolute_suit_idx, color_flag, is_joker).
    Used only for ground-truth labels in --eval_dir mode."""
    name = name.strip().upper()
    if "JOKER" in name:
        return 0, 0, 0, 1
    m = re.match(r"^(10|[2-9]|[JQKA])([SCHD])$", name)
    if not m:
        return None
    rank_str, suit_letter = m.groups()
    rank_idx = RANKS.index(rank_str)
    color = 1 if suit_letter in ("H", "D") else 0
    return rank_idx, SUIT_TO_IDX[suit_letter], color, 0


def decode_prediction(rank_idx: int, suit_idx: int, color: float, is_joker: bool) -> str:
    if is_joker:
        return "Joker"
    rank_name = RANKS[rank_idx]
    suit_name = SUIT_NAMES[SUITS[suit_idx]]
    return f"{rank_name} of {suit_name}s"


@torch.no_grad()
def run_model(model, pil_img, device, debug=False, crop_corner=True, color_threshold=None):
    """Shared inference call -- returns decoded card name + all confidences.

    crop_corner=False when the images ARE already corner-index crops
    (build_warped_dataset.py --corner_crop). Cropping a corner out of a
    corner lands on the blank margin beside the glyph, reads neutral, and
    calls every red card black -- which shows up as Hearts->Clubs and
    Diamonds->Spades in the suit matrix."""
    color = compute_color_flag(pil_img, debug=debug, crop_corner=crop_corner,
                                threshold=color_threshold)
    x = infer_transform(pil_img).unsqueeze(0).to(device)
    color_t = torch.tensor([color], dtype=torch.float32).to(device)

    rank_logits, suit_logits, joker_logits = model(x, color_t)
    rank_probs = F.softmax(rank_logits, dim=1)[0]
    suit_probs = F.softmax(suit_logits, dim=1)[0]
    joker_probs = F.softmax(joker_logits, dim=1)[0]

    rank_idx = rank_probs.argmax().item()
    # Restrict the 4-way suit head to the pair matching the detected colour.
    # This is what makes the colour flag decisive rather than a hint buried in
    # the feature vector -- the failure mode it fixes was the head defaulting
    # to one class (Heart->Diamond, Spade->Club, never the reverse).
    allowed = RED_SUIT_IDX if color >= 0.5 else BLACK_SUIT_IDX
    suit_idx = max(allowed, key=lambda i: suit_probs[i].item())
    is_joker = joker_probs.argmax().item() == 1

    card_name = decode_prediction(rank_idx, suit_idx, color, is_joker)
    return {
        "card_name": card_name,
        "color": color,
        "rank_idx": rank_idx,
        "suit_idx": suit_idx,
        "is_joker": is_joker,
        "rank_conf": rank_probs[rank_idx].item(),
        "suit_conf": suit_probs[suit_idx].item(),
        "joker_conf": joker_probs[int(is_joker)].item(),
    }


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

def predict_image(model, image_path, device, debug=False,
                  crop_corner=True, color_threshold=None):
    pil_img = Image.open(image_path)
    result = run_model(model, pil_img, device, debug=debug,
                       crop_corner=crop_corner, color_threshold=color_threshold)

    print(f"predicted: {result['card_name']}")
    print(f"  joker confidence: {result['joker_conf']:.3f}")
    if not result["is_joker"]:
        print(f"  color flag: {'red' if result['color'] >= 0.5 else 'black'}")
        print(f"  rank confidence: {result['rank_conf']:.3f}")
        print(f"  suit confidence: {result['suit_conf']:.3f}")
        if result["rank_conf"] < 0.6 or result["suit_conf"] < 0.6:
            print("  WARNING: low confidence -- check lighting/crop/focus")
    elif result["joker_conf"] < 0.6:
        print("  WARNING: low confidence joker call -- check lighting/crop/focus")
    return result


# ---------------------------------------------------------------------------
# Label-free batch prediction
# ---------------------------------------------------------------------------

def predict_dir(model, predict_dir, device, crop_corner=True, color_threshold=None):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif"}
    paths = sorted(p for p in Path(predict_dir).iterdir() if p.suffix.lower() in exts)
    if not paths:
        print(f"no images found in {predict_dir}")
        return

    print(f"{'file':<30} {'prediction':<22} {'rank_conf':>10} {'suit_conf':>10} {'joker_conf':>11}")
    print("-" * 87)

    low_conf_files = []
    for path in paths:
        pil_img = Image.open(path)
        result = run_model(model, pil_img, device,
                           crop_corner=crop_corner, color_threshold=color_threshold)

        rc = f"{result['rank_conf']:.3f}" if not result["is_joker"] else "  --  "
        sc = f"{result['suit_conf']:.3f}" if not result["is_joker"] else "  --  "
        jc = f"{result['joker_conf']:.3f}"

        low_conf = (result["is_joker"] and result["joker_conf"] < 0.6) or \
                   (not result["is_joker"] and (result["rank_conf"] < 0.6 or result["suit_conf"] < 0.6))
        flag = "  <-- LOW CONF" if low_conf else ""
        print(f"{path.name:<30} {result['card_name']:<22} {rc:>10} {sc:>10} {jc:>11}{flag}")
        if low_conf:
            low_conf_files.append(path.name)

    if low_conf_files:
        print(f"\n{len(low_conf_files)} low-confidence prediction(s) -- worth checking by eye:")
        for f in low_conf_files:
            print(f"  {f}")


# ---------------------------------------------------------------------------
# Labeled batch evaluation + confusion matrix
# ---------------------------------------------------------------------------

def evaluate_dir(model, eval_dir, device, crop_corner=True, color_threshold=None):
    base = datasets.ImageFolder(eval_dir)
    n_ranks = len(RANKS)
    confusion = [[0] * n_ranks for _ in range(n_ranks)]  # [true][pred], non-joker only

    SUIT_LETTERS = ["S", "C", "H", "D"]  # fixed order for the confusion matrix
    suit_confusion = {t: {p: 0 for p in SUIT_LETTERS} for t in SUIT_LETTERS}

    def suit_letter(suit_idx, color=None):
        """Absolute index -> letter. The colour argument is no longer needed
        (suit indices are absolute now) but is accepted so existing call sites
        keep working."""
        return SUITS[suit_idx]

    total, rank_correct, suit_correct, full_correct = 0, 0, 0, 0
    joker_correct, joker_total, non_joker_total = 0, 0, 0
    true_joker_count = 0
    overall_correct = 0  # joker samples: joker call right; non-joker: rank+suit+not-joker right

    for path, class_idx in base.samples:
        class_name = base.classes[class_idx]
        parsed = parse_class_name(class_name)
        if parsed is None:
            continue  # unparseable class name
        true_rank, true_suit, true_color, true_joker = parsed

        pil_img = Image.open(path)
        result = run_model(model, pil_img, device,
                           crop_corner=crop_corner, color_threshold=color_threshold)
        total += 1

        if true_joker == 1:
            true_joker_count += 1
            joker_total += 1
            is_correct = result["is_joker"]
            joker_correct += int(is_correct)
            overall_correct += int(is_correct)
        else:
            non_joker_total += 1
            confusion[true_rank][result["rank_idx"]] += 1
            rank_ok = result["rank_idx"] == true_rank
            not_joker_ok = not result["is_joker"]

            # suit correctness must account for the DETECTED color, not just the raw
            # suit_idx -- both color pairs use indices {0,1}, so comparing suit_idx
            # alone can "match" by coincidence even when color detection was wrong,
            # silently masking real pipeline errors (confirmed bug: this previously
            # made suit_acc/full_card_acc completely insensitive to color threshold
            # changes, even though the confusion matrix showed real differences)
            true_letter = suit_letter(true_suit, true_color)
            pred_letter = suit_letter(result["suit_idx"], result["color"])
            suit_ok = pred_letter == true_letter

            rank_correct += int(rank_ok)
            suit_correct += int(suit_ok)
            full_correct += int(rank_ok and suit_ok)
            overall_correct += int(rank_ok and suit_ok and not_joker_ok)
            # also track whether the model correctly said "not joker" as part of joker accuracy
            joker_correct += int(not_joker_ok)
            joker_total += 1

            suit_confusion[true_letter][pred_letter] += 1

    print(f"\nevaluated {total} images ({non_joker_total} cards, {true_joker_count} jokers)")
    print(f"rank accuracy (non-joker):       {rank_correct/non_joker_total:.4f}" if non_joker_total else "no non-joker samples")
    print(f"suit+color accuracy (non-joker): {suit_correct/non_joker_total:.4f}" if non_joker_total else "")
    print(f"full card accuracy (non-joker):  {full_correct/non_joker_total:.4f}" if non_joker_total else "")
    print(f"joker head accuracy (all samples, incl. correctly saying 'not joker'): {joker_correct/joker_total:.4f}" if joker_total else "")
    print(f"overall pipeline accuracy (joker call + full card, combined): {overall_correct/total:.4f}")

    if non_joker_total:
        print("\nrank confusion matrix (rows=true, cols=predicted, non-joker samples only):")
        header = "      " + " ".join(f"{r:>4}" for r in RANKS)
        print(header)
        for i, row_label in enumerate(RANKS):
            row = " ".join(f"{confusion[i][j]:>4}" for j in range(n_ranks))
            print(f"{row_label:>5} {row}")

        print("\nsuit confusion matrix (rows=true, cols=predicted -- predicted axis uses DETECTED color):")
        header = "      " + " ".join(f"{s:>4}" for s in SUIT_LETTERS)
        print(header)
        for t in SUIT_LETTERS:
            row = " ".join(f"{suit_confusion[t][p]:>4}" for p in SUIT_LETTERS)
            print(f"{t:>5} {row}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", help="path to a single image for recognition")
    parser.add_argument("--eval_dir", help="ImageFolder-style labeled dir for accuracy + confusion matrix")
    parser.add_argument("--predict_dir", help="folder of unlabeled images -- prints a prediction per file")
    parser.add_argument("--debug", action="store_true", help="print raw color-detection stats (single-image mode)")
    parser.add_argument("--corner_dataset", action="store_true",
                         help="the images ARE already rank/suit index crops (built with "
                              "build_warped_dataset.py --corner_crop). Disables the extra corner "
                              "crop inside colour detection -- cropping a corner out of a corner "
                              "reads the blank margin and calls every red card black.")
    parser.add_argument("--color_threshold", type=float, default=None,
                         help=f"chromaticity threshold for red/black (default {DEFAULT_COLOR_THRESHOLD}). "
                              "Recalibrate with calibrate_color_threshold.py whenever the crop "
                              "region changes.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadCardCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if args.image:
        predict_image(model, args.image, device, debug=args.debug,
                      crop_corner=not args.corner_dataset,
                      color_threshold=args.color_threshold)
    elif args.eval_dir:
        evaluate_dir(model, args.eval_dir, device,
                     crop_corner=not args.corner_dataset,
                     color_threshold=args.color_threshold)
    elif args.predict_dir:
        predict_dir(model, args.predict_dir, device,
                    crop_corner=not args.corner_dataset,
                    color_threshold=args.color_threshold)
    else:
        print("provide one of: --image <path>, --eval_dir <labeled folder>, --predict_dir <unlabeled folder>")


if __name__ == "__main__":
    main()
