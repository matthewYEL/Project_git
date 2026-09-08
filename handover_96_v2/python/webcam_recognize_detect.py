"""
webcam_recognize_detect.py

Webcam card recognition WITH automatic card detection.

Instead of feeding the whole camera frame to the model, this finds the card
in the frame, perspective-corrects it to a canonical upright rectangle, and
works from that. Two things get better as a result:

  1. COLOUR DETECTION becomes reliable. The old fixed "top-left 20% of the
     frame" crop assumed the card sat in a particular place at a particular
     scale -- which is why it grabbed background on webcam close-ups and
     misread red cards as black. Once the card is warped to a known
     rectangle, the rank/suit index is at a KNOWN location, so the crop
     always lands on real ink.

  2. FRAMING becomes consistent. Every detected card is normalised to the
     same size and orientation regardless of how it was held.

Detection is classical CV (threshold -> contours -> 4-corner polygon ->
perspective warp). No ML, cheap, and it maps well onto FPGA fabric later
if you want the same trick in hardware.

CAVEAT ON ACCURACY: the model was trained on photos where the card occupies
a modest portion of a larger frame. A tightly-warped card is a different
composition, so this script composes the warped card back onto a dark
canvas at a similar scale (--card_frac) to better match what the model
expects. If accuracy is still poor, the real fix is retraining on
canonically-cropped images rather than tuning this further.

Requires: pip install opencv-python

Usage:
  python webcam_recognize_detect.py --checkpoint checkpoints_8/best_dualhead_model.pt

Controls:
  q -- quit
  s -- save the current frame and the detected/warped card
  d -- toggle the detection debug view (shows the threshold mask)
"""

import argparse
from collections import deque, Counter
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


IMG_SIZE = 48  # must match train_card_cnn_dualhead.py exactly
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["S", "C", "H", "D"]
SUIT_NAMES = {"S": "Spade", "C": "Club", "H": "Heart", "D": "Diamond"}
BLACK_SUIT_IDX = [0, 1]   # S, C
RED_SUIT_IDX = [2, 3]     # H, D

# canonical warped card size (roughly a playing card's 2.5:3.5 aspect ratio)
CARD_W, CARD_H = 250, 350


# ---------------------------------------------------------------------------
# Model (must match train_card_cnn_dualhead.py exactly)
# ---------------------------------------------------------------------------

class DualHeadCardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        feat_dim = 16 * (IMG_SIZE // 4) * (IMG_SIZE // 4)
        self.fc_shared = nn.Linear(feat_dim + 1, 64)
        self.fc_rank = nn.Linear(64, len(RANKS))
        self.fc_suit = nn.Linear(64, len(SUITS))
        self.fc_joker = nn.Linear(64, 2)

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


# ---------------------------------------------------------------------------
# Card detection: threshold -> contours -> 4-corner polygon -> perspective warp
# ---------------------------------------------------------------------------

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points as [top-left, top-right, bottom-right, bottom-left].
    Uses the standard sum/difference trick: the top-left corner has the
    smallest x+y, bottom-right the largest; top-right has the smallest y-x,
    bottom-left the largest."""
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left
    return ordered


def detect_card(frame_bgr, debug=False):
    """Finds the card and returns (warped_card_bgr, box_for_drawing, mask).

    MUST match build_warped_dataset.py's detector exactly -- if the live
    detector and the one that generated the training data differ, the model
    sees a different composition at inference than it trained on, which is
    the exact failure this whole approach exists to eliminate.

    Uses minAreaRect on the largest bright rectangular contour rather than
    requiring approxPolyDP to land on exactly 4 vertices (rounded card
    corners and edge noise make that unreliable), and runs detection at
    reduced resolution for a smooth, stable outline."""
    H, W = frame_bgr.shape[:2]

    scale = 800.0 / max(H, W)
    if scale < 1.0:
        small = cv2.resize(frame_bgr, (int(W * scale), int(H * scale)))
    else:
        small, scale = frame_bgr, 1.0
    sh, sw = small.shape[:2]
    frame_area = sh * sw

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    bright_frac = float((mask > 0).mean())
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_area, best_box = 0.0, None
    rejects = {"small": 0, "not_rect": 0, "aspect": 0, "dark": 0}

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.05 * frame_area:
            rejects["small"] += 1
            continue

        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        if rw < 1 or rh < 1:
            rejects["not_rect"] += 1
            continue
        if area / (rw * rh) < 0.75:
            rejects["not_rect"] += 1
            continue

        ratio = min(rw, rh) / max(rw, rh)
        if not (0.50 <= ratio <= 0.95):
            rejects["aspect"] += 1
            continue

        box = cv2.boxPoints(rect)
        region = np.zeros((sh, sw), dtype=np.uint8)
        cv2.drawContours(region, [box.astype(np.int32)], -1, 255, -1)
        if cv2.mean(gray, mask=region)[0] < 100:
            rejects["dark"] += 1
            continue

        if area > best_area:
            best_area, best_box = area, box

    if best_box is None:
        if bright_frac > 0.80:   # card already fills the frame
            card = frame_bgr
            if card.shape[1] > card.shape[0]:
                card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
            full_box = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.int32)
            return cv2.resize(card, (CARD_W, CARD_H)), full_box, mask
        if debug:
            print(f"  [debug] no card. rejects: {rejects}  bright_frac={bright_frac:.2f}")
        return None, None, mask

    box_full = best_box / scale          # back to full-resolution coordinates
    src = order_corners(box_full)

    width = np.linalg.norm(src[1] - src[0])
    height = np.linalg.norm(src[3] - src[0])
    if width > height:
        src = np.roll(src, -1, axis=0)

    dst = np.array([[0, 0], [CARD_W - 1, 0],
                    [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    warped = cv2.warpPerspective(frame_bgr, M, (CARD_W, CARD_H))
    return warped, box_full.astype(np.int32), mask


# ---------------------------------------------------------------------------
# Colour detection -- now operating on the WARPED card, where the rank/suit
# index is at a known location
# ---------------------------------------------------------------------------

def white_balance(arr: np.ndarray) -> np.ndarray:
    """White-patch normalization -- must match recognize_card.py exactly.
    Cancels a camera colour cast by normalizing against the card's white face."""
    flat = arr.reshape(-1, 3)
    white_point = np.maximum(np.percentile(flat, 95, axis=0), 1.0)
    gains = white_point.mean() / white_point
    return np.clip(arr * gains[np.newaxis, np.newaxis, :], 0, 255)


def chromaticity_signal(patch_rgb: np.ndarray) -> float:
    """98th-percentile red chromaticity of a patch, with brightness floor."""
    arr = patch_rgb.astype(np.float32)
    arr = white_balance(arr)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    BRIGHTNESS_FLOOR = 30
    chromaticity = (r / (r + g + b + BRIGHTNESS_FLOOR)) * 255.0
    return float(np.percentile(chromaticity, 98))


def compute_color_flag_from_card(warped_bgr, threshold=110.0, debug=False):
    """Reads the rank/suit index colour from the warped card.

    Checks BOTH the top-left and bottom-right index corners and takes the
    stronger red signal -- playing cards print the index at both, so this is
    robust to the card being upside down (a 180-degree flip that contour
    detection can't distinguish and doesn't need to)."""
    warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    h, w = warped_rgb.shape[:2]
    ch, cw = int(h * 0.25), int(w * 0.25)

    top_left = warped_rgb[0:ch, 0:cw]
    bottom_right = warped_rgb[h - ch:h, w - cw:w]

    s_tl = chromaticity_signal(top_left)
    s_br = chromaticity_signal(bottom_right)
    signal = max(s_tl, s_br)

    if debug:
        print(f"  [debug] index-corner chromaticity: TL={s_tl:.1f} BR={s_br:.1f} "
              f"-> using {signal:.1f} (threshold {threshold})")

    return 1.0 if signal > threshold else 0.0


# ---------------------------------------------------------------------------
# Compose the warped card back onto a canvas matching training framing
# ---------------------------------------------------------------------------

def corner_crops(warped_bgr, frac_w=0.21, frac_h=0.30):
    """The two rank/suit index regions of a warped card, both upright.

    Must match build_warped_dataset.py exactly. 0.21 x 0.30 gives a 1:2 aspect,
    matching downsample_96x96.v's 96-column by 192-row hardware crop. The
    earlier 0.25 x 0.30 was 0.595 and no longer matches what the camera
    delivers."""
    h, w = warped_bgr.shape[:2]
    cw, ch = int(w * frac_w), int(h * frac_h)
    top_left = warped_bgr[0:ch, 0:cw]
    bottom_right = cv2.rotate(warped_bgr[h - ch:h, w - cw:w], cv2.ROTATE_180)
    return top_left, bottom_right


def compose_for_model(warped_bgr, card_frac=0.5, canvas_size=400):
    """The model was trained on photos where the card occupied a modest
    portion of a larger, mostly-dark frame. A tightly-cropped card is a
    different composition, so we paste the warped card onto a dark canvas at
    roughly the scale the model saw during training."""
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    target_h = int(canvas_size * card_frac)
    target_w = int(target_h * CARD_W / CARD_H)
    resized = cv2.resize(warped_bgr, (target_w, target_h))
    y0 = (canvas_size - target_h) // 2
    x0 = (canvas_size - target_w) // 2
    canvas[y0:y0 + target_h, x0:x0 + target_w] = resized
    return canvas


def decode_prediction(rank_idx, suit_idx, color, is_joker) -> str:
    if is_joker:
        return "Joker"
    suit_name = SUIT_NAMES[SUITS[suit_idx]]
    return f"{RANKS[rank_idx]} of {suit_name}s"


@torch.no_grad()
def _classify_image(model, img_bgr, color, device):
    """Runs one BGR image through the model and returns the decoded result."""
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    x = infer_transform(pil_img).unsqueeze(0).to(device)
    color_t = torch.tensor([color], dtype=torch.float32).to(device)

    rank_logits, suit_logits, joker_logits = model(x, color_t)
    rank_probs = F.softmax(rank_logits, dim=1)[0]
    suit_probs = F.softmax(suit_logits, dim=1)[0]
    joker_probs = F.softmax(joker_logits, dim=1)[0]

    rank_idx = rank_probs.argmax().item()
    # restrict the 4-way suit head to the pair matching the detected colour
    allowed = RED_SUIT_IDX if color >= 0.5 else BLACK_SUIT_IDX
    suit_idx = max(allowed, key=lambda i: suit_probs[i].item())
    is_joker = joker_probs.argmax().item() == 1

    name = decode_prediction(rank_idx, suit_idx, color, is_joker)
    conf = joker_probs[1].item() if is_joker else rank_probs[rank_idx].item()
    return name, conf, x


def predict_card(model, warped_bgr, device, card_frac, threshold,
                 debug=False, compose=True, corner_crop=False):
    color = compute_color_flag_from_card(warped_bgr, threshold=threshold, debug=debug)

    if corner_crop:
        # Classify BOTH index corners and keep the more confident. The
        # detector fits a rectangle, so it cannot distinguish a card from the
        # same card rotated 180 degrees -- one of these two crops is always
        # the right way up, and the confidence tells us which.
        tl, br = corner_crops(warped_bgr)
        name_tl, conf_tl, x_tl = _classify_image(model, tl, color, device)
        name_br, conf_br, x_br = _classify_image(model, br, color, device)
        if conf_tl >= conf_br:
            name, conf, x, model_img = name_tl, conf_tl, x_tl, tl
        else:
            name, conf, x, model_img = name_br, conf_br, x_br, br
        if debug:
            print(f"  [debug] corners: TL={name_tl} ({conf_tl:.2f})  "
                  f"BR={name_br} ({conf_br:.2f})  -> {name}")
    else:
        if compose:
            # pastes the warped card onto a dark canvas to imitate the
            # whole-frame composition the ORIGINAL model was trained on
            model_img = compose_for_model(warped_bgr, card_frac=card_frac)
        else:
            # feed the warped card directly -- correct once the model has been
            # retrained on warped cards, where the card fills the frame
            model_img = warped_bgr
        name, conf, x = _classify_image(model, model_img, color, device)

    # the exact 48x48 grayscale tensor the model sees, upscaled for viewing --
    # if the symbol isn't distinguishable here, the model cannot possibly
    # classify it, no matter how confident its output looks
    model_input_view = (x[0, 0].cpu().numpy() * 255).astype(np.uint8)
    model_input_view = cv2.resize(model_input_view, (240, 240), interpolation=cv2.INTER_NEAREST)

    return name, conf, model_img, model_input_view


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./webcam_captures")
    parser.add_argument("--card_frac", type=float, default=0.5,
                         help="fraction of the composed canvas the card fills -- tune to match "
                              "how much of the frame cards occupied in your training photos")
    parser.add_argument("--color_threshold", type=float, default=99.0,
                         help="chromaticity threshold for red/black. Recalibrate for a new camera "
                              "(e.g. the D8M) via calibrate_color_threshold.py")
    parser.add_argument("--debug", action="store_true", help="print colour detection numbers each frame")
    parser.add_argument("--width", type=int, default=1280,
                         help="requested capture width. Webcams often default to 640x480, which loses "
                              "pip detail once the card is far enough back for its corners to be in "
                              "frame. 1280 or 1920 widens the usable distance range if your camera "
                              "supports it.")
    parser.add_argument("--height", type=int, default=720, help="requested capture height")
    parser.add_argument("--vote_frames", type=int, default=15,
                         help="how many recent frames the majority vote considers. Larger = steadier "
                              "but slower to react when you swap cards.")
    parser.add_argument("--corner_crop", action="store_true",
                         help="classify the rank/suit index corner instead of the whole card. Use "
                              "this when the model was trained with build_warped_dataset.py "
                              "--corner_crop. Gives ~3x the linear resolution on the suit symbol, "
                              "which is what separates Heart from Diamond.")
    parser.add_argument("--no_compose", action="store_true",
                         help="feed the warped card straight to the model instead of pasting it onto "
                              "a dark canvas. Use this once you've retrained on warped cards via "
                              "build_warped_dataset.py -- the canvas trick exists only to imitate the "
                              "whole-frame composition the original model was trained on.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadCardCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"could not open camera index {args.camera_index} -- try --camera_index 1")
        return

    # request higher resolution -- the camera may silently ignore this and keep
    # its default, so we read back what we actually got
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"capture resolution: {actual_w}x{actual_h}"
          + ("" if (actual_w, actual_h) == (args.width, args.height)
             else f"  (requested {args.width}x{args.height} -- camera chose its own)"))

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print("q = quit | s = save | d = mask view | y = mark correct | n = mark wrong")
    print("hold a card still -- the STABLE line is a majority vote over recent frames\n")

    show_mask = False
    vote_window = deque(maxlen=args.vote_frames)
    last_announced = None
    n_correct, n_wrong = 0, 0
    stable_name, stable_conf = None, 0.0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("failed to grab frame")
            break

        warped, quad, mask = detect_card(frame_bgr, debug=args.debug)
        display = frame_bgr.copy()

        if warped is not None:
            cv2.drawContours(display, [quad], -1, (0, 255, 0), 3)
            name, conf, composed, model_input_view = predict_card(
                model, warped, device, args.card_frac, args.color_threshold,
                debug=args.debug, compose=not args.no_compose,
                corner_crop=args.corner_crop)
            vote_window.append(name)
            cv2.imshow("What the model actually sees (48x48)", model_input_view)

            # MAJORITY VOTE: a single frame can flicker on a marginal read.
            # Voting over recent frames gives a far more trustworthy answer,
            # and it's the same technique the real game logic will want during
            # its card-exposure window.
            counts = Counter(vote_window)
            stable_name, votes = counts.most_common(1)[0]
            stable_conf = votes / len(vote_window)

            # instantaneous reading (this frame only)
            cv2.putText(display, f"frame: {name} ({conf:.2f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if conf >= 0.7 else (0, 165, 255), 2)

            # stable reading (majority vote)
            stable_color = (0, 255, 0) if stable_conf >= 0.7 else (0, 165, 255)
            cv2.putText(display, f"STABLE: {stable_name}  [{votes}/{len(vote_window)} frames]",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, stable_color, 2)

            # running tally, if the user has been scoring
            if n_correct + n_wrong > 0:
                total = n_correct + n_wrong
                cv2.putText(display, f"scored: {n_correct}/{total} correct "
                                      f"({100*n_correct/total:.0f}%)",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # also label the warped window, since that's the one you watch
            warped_labeled = warped.copy()
            cv2.rectangle(warped_labeled, (0, 0), (CARD_W, 34), (0, 0, 0), -1)
            cv2.putText(warped_labeled, stable_name, (6, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, stable_color, 2)
            cv2.imshow("Detected card (warped)", warped_labeled)

            # console: only announce when the stable reading actually changes,
            # so the log stays readable instead of scrolling every frame
            if stable_name != last_announced and stable_conf >= 0.6:
                print(f"  -> {stable_name}   (vote {votes}/{len(vote_window)}, "
                      f"frame conf {conf:.2f})")
                last_announced = stable_name
        else:
            vote_window.clear()
            last_announced = None
            cv2.putText(display, "no card detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if show_mask:
            cv2.imshow("Detection mask", mask)

        cv2.imshow("Card Recognition (q quit, s save, d mask, y/n score)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Detection mask")
        elif key == ord('y') and stable_name is not None:
            n_correct += 1
            total = n_correct + n_wrong
            print(f"  [correct] {stable_name}   running: {n_correct}/{total} "
                  f"({100*n_correct/total:.0f}%)")
        elif key == ord('n') and stable_name is not None:
            n_wrong += 1
            total = n_correct + n_wrong
            print(f"  [WRONG]   {stable_name}   running: {n_correct}/{total} "
                  f"({100*n_correct/total:.0f}%)")
        elif key == ord('s'):
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            cv2.imwrite(str(Path(args.save_dir) / f"frame_{stamp}.jpg"), frame_bgr)
            if warped is not None:
                cv2.imwrite(str(Path(args.save_dir) / f"card_{stamp}.jpg"), warped)
            print(f"  saved frame_{stamp}.jpg"
                  + (f" and card_{stamp}.jpg" if warped is not None else ""))

    if n_correct + n_wrong > 0:
        total = n_correct + n_wrong
        print(f"\nsession score: {n_correct}/{total} correct ({100*n_correct/total:.1f}%)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
