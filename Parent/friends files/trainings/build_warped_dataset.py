"""
build_warped_dataset.py

Runs the SAME card detector used by webcam_recognize_detect.py over an
existing class-labeled photo dataset, and writes out perspective-corrected
warped cards in a parallel folder structure.

WHY: the model was trained on whole-frame photos (card sitting inside a
larger scene). The live pipeline now feeds it a warped, tightly-cropped
card. Those are different compositions, and the model collapses to a
constant output on the unfamiliar one -- always the same rank, always the
same suit index, at high confidence.

Rather than distorting the live input to imitate the training photos, this
converts the TRAINING photos into the same warped form the live detector
produces. Existing labels stay valid (a warped 3 of Clubs is still a 3 of
Clubs), and training and inference compositions become identical by
construction.

Detection failures are expected on some photos -- a card clipped by the
frame edge or sitting on a low-contrast background won't produce a clean
quadrilateral. The per-class success rate is reported so you can see
whether a class ends up too thin to train on.

Usage:
  python build_warped_dataset.py --data_dir ./image_data/combined --out_dir ./image_data/warped

Then the usual pipeline:
  python augment_dataset.py --data_dir ./image_data/warped --copies 20 --max_rotation 5 --max_shear 0.02 --min_crop_frac 0.97 --color_jitter_strength 0
  python split_augmented_train_val.py --data_dir ./image_data/warped --out_dir ./image_data/warped_final --val_count 3
  python train_card_cnn_dualhead.py --data_dir ./image_data/warped_final --epochs 15 --lr 1e-4 --init_checkpoint checkpoints_8/best_dualhead_model.pt --suit_weight 2.5 --out_dir ./checkpoints_9
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Canonical warped card size. Raised 250x350 -> 1000x1400 for the 384x384
# model: at 250x350 the corner crop below was only 62x125 real pixels, so
# resizing it to 384x384 would have been pure upsampling -- the network would
# have trained on blur while the camera delivers genuinely sharp pixels. The
# source photos in my_deck_raw/ are ~3800x5800, so 1000x1400 is still a
# downsample of real detail, not an invention of it. Aspect is unchanged
# (0.714, a playing card's 2.5:3.5).
CARD_W, CARD_H = 1000, 1400

# THE CORNER-CROP FRACTIONS ENCODE THE HARDWARE CROP -- see corner_crops().
# The fabric window is now SQUARE (384 x 384 camera pixels, sampled 1:1), so
# these must satisfy CARD_W*frac_w == CARD_H*frac_h. Keeping the same vertical
# extent the 96x96 build had (0.357) and squaring it gives frac_w = 0.50.
CORNER_FRAC_W = 0.50
CORNER_FRAC_H = 0.357

_crop_w = int(CARD_W * CORNER_FRAC_W)
_crop_h = int(CARD_H * CORNER_FRAC_H)
assert abs(_crop_w - _crop_h) <= 2, (
    f"corner crop is {_crop_w}x{_crop_h}, not square -- the fabric samples a "
    "square 384x384 window, so training and inference would look at different "
    "shapes of card"
)
assert min(_crop_w, _crop_h) >= 384, (
    f"corner crop is {_crop_w}x{_crop_h}, smaller than the model's 384x384 "
    "input -- raise CARD_W/CARD_H rather than upsampling blur into the network"
)


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def detect_card(frame_bgr, debug=False):
    """Finds the card and returns (warped_card_bgr, reason).

    Three changes from a naive contour approach, each fixing a real failure
    seen on this project's photos:

      1. DETECTION RUNS AT REDUCED RESOLUTION. On a 3000px-wide photo a small
         blur kernel is negligible, so edges stay noisy and polygon
         approximation rarely lands on exactly 4 vertices. Downscaling first
         makes the outline smooth and the fit reliable. Corners are scaled
         back up so the warp still uses the full-resolution image.

      2. minAreaRect INSTEAD OF approxPolyDP==4. A card IS a rectangle, so
         fitting the minimum-area rotated rectangle to the largest bright
         contour always yields 4 corners -- no dependence on the polygon
         simplifier cooperating with rounded card corners.

      3. FULL-FRAME CARDS ARE VALID. Scan-app photos are cropped tight to the
         card, so the card touches every border and covers ~100% of the
         frame. That's not a failure, it's an already-cropped card: use the
         whole image."""
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

        # rectangularity: how much of the fitted rectangle the contour fills.
        # A card fills nearly all of it; an irregular blob doesn't.
        if area / (rw * rh) < 0.75:
            rejects["not_rect"] += 1
            continue

        ratio = min(rw, rh) / max(rw, rh)
        if not (0.50 <= ratio <= 0.95):   # playing card is ~0.71
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
        # already-cropped card (scan-app style): the bright region covers most
        # of the frame, so the image itself is the card
        if bright_frac > 0.80:
            card = frame_bgr
            if card.shape[1] > card.shape[0]:   # landscape -> rotate upright
                card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
            return cv2.resize(card, (CARD_W, CARD_H)), "whole_frame"
        reason = max(rejects, key=rejects.get) if any(rejects.values()) else "no_contours"
        if debug:
            print(f"    rejects: {rejects}  bright_frac={bright_frac:.2f}")
        return None, reason

    src = order_corners(best_box / scale)   # back to full-resolution coordinates

    width = np.linalg.norm(src[1] - src[0])
    height = np.linalg.norm(src[3] - src[0])
    if width > height:
        src = np.roll(src, -1, axis=0)      # card lying sideways -> rotate mapping

    dst = np.array([[0, 0], [CARD_W - 1, 0],
                    [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    return cv2.warpPerspective(frame_bgr, M, (CARD_W, CARD_H)), "ok"


def corner_crops(warped_bgr, frac_w=CORNER_FRAC_W, frac_h=CORNER_FRAC_H):
    """Returns the two rank/suit index regions of a warped card, both upright.

    THE FRACTIONS ENCODE THE HARDWARE CROP. On a 1000x1400 warped card these
    give 500 x 499 px, aspect 1.0, which is what capture_384.v samples: a
    square 384 x 384 window taken 1:1 out of the camera's 640x480 frame.
    Change one without the other and training and inference stop looking at the
    same shape of card. History: frac_h was 0.30 when the fabric cropped a
    square 240x240 and the host trimmed it; then 0.25 x 0.357 (aspect 0.50)
    for downsample_96x96.v's 96 x 192 buffer-pixel window.

    THE WINDOW IS A CAMERA-DISTANCE CHOICE, NOT A CONSTANT. 384x384 camera
    pixels cover whatever the lens covers; these fractions say how much of a
    card that is. Put a card under the camera, look at the green box on HDMI,
    and set the fractions to match what the box actually frames -- keeping them
    square.

    A playing card prints its index at the top-left AND bottom-right, the
    latter rotated 180 degrees. Rotating the bottom-right crop back gives a
    second, equally valid upright view of the same index.

    This matters twice over:
      - RESOLUTION. On a 48x48 whole card a suit pip is ~5-7px, where a heart
        and a diamond are both just "small pointed blob". Cropping to the
        index gives roughly 3x the linear resolution on the symbol for the
        same input size -- and unlike raising IMG_SIZE, it costs no extra
        conv MACs. (IMG_SIZE was raised to 96, then to 384, because the board
        still misread real cards; the extra conv+pool stages keep the
        flattened vector, and so fc_shared, exactly the size it was.)
      - ORIENTATION. The card detector fits a rectangle, so it cannot tell a
        card from the same card rotated 180 degrees. Training on both crops
        makes the model indifferent to which way up the card sits, and
        doubles the dataset for free."""
    h, w = warped_bgr.shape[:2]
    cw, ch = int(w * frac_w), int(h * frac_h)
    top_left = warped_bgr[0:ch, 0:cw]
    bottom_right = cv2.rotate(warped_bgr[h - ch:h, w - cw:w], cv2.ROTATE_180)
    return top_left, bottom_right


MODEL_DIM = 384             # card_cnn_model.IMG_SIZE, capture_384.v's window
# A real card's corner radius as a fraction of its width: ~3 mm on 63 mm. The
# scans in deck_labeled/ are cropped square, so framed_corner() cuts it back in.
CORNER_RADIUS_FRAC = 0.05


def framed_corner(card_bgr, dx=0.0, dy=0.0, scale=1.0, bg=0):
    """The top-left index window as the board's FIXED crop box frames a card
    placed by hand -- corner_crops() is only the perfect-aim case, and the
    weights trained on it alone lose half the deck 3 mm off it.

    dx, dy: the window's top-left relative to the card's corner, as a fraction
    of the card WIDTH. Negative = off the card, so bg shows along the left/top;
    positive = into the card, so the index slides toward the window's edge.
    scale: window size relative to corner_crops()'; > 1 = card farther away.
    bg: the grey level of whatever the card lies on.

    Returns the MODEL_DIM x MODEL_DIM grey image conv1 would see. Grey
    conversion and BILINEAR resize match sim_card_cnn.py's selftest, so
    dx=dy=0, scale=1 is that crop except for the rounded corner."""
    g = np.array(Image.fromarray(card_bgr[:, :, ::-1]).convert("L"))
    r = int(CORNER_RADIUS_FRAC * CARD_W)
    arc = np.zeros((r, r), np.uint8)
    cv2.circle(arc, (r, r), r, 255, -1)                  # quarter disc = card
    g[:r, :r][arc == 0] = int(bg)
    cw, ch = int(round(_crop_w * scale)), int(round(_crop_h * scale))
    x0, y0 = int(round(dx * CARD_W)), int(round(dy * CARD_W))
    pad = max(0, -x0, -y0)
    g = cv2.copyMakeBorder(g, pad, 0, pad, 0, cv2.BORDER_CONSTANT, value=int(bg))
    win = g[pad + y0:pad + y0 + ch, pad + x0:pad + x0 + cw]
    assert win.shape == (ch, cw), f"window {dx},{dy} x{scale} runs off the card's far side"
    return np.asarray(Image.fromarray(win).resize((MODEL_DIM, MODEL_DIM), Image.BILINEAR))


def jittered_corner(card_bgr, rng):
    """One training sample: the index corner as the board's camera frames a
    hand-placed card -- off-aim, at the wrong distance, on some surface, and
    dimmer, softer and noisier than a scan (the board's white is 115-167).

    Aim is asymmetric on purpose: more than ~4% INTO the card clips the index
    itself, which no training recovers; off the card only adds background.
    Half the backgrounds are near-black (the rig's mat), half any grey, so a
    different surface still reads. Rotation is left to get_transforms()."""
    bg = rng.uniform(0, 40) if rng.random() < 0.5 else rng.uniform(0, 255)
    img = framed_corner(card_bgr, dx=rng.uniform(-0.15, 0.04), dy=rng.uniform(-0.10, 0.04),
                        scale=rng.uniform(0.85, 1.35), bg=bg).astype(np.float32)
    img = img * rng.uniform(0.45, 1.0) + rng.uniform(0, 25)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.1, 2.0))
    img += rng.normal(0, rng.uniform(0, 8), img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Card scenes: the card anywhere from index close-up to whole-card-in-view.
# framed_corner() only ever frames the index corner (27-42 mm of card in the
# box), so a card held farther away than that was unfamiliar. A scene places
# the whole card on a MODEL_DIM canvas -- the green box's view -- through one
# homography (scale, tilt, keystone, position), on a background, under uneven
# light, then through a camera pass. scene_card() is deterministic from its
# parameter dict so the simulator can replay fixed scenes; sample_scene() draws
# the dict at random for the training set.
# ---------------------------------------------------------------------------

# The top-left index (rank + small pip) as fractions of the card: measured on
# the scans, e.g. 10C's "10" and club fill x 3.5-18% and y 5-32%.
INDEX_BOX = (0.02, 0.03, 0.21, 0.34)            # u0, v0, u1, v1
# card width on the canvas, px. Close = the corner fills the box, as the
# corner crops did (768 = corner_crops' scale); above ~830 the index itself no
# longer fits. Far = the whole card in the box: it is 1.4x taller than wide.
BANDS = {"close": (600, 830), "mid": (260, 600), "far": (150, 260)}


def _card_back(h=CARD_H, w=CARD_W):
    """A generic face-down back: dark blue or red lattice inside a white border."""
    back = np.full((h, w, 3), 255, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    lattice = ((xx + yy) // 40 + (xx - yy + 4000) // 40) % 2
    inner = np.where(lattice[..., None] == 1, (150, 40, 20), (110, 25, 10)).astype(np.uint8)
    m = int(0.06 * w)
    back[m:h - m, m:w - m] = inner[m:h - m, m:w - m]
    return back


_BACKS = {}


def _alpha_mask():
    """The card's silhouette with its rounded corners, 1.0 inside."""
    a = np.zeros((CARD_H, CARD_W), np.float32)
    r = int(CORNER_RADIUS_FRAC * CARD_W)
    cv2.rectangle(a, (r, 0), (CARD_W - 1 - r, CARD_H - 1), 1.0, -1)
    cv2.rectangle(a, (0, r), (CARD_W - 1, CARD_H - 1 - r), 1.0, -1)
    for cx, cy in ((r, r), (CARD_W - 1 - r, r), (r, CARD_H - 1 - r), (CARD_W - 1 - r, CARD_H - 1 - r)):
        cv2.circle(a, (cx, cy), r, 1.0, -1)
    return a


_ALPHA = None


def scene_homography(p):
    """Card pixels -> canvas pixels for scene parameters p (see sample_scene)."""
    s = p["w"] / CARD_W
    if p["band"] == "far":
        au, av = 0.5 * CARD_W, 0.5 * CARD_H                                   # anchor: card centre
    else:
        au, av = 0.5 * (INDEX_BOX[0] + INDEX_BOX[2]) * CARD_W, 0.5 * (INDEX_BOX[1] + INDEX_BOX[3]) * CARD_H
    a = np.deg2rad(p["angle"])
    c, sn = np.cos(a), np.sin(a)
    m = np.array([[s * c, -s * sn, p["cx"] - s * (c * au - sn * av)],
                  [s * sn, s * c, p["cy"] - s * (sn * au + c * av)],
                  [0, 0, 1]], np.float64)
    kx, ky = p["keystone"]
    half = MODEL_DIM / 2.0
    t = np.array([[1, 0, half], [0, 1, half], [0, 0, 1]], np.float64)
    k = np.array([[1, 0, 0], [0, 1, 0], [kx / half, ky / half, 1]], np.float64)
    return t @ k @ np.linalg.inv(t) @ m


def _project(h, pts):
    q = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ h.T
    return q[:, :2] / q[:, 2:3]


def scene_valid(p, margin=4):
    """An index must be wholly inside the canvas (either one: the card may be
    upside down), and a far card at least ~90% inside."""
    h = scene_homography(p)
    u0, v0, u1, v1 = INDEX_BOX
    boxes = [(u0, v0, u1, v1), (1 - u1, 1 - v1, 1 - u0, 1 - v0)]         # TL, and BR turned upright
    lo, hi = margin, MODEL_DIM - 1 - margin
    ok = False
    for a0, b0, a1, b1 in boxes:
        c = _project(h, np.array([[a0 * CARD_W, b0 * CARD_H], [a1 * CARD_W, b0 * CARD_H],
                                  [a0 * CARD_W, b1 * CARD_H], [a1 * CARD_W, b1 * CARD_H]]))
        ok |= bool(((c >= lo) & (c <= hi)).all())
    if ok and p["band"] == "far":
        c = _project(h, np.array([[0, 0], [CARD_W, 0], [0, CARD_H], [CARD_W, CARD_H]], np.float64))
        slack = 0.05 * p["w"]
        ok = bool(((c >= -slack) & (c <= MODEL_DIM - 1 + slack)).all())
    return ok


def _background(p):
    rng = np.random.default_rng(p["bg_seed"])
    kind, lvl = p["bg"], p["bg_level"]
    yy, xx = np.mgrid[0:MODEL_DIM, 0:MODEL_DIM].astype(np.float32)
    if kind == "wood":
        a = rng.uniform(0, np.pi)
        t = xx * np.cos(a) + yy * np.sin(a)
        grain = np.sin(t * rng.uniform(0.08, 0.25) + 4 * np.sin(t * 0.013 + rng.uniform(0, 6)))
        base = np.array([0.35, 0.55, 0.8], np.float32) * lvl                  # B, G, R: brown
        bg = base[None, None, :] * (1 + 0.18 * grain[..., None])
    else:
        tone = rng.uniform(0.85, 1.15, 3).astype(np.float32) if kind == "cloth" else np.ones(3, np.float32)
        bg = np.broadcast_to(lvl * tone, (MODEL_DIM, MODEL_DIM, 3)).astype(np.float32).copy()
        if kind == "cloth":
            tex = cv2.GaussianBlur(rng.normal(0, 22, (MODEL_DIM, MODEL_DIM)).astype(np.float32), (0, 0), 1.5)
            bg += tex[..., None]
    return bg + rng.normal(0, 2, bg.shape).astype(np.float32)


def _warp_card(card_bgr, h, extra=None):
    """card and its alpha on the canvas; extra = a card-space pre-translation"""
    global _ALPHA
    if _ALPHA is None:
        _ALPHA = _alpha_mask()
    hh = h if extra is None else h @ extra
    s = np.sqrt(abs(np.linalg.det(hh[:2, :2])))
    pre = min(1.0, 2.0 * s)                       # shrink first so a far card is not aliased
    src, alpha = card_bgr, _ALPHA
    if pre < 1.0:
        size = (max(8, int(CARD_W * pre)), max(8, int(CARD_H * pre)))
        src = cv2.resize(card_bgr, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(_ALPHA, size, interpolation=cv2.INTER_AREA)
        hh = hh @ np.diag([CARD_W / size[0], CARD_H / size[1], 1.0])
    img = cv2.warpPerspective(src, hh, (MODEL_DIM, MODEL_DIM), flags=cv2.INTER_LINEAR).astype(np.float32)
    a = cv2.warpPerspective(alpha, hh, (MODEL_DIM, MODEL_DIM), flags=cv2.INTER_LINEAR)
    return img, a[..., None]


def scene_card(card_bgr, p):
    """The MODEL_DIM x MODEL_DIM grey image conv1 would see for scene p."""
    card = cv2.rotate(card_bgr, cv2.ROTATE_180) if p["flip"] else card_bgr
    h = scene_homography(p)
    out = _background(p)

    # neighbouring cards of a grid, face down, one pitch away in card space
    if p["backs"]:
        if "b" not in _BACKS:
            _BACKS["b"] = _card_back()
        gap = 0.12 * CARD_W
        for du, dv in p["backs"]:
            shift = np.array([[1, 0, du * (CARD_W + gap)], [0, 1, dv * (CARD_H + gap)], [0, 0, 1]], np.float64)
            img, a = _warp_card(_BACKS["b"], h, shift)
            out = a * img + (1 - a) * out

    img, a = _warp_card(card, h)
    out = a * img + (1 - a) * out

    # a finger holding a long edge, level with the card's middle -- never on an index
    if p["finger"]:
        side, v, length, width, tone = p["finger"]
        u_edge, u_in = (0.0, 1.0) if side == 0 else (CARD_W, -1.0)
        base = _project(h, np.array([[u_edge, v * CARD_H], [u_edge + u_in * length * CARD_W, v * CARD_H]]))
        (x0, y0), (x1, y1) = base
        ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        w_px = width * p["w"]
        mask = np.zeros((MODEL_DIM, MODEL_DIM), np.float32)
        cv2.ellipse(mask, (int(x0), int(y0)), (int(np.hypot(x1 - x0, y1 - y0)), int(w_px / 2)), ang, -90, 90, 1.0, -1)
        cv2.ellipse(mask, (int(x0), int(y0)), (int(w_px), int(w_px / 2)), ang, 90, 270, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.5)[..., None]
        out = mask * np.array(tone, np.float32) + (1 - mask) * out

    # the light: colour, level, a gradient across the frame, vignetting, a
    # shadow, glare -- all before the fabric's grey conversion
    yy, xx = np.mgrid[0:MODEL_DIM, 0:MODEL_DIM].astype(np.float32) / (MODEL_DIM - 1)
    out = out * np.array(p["tint"], np.float32)[None, None, :] * p["gain"]
    if p["grad"]:
        g, a = p["grad"]
        t = (xx - 0.5) * np.cos(a) + (yy - 0.5) * np.sin(a) + 0.5
        out = out * (1 - g * np.clip(t, 0, 1))[..., None]
    if p["vignette"]:
        r2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
        out = out * (1 - p["vignette"] * r2 / 0.5)[..., None]
    if p["shadow"]:
        cx, cy, rx, ry, ang, depth = p["shadow"]
        m = np.zeros((MODEL_DIM, MODEL_DIM), np.float32)
        cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), ang, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 18)
        out = out * (1 - depth * m)[..., None]
    if p["glare"]:
        cx, cy, rx, ry, ang, amp = p["glare"]
        m = np.zeros((MODEL_DIM, MODEL_DIM), np.float32)
        cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), ang, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 10)
        out = out + amp * m[..., None]

    # the fabric's grey (capture_384.v: 77R + 150G + 29B >> 8), then the camera
    out = np.clip(out, 0, 255)
    grey = 0.114 * out[..., 0] + 0.587 * out[..., 1] + 0.299 * out[..., 2]
    grey = 255.0 * (grey / 255.0) ** p["gamma"]
    if p["alias"] > 1:
        small = cv2.resize(grey, None, fx=1 / p["alias"], fy=1 / p["alias"], interpolation=cv2.INTER_NEAREST)
        grey = cv2.resize(small, (MODEL_DIM, MODEL_DIM), interpolation=cv2.INTER_LINEAR)
    if p["blur"] > 0.05:
        grey = cv2.GaussianBlur(grey, (0, 0), p["blur"])
    if p["motion"]:
        n, ang = p["motion"]
        k = np.zeros((n, n), np.float32)
        k[n // 2, :] = 1.0 / n
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), ang, 1.0), (n, n))
        grey = cv2.filter2D(grey, -1, k / max(k.sum(), 1e-6))
    grey = grey + np.random.default_rng(p["bg_seed"] + 1).normal(0, p["noise"], grey.shape)
    return np.clip(np.rint(grey), 0, 255).astype(np.uint8)


def sample_scene(rng):
    """Random scene parameters: the ranges below are the training set's
    variations (distance band, position, orientation, background, light, camera)."""
    band = str(rng.choice(["close", "mid", "far"], p=[0.25, 0.30, 0.45]))
    lo, hi = BANDS[band]
    u0, v0, u1, v1 = INDEX_BOX
    must_see = (np.array([[0, 0], [CARD_W, 0], [0, CARD_H], [CARD_W, CARD_H]], np.float64) if band == "far" else
                np.array([[u0, v0], [u1, v0], [u0, v1], [u1, v1]]) * [CARD_W, CARD_H])
    while True:                     # size, tilt and keystone first, then a position where it fits
        half = MODEL_DIM / 2.0
        p = {
            "band": band, "w": rng.uniform(lo, hi), "flip": bool(rng.random() < 0.5),
            "angle": rng.uniform(-15, 15),
            "keystone": (rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)) if rng.random() < 0.5 else (0.0, 0.0),
            "cx": half, "cy": half,
        }
        c = _project(scene_homography(p), must_see)
        edge = -0.05 * p["w"] if band == "far" else 4.0     # far: up to 5% may spill; an index: 4 px inside
        rx = (edge - c[:, 0].min(), MODEL_DIM - 1 - edge - c[:, 0].max())
        ry = (edge - c[:, 1].min(), MODEL_DIM - 1 - edge - c[:, 1].max())
        if rx[0] > rx[1] or ry[0] > ry[1]:
            continue                # too big at this tilt: draw again
        p["cx"], p["cy"] = half + rng.uniform(*rx), half + rng.uniform(*ry)
        if scene_valid(p):
            break
    bg = str(rng.choice(["black", "dark", "light", "wood", "cloth"], p=[0.35, 0.15, 0.15, 0.15, 0.20]))
    p["bg"] = bg
    p["bg_level"] = {"black": rng.uniform(5, 40), "dark": rng.uniform(40, 90), "light": rng.uniform(170, 240),
                     "wood": rng.uniform(90, 200), "cloth": rng.uniform(30, 220)}[bg]
    p["bg_seed"] = int(rng.integers(1 << 30))
    near = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    p["backs"] = ([near[i] for i in rng.choice(4, size=int(rng.integers(1, 4)), replace=False)]
                  if band != "close" and rng.random() < 0.20 else [])
    p["finger"] = ((int(rng.integers(2)), rng.uniform(0.45, 0.55), rng.uniform(0.10, 0.25), rng.uniform(0.10, 0.16),
                    tuple(rng.uniform(0.8, 1.1) * np.array([150, 170, 205])) if rng.random() < 0.8 else (40, 40, 45))
                   if rng.random() < 0.15 else None)
    level = rng.random()
    p["gain"] = rng.uniform(0.3, 0.5) if level < 0.2 else rng.uniform(0.9, 1.3) if level > 0.8 else rng.uniform(0.5, 0.9)
    p["tint"] = (1 + rng.uniform(-0.15, 0.15), 1.0, 1 + rng.uniform(-0.15, 0.15))
    p["gamma"] = rng.uniform(0.7, 1.5)
    p["grad"] = (rng.uniform(0, 0.6), rng.uniform(0, 2 * np.pi)) if rng.random() < 0.5 else None
    p["vignette"] = rng.uniform(0, 0.4) if rng.random() < 0.5 else 0.0
    p["shadow"] = ((rng.uniform(0, MODEL_DIM), rng.uniform(0, MODEL_DIM), rng.uniform(60, 180), rng.uniform(30, 90),
                    rng.uniform(0, 180), rng.uniform(0.2, 0.5)) if rng.random() < 0.20 else None)
    p["glare"] = ((rng.uniform(0, MODEL_DIM), rng.uniform(0, MODEL_DIM), rng.uniform(20, 80), rng.uniform(6, 25),
                   rng.uniform(0, 180), rng.uniform(60, 160)) if rng.random() < 0.15 else None)
    p["blur"] = rng.uniform(0, 3)
    p["motion"] = (int(rng.integers(3, 10)), rng.uniform(0, 180)) if rng.random() < 0.15 else None
    p["noise"] = rng.uniform(2, 10)
    p["alias"] = rng.uniform(1.5, 3) if rng.random() < 0.30 else 1.0
    return p


def load_bgr(path: Path):
    """Loads via PIL (so HEIC works) then converts to OpenCV BGR."""
    pil = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                         help="class-labeled folder of ORIGINAL photos (augmented copies are skipped)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--include_augmented", action="store_true",
                         help="also warp augmented copies (normally you warp originals, then augment after)")
    parser.add_argument("--debug", action="store_true",
                         help="print per-image rejection details when detection fails")
    parser.add_argument("--corner_crop", action="store_true",
                         help="save the rank/suit index corners instead of the whole card. Emits two "
                              "images per photo (top-left, and bottom-right rotated upright), giving "
                              "~3x the linear resolution on the suit symbol and orientation "
                              "robustness. Use this if Heart/Diamond confusion persists.")
    parser.add_argument("--scenes", type=int, default=0, metavar="N",
                         help="write N random card scenes per photo (see sample_scene): the card from "
                              "index close-up to whole card in view, any position/tilt, on varied "
                              "backgrounds and light -- instead of warped cards or corner crops")
    parser.add_argument("--jitter", type=int, default=0, metavar="N",
                         help="with --corner_crop: instead of the one aligned crop per corner, write N "
                              "off-aim / off-distance / camera-degraded ones (see jittered_corner), "
                              "so the model tolerates a hand-placed card under the fixed crop box")
    args = parser.parse_args()
    rng = np.random.default_rng(42)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif"}
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    total_ok, total_fail, total_whole = 0, 0, 0
    fail_reasons = {}
    thin_classes = []

    for class_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        files = [p for p in class_dir.iterdir()
                 if p.suffix.lower() in exts
                 and (args.include_augmented or "_aug" not in p.stem)]
        if not files:
            continue

        dest = out_dir / class_dir.name
        dest.mkdir(parents=True, exist_ok=True)

        n_ok = 0
        for path in files:
            try:
                frame = load_bgr(path)
            except Exception:
                fail_reasons["load_error"] = fail_reasons.get("load_error", 0) + 1
                total_fail += 1
                continue

            if args.debug:
                print(f"  {class_dir.name}/{path.name}")
            warped, reason = detect_card(frame, debug=args.debug)
            if warped is None:
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                total_fail += 1
                continue

            if args.scenes:
                for k in range(args.scenes):
                    cv2.imwrite(str(dest / f"{path.stem}_s{k}.jpg"),
                                scene_card(warped, sample_scene(rng)), [cv2.IMWRITE_JPEG_QUALITY, 92])
                n_ok += args.scenes
                total_ok += args.scenes
            elif args.corner_crop and args.jitter:
                # bottom-right index = top-left of the card turned 180, as in corner_crops()
                for tag, card in (("tl", warped), ("br", cv2.rotate(warped, cv2.ROTATE_180))):
                    for k in range(args.jitter):
                        cv2.imwrite(str(dest / f"{path.stem}_{tag}_j{k}.jpg"),
                                    jittered_corner(card, rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
                n_ok += 2 * args.jitter
                total_ok += 2 * args.jitter
            elif args.corner_crop:
                tl, br = corner_crops(warped)
                cv2.imwrite(str(dest / f"{path.stem}_tl.jpg"), tl, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(dest / f"{path.stem}_br.jpg"), br, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n_ok += 2
                total_ok += 2
            else:
                cv2.imwrite(str(dest / f"{path.stem}.jpg"), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n_ok += 1
                total_ok += 1
            if reason == "whole_frame":
                total_whole += 1

        print(f"{class_dir.name:<8} {n_ok}/{len(files)} detected")
        if n_ok < 2:
            thin_classes.append(f"{class_dir.name} ({n_ok})")

    print(f"\ntotal: {total_ok} warped ({total_whole} were already-cropped full-frame cards), "
          f"{total_fail} failed")
    if fail_reasons:
        print("failure reasons: " + "  ".join(f"{k}:{v}" for k, v in sorted(fail_reasons.items())))
        print("  small     = no bright region large enough to be a card (dim photo? card too far?)")
        print("  not_rect  = the bright region isn't rectangular enough (shadows merging with the card?)")
        print("  aspect    = found a rectangle, but not card-shaped")
        print("  dark      = found a rectangle, but too dark to be a card face")
        print("  re-run with --debug to see the per-image reject counts and bright_frac")
    if thin_classes:
        print(f"\nWARNING -- classes with fewer than 2 detected cards (too thin to train/val split "
              f"reliably even after augmentation):\n  {', '.join(thin_classes)}")

    print(f"\nwritten to {out_dir}/ -- next: augment, split, then fine-tune from checkpoints_8")


if __name__ == "__main__":
    main()