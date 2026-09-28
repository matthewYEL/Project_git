"""Bit-exact software model of the 3-head card CNN datapath (card_cnn_core.v).

  image 384x384 -> conv1 C@5x5 pad2 -> pool 4 -> conv2 16@5x5 pad2 -> pool 2
                -> conv3 16@5x5 pad2 -> pool 2 -> conv4 16@5x5 pad2 -> pool 2
                -> fc_shared 2305->64 (2304 features + colour bit)
                -> fc_rank 13 / fc_suit 4 / fc_joker 2 -> colour-constrained argmax

Four conv stages with pools (4,2,2,2) take 384x384 down to the same 12x12 the
96x96 model reached in three -- fc_shared is unchanged at 2305->64. conv1's
pool is 4 and not 2 because of the M10K budget, not preference: at pool 2 its
output would be 8 x 192 x 192 = 576 M10K blocks on a device that has 553. See
friends files/trainings/card_cnn_model.py for the full arithmetic.

In the RTL each conv writes its pooled output directly (conv_layer.v's POOL
parameter); the pool is still a separate step here because it is clearer and
the result is identical.

Reads the same weights/*.hex the RTL $readmemh's, and reproduces Q6.10 with
round-to-nearest before the shift, so what it prints is what the accelerator
computes -- no board, no camera.

Companion to sim_cnn.py, which models the OLD 28x28 single-head datapath
(conv_full.v -> maxpool.v -> fc_layer.v) and no longer matches what is loaded.

  python sim_card_cnn.py capture.txt          # paste of the Arm DS console, or a bare .pgm
  python sim_card_cnn.py capture.txt --png    # also render the capture to a .png

Both colour branches are printed; read the one matching the board's colour_hw.

C (conv1 channels) and the image depth come from weights/weights_manifest.json,
so the sim is configured by the same export that produced the weights.

The board takes a PGM dump with SW1 up; --zoom and --sweep are offline
what-if tools for choosing capture_384.v's crop -- the board no longer crops.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, 'weights')
FRAC = 10
HALF = 1 << (FRAC - 1)
DIM = 384                           # network input, capture_384.v
POOLS = (4, 2, 2, 2)                # conv1..conv4, product 32: 384 -> 12


def _manifest():
    try:
        with open(os.path.join(W, 'weights_manifest.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# conv1 width and image depth are build-time choices (the M10K budget's two
# levers), so take them from the manifest export_weights_rtl.py wrote beside
# the weights rather than from constants here that can drift. The fallbacks
# are the trainer's defaults; a mismatch with the files fails read_hex's size
# assert loudly rather than simulating the wrong network.
_MAN = _manifest()
CONV1_CH = _MAN.get('conv1_ch', 8)  # ghrd_top.v CONV1_CH
IMG_DW = _MAN.get('img_dw', 10)     # ghrd_top.v IMG_DW
# fc_shared's weights carry more fractional bits than Q6.10 (they live in DDR3
# at full 16 bits) -- card_cnn_core.v FCS_W_FRAC. Older exports: Q6.10.
FCS_W_FRAC = _MAN.get('fcs_w_frac', FRAC)


def to_q(gray):
    """8-bit gray -> the Q6.10 value conv1 reads, exactly as the fabric makes it.

    capture_384.v stores ({gray, 2'b00}) >> (10 - IMG_DW) and card_cnn_core.v
    shifts the dropped bits back in as zero. So full scale is gray*4 = 1020,
    NOT gray*1024/255 = 1024: the 96x96 build's host-side rescale is gone, and
    matching the shift rather than the ideal ratio is what keeps this model
    bit-exact with the board. (Against training's gray/255 it is a uniform
    255/256 gain -- invisible to a model trained with +/-30% brightness jitter.)
    """
    drop = 10 - IMG_DW
    return ((np.asarray(gray, dtype=np.int64) << 2) >> drop) << drop

RANK = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUIT = ["Spades", "Clubs", "Hearts", "Diamonds"]


def read_hex(name, shape):
    path = os.path.join(W, name + '.hex')
    if not os.path.exists(path):
        sys.exit(f'{path} missing.\n'
                 '  The 384x384 model has four conv stages, so conv4_w/conv4_b are new.\n'
                 '  Retrain (train_card_cnn_dualhead.py), then export with\n'
                 '  export_weights_rtl.py -- it writes Q6.10 under the RTL filenames directly.\n'
                 '  (export_weights_hex.py is the INT8 C-header path; the RTL cannot read it.)')
    v = [int(l, 16) for l in open(path) if l.strip()]
    a = np.array([x - 0x10000 if x >= 0x8000 else x for x in v], dtype=np.int64)
    assert a.size == int(np.prod(shape)), (name, a.size, shape)
    return a.reshape(shape)


c1w = read_hex('conv1_w', (CONV1_CH, 1, 5, 5));   c1b = read_hex('conv1_b', (CONV1_CH,))
c2w = read_hex('conv2_w', (16, CONV1_CH, 5, 5));  c2b = read_hex('conv2_b', (16,))
c3w = read_hex('conv3_w', (16, 16, 5, 5)); c3b = read_hex('conv3_b', (16,))
c4w = read_hex('conv4_w', (16, 16, 5, 5)); c4b = read_hex('conv4_b', (16,))
fsw = read_hex('fcs_w', (64, 2305));       fsb = read_hex('fcs_b', (64,))
frw = read_hex('fcrank_w', (13, 64));      frb = read_hex('fcrank_b', (13,))
fuw = read_hex('fcsuit_w', (4, 64));       fub = read_hex('fcsuit_b', (4,))
fjw = read_hex('fcjoker_w', (2, 64));      fjb = read_hex('fcjoker_b', (2,))


# ---------------- RTL-faithful datapath ----------------

def conv(x, w, b):
    """conv_layer.v: 5x5 pad 2, bias<<FRAC, ReLU, +half LSB, >>>FRAC, sat 32767."""
    C, H, Wd = x.shape
    xp = np.zeros((C, H + 4, Wd + 4), dtype=np.int64)
    xp[:, 2:2 + H, 2:2 + Wd] = x
    acc = np.empty((w.shape[0], H, Wd), dtype=np.int64)
    for f in range(w.shape[0]):
        a = np.full((H, Wd), int(b[f]) << FRAC, dtype=np.int64)
        for c in range(C):
            for ky in range(5):
                for kx in range(5):
                    a += xp[c, ky:ky + H, kx:kx + Wd] * int(w[f, c, ky, kx])
        acc[f] = a
    # conv_layer.v's accumulator is 40 bits (16x16 products, at most 400 taps)
    assert np.abs(acc).max() < 2 ** 39, 'acc would wrap the 40-bit sum register'
    return np.minimum((np.maximum(acc, 0) + HALF) >> FRAC, 32767)


def pool(x, k=2):
    """conv_layer.v's fused max pool: k x k stride k, channel-major."""
    C, H, Wd = x.shape
    assert H % k == 0 and Wd % k == 0, f'pool {k} does not divide {H}x{Wd}'
    return x.reshape(C, H // k, k, Wd // k, k).max(axis=(2, 4))


def fc(x, w, b, relu, wfrac=FRAC):
    """fc_layer.v: 48-bit acc, bias<<W_FRAC, optional ReLU, +half LSB, >>>W_FRAC, sat.

    wfrac is the weights' fractional bits: FRAC (10) for the heads, FCS_W_FRAC
    (14) for fc_shared, whose Q1.14 weights come from DDR3. Activations and
    biases are Q6.10 either way, and the output lands back in Q6.10."""
    acc = (b.astype(np.int64) << wfrac) + w @ x
    assert np.abs(acc).max() < 2 ** 47, 'acc would wrap the 48-bit accumulator'
    if relu:
        acc = np.maximum(acc, 0)
    return np.clip((acc + (1 << (wfrac - 1))) >> wfrac, -32768, 32767)


def infer(img, colour_red, verbose=False):
    """img: (384,384) int64 Q6.10, 0..1024. Returns (rank, suit, score, joker)."""
    a = pool(conv(img[None, :, :], c1w, c1b), POOLS[0])   # 384 -> 96
    a = pool(conv(a, c2w, c2b), POOLS[1])                 #  96 -> 48
    a = pool(conv(a, c3w, c3b), POOLS[2])                 #  48 -> 24
    a = pool(conv(a, c4w, c4b), POOLS[3])                 #  24 -> 12
    feat = np.concatenate([a.ravel(), [1024 if colour_red else 0]])
    h = fc(feat, fsw, fsb, relu=True, wfrac=FCS_W_FRAC)
    rank, suit, jok = (fc(h, frw, frb, False),
                       fc(h, fuw, fub, False),
                       fc(h, fjw, fjb, False))
    ri = int(np.argmax(rank))
    # card_cnn_core.v:188 -- the suit argmax is restricted to {H,D} when red,
    # {S,C} when black. Without it the head runs Heart->Diamond, Spade->Club.
    si = max([2, 3] if colour_red else [0, 1], key=lambda i: suit[i])
    if verbose:
        print('  rank :', ' '.join(f'{RANK[i]}={rank[i]}' for i in range(13)))
        print('  suit :', ' '.join(f'{SUIT[i]}={suit[i]}' for i in range(4)))
        print('  joker: not=%d joker=%d' % (jok[0], jok[1]))
    return RANK[ri], SUIT[si], int(rank[ri]), bool(jok[1] > jok[0])


# ---------------- input ----------------

def load_capture(path):
    """Accept a bare P2 .pgm or a raw paste of the Arm DS console around it.

    Returns (pixels, zoom) where zoom is (x, y, w, h) if the paste carried
    atlas_main.c's "zoom window:" line, else None.
    """
    text = open(path).read()
    tok = text.split()
    try:
        i = tok.index('P2')
    except ValueError:
        sys.exit(f'{path}: no "P2" found -- paste the block from the console')
    w, h, _maxval = int(tok[i + 1]), int(tok[i + 2]), tok[i + 3]
    px = tok[i + 4:i + 4 + w * h]
    if len(px) < w * h:
        sys.exit(f'{path}: expected {w * h} pixels, found {len(px)} -- paste is truncated')

    zoom = None
    if 'zoom window:' in text:
        f = text.split('zoom window:', 1)[1].split()[:4]
        try:
            zoom = tuple(int(t.split('=')[1]) for t in f)      # x= y= w= h=
        except (IndexError, ValueError):
            print(f'{path}: "zoom window:" line unreadable, ignoring', file=sys.stderr)
    return np.array(px, dtype=np.int64).reshape(h, w), zoom


def apply_zoom(p, zoom):
    """Nearest-neighbour crop+rescale of a capture to DIM x DIM.

    The host no longer crops anything -- capture_384.v samples its 384x384
    window in fabric and writes the image RAM directly, so there is no
    ZOOM_* to set. This stays as an OFFLINE tool: given a capture, it answers
    "what would the network have read had the fabric window been elsewhere",
    which is how you choose capture_384.v's CROP_* constants before paying for
    a Quartus compile. See sweep().
    """
    x, y, w, h = zoom
    if not (0 <= x and 0 <= y and w >= 1 and h >= 1
            and x + w <= p.shape[1] and y + h <= p.shape[0]):
        sys.exit(f'zoom window {zoom} falls outside the {p.shape[1]}x{p.shape[0]} capture')
    sy = y + np.arange(DIM) * h // DIM
    sx = x + np.arange(DIM) * w // DIM
    return p[sy][:, sx]


SUIT_LETTER = {'S': 'Spades', 'C': 'Clubs', 'H': 'Hearts', 'D': 'Diamonds'}


def parse_truth(label):
    """'8H' / '10S' / 'QD' / 'JOKER' -> (rank, suit, colour_red)."""
    up = label.strip().upper()
    if up == 'JOKER':
        return None, None, None
    rank, suit = up[:-1], up[-1:]
    if rank not in RANK or suit not in SUIT_LETTER:
        sys.exit(f'bad truth label {label!r} -- want rank 2-10/J/Q/K/A plus S, C, H or D')
    return rank, SUIT_LETTER[suit], suit in 'HD'


def sweep(caps, truths):
    """Search crop windows for the one that reads the most captures correctly.

    caps: list of (name, pixels). truths: list of (rank, suit, colour_red).
    Windows are scored on exact rank+suit matches, tie-broken by mean winning
    logit. Colour comes from the truth label, so this isolates framing from the
    red/black detector.

    The answer is now a set of capture_384.v CROP_* constants, not host
    #defines -- changing it costs a Quartus compile, so it is worth sweeping
    several captures before committing.
    """
    step = DIM // 24                       # 16 at DIM 384
    grid = [(x, y, w, h)
            for w in (DIM // 4, DIM // 3, DIM // 2, DIM * 2 // 3, DIM)
            for h in (DIM // 2, DIM * 2 // 3, DIM * 3 // 4, DIM)
            for x in range(0, DIM * 2 // 3, step)
            for y in range(0, DIM * 2 // 3, step)
            if x + w <= DIM and y + h <= DIM]
    print(f'sweeping {len(grid)} windows over {len(caps)} capture(s)...\n')

    scored = []
    for win in grid:
        hits, logits = 0, []
        for (_, p), (tr, ts, tc) in zip(caps, truths):
            q = to_q(apply_zoom(p, win))
            r, s, sc, j = infer(q, tc, verbose=False)
            if not j and r == tr and s == ts:
                hits += 1
            logits.append(sc)
        scored.append((hits, sum(logits) / len(logits), win))
    scored.sort(key=lambda t: (-t[0], -t[1]))

    print(f'{"hits":>5}  {"mean logit":>10}  window')
    for hits, ml, (x, y, w, h) in scored[:8]:
        print(f'{hits:>3}/{len(caps)}  {ml:>10.0f}  x={x} y={y} w={w} h={h}')

    best = scored[0]
    if best[0] == 0:
        print('\nNo window reads any capture correctly. Either the captures are '
              'clipped or mis-aimed, or the crop is not what is wrong.')
        return
    bx, by, bw, bh = best[2]
    print('\n  capture_384.v, relative to the current window:')
    print('    localparam CROP_X0 = CROP_X0 + %d, spanning %d columns' % (bx, bw))
    print('    localparam CROP_Y0 = CROP_Y0 + %d, spanning %d rows' % (by, bh))
    print('  (offsets are in captured pixels, which are camera pixels 1:1)')
    if len(caps) < 3:
        print('\nPROVISIONAL -- tuned on %d capture(s). A window fitted to one card '
              'means nothing;\nre-run with at least 3 different cards before '
              'trusting it.' % len(caps))


def write_png(p, path):
    try:
        from PIL import Image
    except ImportError:
        sys.exit('--png needs Pillow: python -m pip install pillow')
    Image.fromarray(p.astype(np.uint8)).resize((480, 480), Image.NEAREST).save(path)
    return path


# ---------------- regression check ----------------

MIN_CORNER_ACC = 52      # of 54; measured 54/54 when this was written


def truth_from_folder(name):
    """'8H' / '10S' / 'JOKER' -> (rank, suit, colour_red); None if the folder is
    not a full card code (e.g. my_deck's rank-only '10', 'A', 'Joker')."""
    up = name.strip().upper()
    if up == 'JOKER':
        return None, None, None
    rank, suit = up[:-1], up[-1:]
    if rank in RANK and suit in SUIT_LETTER:
        return rank, SUIT_LETTER[suit], suit in 'HD'
    return None


def selftest(root=None):
    """Feed a folder of card photos through the friend's own crop and check the
    weights still read it.

    This is the one check that exercises weights, fixed-point datapath and
    composition together. It reuses the friend's detector and crop rather than
    reimplementing them, so if their pipeline changes this follows.

    Two folder layouts:
      my_deck/<rank>/       (the default)  rank-only folders -> scores RANK, and
                            accepts whichever colour flag wins, since the folder
                            does not say.
      <dir>/<rank><suit>/   e.g. image_data/deck_labeled -> scores RANK AND SUIT
                            with the TRUE colour flag. This is the one that gates
                            the historical Heart->Diamond failure through the
                            fixed-point path; the rank-only mode cannot.

    A pass does NOT mean the board works: both folders are the deck the model
    was trained on (my_deck is byte-identical to my_deck_raw), so this measures
    that the export and the fixed-point datapath reproduce the float model, not
    generalisation, and the board's framing is a separate problem. A FAILURE
    means the weights or this simulator stopped agreeing with the pipeline they
    came from.
    """
    import glob
    sys.path.insert(0, os.path.join(HERE, 'friends files', 'trainings'))
    try:
        import cv2                                     # noqa: F401  (used via bw)
        import build_warped_dataset as bw
        from PIL import Image
    except ImportError as e:
        sys.exit(f'--selftest needs opencv, pillow and friends files/trainings/: {e}')

    root = root or os.path.join(HERE, 'my_deck')
    files = sorted(glob.glob(os.path.join(root, '*', '*')))
    if not files:
        sys.exit(f'--selftest needs {root}/<class>/*.png')

    ok = n = 0
    misses = []
    full_codes = rank_only = 0          # how the non-joker folders were labelled
    for f in files:
        warped, _ = bw.detect_card(bw.load_bgr(f))
        if warped is None:
            continue
        n += 1
        label = os.path.basename(os.path.dirname(f))
        crop, _br = bw.corner_crops(warped)             # top-left index corner
        pil = Image.fromarray(crop[:, :, ::-1]).convert('L').resize((DIM, DIM), Image.BILINEAR)
        q = to_q(np.asarray(pil))
        truth = truth_from_folder(label)
        if truth is not None:
            # full card code: one inference with the true colour, rank AND suit must match
            t_rank, t_suit, t_red = truth
            if t_rank is None:                          # JOKER
                got = [infer(q, c, verbose=False) for c in (True, False)]
                hit = any(g[3] for g in got)
            else:
                full_codes += 1
                got = [infer(q, t_red, verbose=False)]
                hit = got[0][0] == t_rank and got[0][1] == t_suit and not got[0][3]
        else:
            # rank-only folder: accept whichever colour flag wins
            got = [infer(q, c, verbose=False) for c in (True, False)]
            if label.upper() == 'JOKER':
                # the rank head cannot express Joker -- that is the joker head's job
                hit = any(g[3] for g in got)
            else:
                rank_only += 1
                hit = any(g[0] == label and not g[3] for g in got)
        if hit:
            ok += 1
        else:
            g = got[0]
            misses.append(f'{label}->{"JOKER" if g[3] else g[0] + g[1][0]}')

    floor = MIN_CORNER_ACC if n == 54 else int(round(n * MIN_CORNER_ACC / 54))
    what = ('rank+suit' if rank_only == 0 else
            'rank' if full_codes == 0 else 'mixed rank/rank+suit')
    print(f'corner-crop regression: {ok}/{n} {what} correct '
          f'({100 * ok / n:.0f}%), floor is {floor}   [{root}]')
    if misses:
        print('  misses:', ' '.join(misses))
    if ok < floor:
        sys.exit(f'FAIL -- expected at least {floor}')
    print('PASS')


# (label, dx, dy, scale, bg) for build_warped_dataset.framed_corner(): offsets
# are fractions of the card width (5% = 3.2 mm), negative = off the card.
FRAMING_ROWS = [
    ('aligned',                 0.00,  0.00, 1.00,   0),
    ('5% into card, right',     0.05,  0.00, 1.00,   0),
    ('10% into card, right',    0.10,  0.00, 1.00,   0),
    ('5% into card, down',      0.00,  0.05, 1.00,   0),
    ('5% off card, left',      -0.05,  0.00, 1.00,   0),
    ('10% off card, left',     -0.10,  0.00, 1.00,   0),
    ('5% off card, up',         0.00, -0.05, 1.00,   0),
    ('10% off card, up',        0.00, -0.10, 1.00,   0),
    ('card 10% closer',         0.00,  0.00, 0.90,   0),
    ('card 20% farther',        0.00,  0.00, 1.20,   0),
    ('card 30% farther',        0.00,  0.00, 1.30,   0),
    ('5% off left,  light bg', -0.05,  0.00, 1.00, 200),
    ('5% off up,    light bg',  0.00, -0.05, 1.00, 200),
    ('20% farther,  light bg',  0.00,  0.00, 1.20, 200),
]


def _scene(**kw):
    """A fixed build_warped_dataset.scene_card() scene: a whole card centred on
    a black mat in fair light, then whatever the row changes."""
    p = dict(band='far', w=220.0, flip=False, angle=0.0, keystone=(0.0, 0.0), cx=192.0, cy=192.0,
             bg='black', bg_level=20.0, bg_seed=1, backs=[], finger=None, gain=0.8,
             tint=(1.0, 1.0, 1.0), gamma=1.0, grad=None, vignette=0.0, shadow=None, glare=None,
             blur=0.8, motion=None, noise=4.0, alias=1.0)
    p.update(kw)
    return p


# The card farther away than a corner crop (w = card width on the 384 canvas;
# 768 is corner_crops' scale), tilted, upside down, in poor light.
SCENE_ROWS = [
    ('scene: whole card, w150',       _scene(w=150.0)),
    ('scene: whole card, w220',       _scene()),
    ('scene: whole card, w255',       _scene(w=255.0)),
    ('scene: half card, w400',        _scene(band='mid', w=400.0, cx=110.0, cy=110.0)),
    ('scene: corner, w768',           _scene(band='close', w=768.0, cx=110.0, cy=200.0)),
    ('scene: whole card, tilt 15',    _scene(w=200.0, angle=15.0)),
    ('scene: whole card, upside down', _scene(flip=True)),
    ('scene: whole card, dim 0.35',   _scene(gain=0.35)),
    ('scene: whole card, shadow',     _scene(shadow=(150.0, 150.0, 140.0, 70.0, 30.0, 0.45))),
    ('scene: whole card, glare',      _scene(glare=(200.0, 150.0, 50.0, 15.0, 20.0, 120.0))),
    ('scene: grey table + backs',     _scene(w=180.0, bg='dark', bg_level=70.0, backs=[(1, 0), (-1, 0)])),
]


def framing(root):
    """How far off-aim the weights still read a card. The board's crop box is
    fixed and cards are placed by hand, so this -- not the aligned selftest --
    is what predicts the board.

    Every card in root (<rank><suit>/ folders, e.g. deck_labeled) goes through
    build_warped_dataset.framed_corner(), the geometry the --jitter training
    set is cut with, then this bit-exact model; rank AND suit must match with
    the true colour flag. It is the training deck, so this measures aim
    tolerance, not generalisation."""
    import glob
    sys.path.insert(0, os.path.join(HERE, 'friends files', 'trainings'))
    import build_warped_dataset as bw

    cards = []
    for f in sorted(glob.glob(os.path.join(root, '*', '*'))):
        truth = truth_from_folder(os.path.basename(os.path.dirname(f)))
        warped, _ = bw.detect_card(bw.load_bgr(f))
        if truth is not None and warped is not None:
            cards.append((truth, warped))
    print(f'{len(cards)} cards [{root}]; offsets are % of card width, 5% = 3.2 mm\n')
    def score(make):
        ok = 0
        for (t_rank, t_suit, t_red), warped in cards:
            q = to_q(make(warped))
            if t_rank is None:                          # JOKER: either colour flag
                ok += any(infer(q, c)[3] for c in (True, False))
            else:
                r = infer(q, t_red)
                ok += r[0] == t_rank and r[1] == t_suit and not r[3]
        return ok

    for name, dx, dy, scale, bg in FRAMING_ROWS:
        ok = score(lambda w: bw.framed_corner(w, dx, dy, scale, bg))
        print(f'  {name:<24} bg {bg:>3}   rank+suit {ok:2d}/{len(cards)}', flush=True)
    for name, p in SCENE_ROWS:
        ok = score(lambda w: bw.scene_card(w, p))
        print(f'  {name:<31}   rank+suit {ok:2d}/{len(cards)}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('capture', nargs='*', default=[os.path.join(HERE, 'card.pgm')],
                    help='Arm DS console paste(s), or bare P2 .pgm file(s)')
    ap.add_argument('--zoom', metavar='X,Y,W,H',
                    help='override the crop; defaults to the paste\'s "zoom window:" line')
    ap.add_argument('--png', action='store_true',
                    help='render the capture (and the zoomed upload) next to the input')
    ap.add_argument('--sweep', metavar='TRUTH',
                    help='search zoom windows; comma-separated true cards, one per '
                         'capture, e.g. 8H,QS,6D')
    ap.add_argument('--selftest', action='store_true',
                    help='check the weights against my_deck/ (rank only) through the friend\'s own crop')
    ap.add_argument('--selftest_dir', metavar='DIR',
                    help='like --selftest but on a folder of <rank><suit>/ classes, scoring rank AND '
                         'suit with the true colour, e.g. "friends files/trainings/image_data/deck_labeled"')
    ap.add_argument('--framing', metavar='DIR',
                    help='aim-tolerance sweep: the <rank><suit>/ folder\'s cards cut off-aim, '
                         'off-distance and on dark/light backgrounds (FRAMING_ROWS), rank+suit per row')
    a = ap.parse_args()

    if a.selftest or a.selftest_dir:
        selftest(a.selftest_dir)
        sys.exit(0)

    if a.framing:
        framing(a.framing)
        sys.exit(0)

    if a.sweep:
        truths = [parse_truth(t) for t in a.sweep.split(',')]
        if len(truths) != len(a.capture):
            sys.exit(f'--sweep has {len(truths)} labels but {len(a.capture)} capture(s)')
        if any(t[0] is None for t in truths):
            sys.exit('--sweep cannot score JOKER captures; drop them')
        caps = [(f, load_capture(f)[0]) for f in a.capture]
        sweep(caps, truths)
        sys.exit(0)

    if len(a.capture) > 1:
        sys.exit('pass one capture, or use --sweep to score several at once')

    path = a.capture[0]
    p, zoom = load_capture(path)
    if a.zoom:
        zoom = tuple(int(v) for v in a.zoom.split(','))
    full = (0, 0, p.shape[1], p.shape[0])

    print(f'{path}: {p.shape[1]}x{p.shape[0]}, '
          f'raw 8-bit min {p.min()} max {p.max()}')
    if zoom and tuple(zoom) != full:
        print(f'  zoom x={zoom[0]} y={zoom[1]} w={zoom[2]} h={zoom[3]}')
        p = apply_zoom(p, zoom)
    print()

    if a.png:
        stem = os.path.splitext(path)[0]
        print('  wrote', write_png(p, stem + '.png'), '\n')

    # The board dumps gray = q >> 2 of the exact values conv1 read, so to_q()
    # rebuilds them losslessly -- the 96x96 build's requantisation, where the
    # PGM carried raw*255/510 against an upload of raw*1024/510, is gone.
    q = to_q(p)

    for colour_red in (True, False):
        r, s, score, j = infer(q, colour_red, verbose=colour_red)
        print('  colour=%-5s -> %s   rank logit %d\n'
              % ('red' if colour_red else 'black',
                 'JOKER' if j else f'{r} of {s}', score))
