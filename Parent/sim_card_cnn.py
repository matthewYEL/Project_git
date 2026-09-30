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
  python sim_card_cnn.py --frames dumps/deck.log   # every SW1 dump in a PuTTY log -> PNG
  python sim_card_cnn.py --scan dumps/deck.log     # replay every grid scan in it

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

def _pgm_blocks(path):
    """Every P2 block in a file: a bare .pgm, one console paste, or a whole
    PuTTY session log holding many SW1 dumps. Yields (k, w, h, maxval, words),
    k counting every P2 header from 1 so it follows the shot order even when a
    truncated dump (PuTTY closed, a key typed mid-dump) is skipped."""
    tok = open(path, encoding='latin-1').read().split()    # latin-1: line noise never fails
    k = 0
    for i, t in enumerate(tok):
        if t != 'P2':
            continue
        k += 1
        try:
            w, h, maxval = int(tok[i + 1]), int(tok[i + 2]), int(tok[i + 3])
        except (IndexError, ValueError):
            print(f'{path}: dump {k}: unreadable P2 header, skipped', file=sys.stderr)
            continue
        px = tok[i + 4:i + 4 + w * h]
        n = next((j for j, v in enumerate(px) if not v.isdigit()), len(px))
        if n < w * h:
            print(f'{path}: dump {k}: expected {w * h} pixels, found {n} -- truncated, skipped',
                  file=sys.stderr)
            continue
        yield k, w, h, maxval, np.array(px, dtype=np.int64).reshape(h, w)


def _first_block(path):
    blk = next(_pgm_blocks(path), None)
    if blk is None:
        sys.exit(f'{path}: no complete "P2" block -- paste the block from the console')
    return blk


def load_capture(path):
    """Accept a bare P2 .pgm or a raw paste of the Arm DS console around it.

    Returns (pixels, zoom) where zoom is (x, y, w, h) if the paste carried
    atlas_main.c's "zoom window:" line, else None.
    """
    text = open(path, encoding='latin-1').read()
    _, w, h, maxval, px = _first_block(path)
    if maxval == 1023:
        # a 29 Sep whole-frame dump: the raw words, gray*4 with the red bit in
        # bit 0 -- the grey is what every caller here wants (load_frame keeps red)
        px = px >> 2

    zoom = None
    if 'zoom window:' in text:
        f = text.split('zoom window:', 1)[1].split()[:4]
        try:
            zoom = tuple(int(t.split('=')[1]) for t in f)      # x= y= w= h=
        except (IndexError, ValueError):
            print(f'{path}: "zoom window:" line unreadable, ignoring', file=sys.stderr)
    return px, zoom


def load_frame(path):
    """A board whole-frame dump (SW1, maxval 1023) -> (grey, red mask), both
    FRAME_H x FRAME_W, exactly what the board's locate and window mapper read.
    The first dump in the file; scan() and frames() walk all of them."""
    _, w, h, maxval, words = _first_block(path)
    if maxval != 1023:
        sys.exit(f'{path}: maxval {maxval} -- a --scan needs the raw 1023-scale frame dump of the 29 Sep app')
    return words >> 2, (words & 1).astype(bool)


# app_rtos.c ORIENTS[]: 0 upright, 1 camera turned clockwise, 2 anticlockwise, 3 180
SCAN_ORIENTS = [(False, False, False), (True, True, False), (True, False, True), (False, True, True)]


def scan(path, orient_idx=0):
    """Replay every board grid scan in a file -- one SW1 whole-frame dump or a
    PuTTY log of many: each frame through locate + the window mapper + the
    bit-exact network with the board's vote (read_cards, prvReadCard), printed
    like the board's UART lines so the two can be compared card for card."""
    import locate
    orient = SCAN_ORIENTS[orient_idx]
    done = 0
    for k, w, h, maxval, words in _pgm_blocks(path):
        if maxval != 1023:
            print(f'{path} dump {k}: maxval {maxval} -- not a 29 Sep whole-frame dump, skipped')
            continue
        if done:
            print()
        done += 1
        gray, red = words >> 2, (words & 1).astype(bool)
        info = {}
        cards = locate.locate(gray.astype(np.uint8), orient=orient, info=info)
        nr = max((c.row for c in cards), default=0)
        nc = max((c.col for c in cards), default=0)
        print(f'{path} dump {k}: {len(cards)} card(s) in {nr} row(s) x {nc} column(s)'
              f'  (threshold {info["threshold"]})')
        reads = read_cards(gray, red, orient)
        got = []
        for card, (r, s, score, j), rf, win, is_red, nread, agree, _single in reads:
            name = 'JOKER' if j else f'{r} of {s}'
            print(f'  ({card.row},{card.col})  {name:<18} logit {score:6d}  {"red" if is_red else "black":<5}'
                  f'  [{nread} reads, {agree} agree]   box x{card.x0}-{card.x1} y{card.y0}-{card.y1}')
            got.append((card.row, card.col, r, s, j))
        print('  pairs:')
        for line in pair_lines(got) or ['none']:
            print('    ' + line)
    if not done:
        sys.exit(f'{path}: no whole-frame dump (P2, maxval 1023) found')


def frames(path):
    """--frames: every frame dump in a file (a paste or a whole PuTTY log) ->
    <file>_<k>.png beside it, 2x nearest, the red-test bits tinted red, plus one
    exposure line each. Mat and card levels are split the way locate splits
    them (Otsu): the card whites want ~170-230 with little at 255."""
    import locate
    try:
        from PIL import Image
    except ImportError:
        sys.exit('--frames needs Pillow: python -m pip install pillow')
    stem, n = os.path.splitext(path)[0], 0
    for k, w, h, maxval, words in _pgm_blocks(path):
        n += 1
        gray = words >> 2 if maxval == 1023 else words * 255 // maxval
        rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)
        if maxval == 1023:
            red = (words & 1).astype(bool)
            g = gray[red]
            rgb[red] = np.stack([(g + 255) // 2, g // 2, g // 2], axis=1)
        out = f'{stem}_{k}.png'
        Image.fromarray(rgb).resize((2 * w, 2 * h), Image.NEAREST).save(out)
        t = locate.otsu(np.bincount(gray.ravel(), minlength=256))
        mat, card = gray[gray <= t], gray[gray > t]
        print(f'dump {k}: {w}x{h}  grey {gray.min()}-{gray.max()}'
              f'  mat {int(np.median(mat)) if mat.size else "-"}'
              f'  cards {int(np.median(card)) if card.size else "-"} (split {t})'
              f'  at 255: {100 * np.mean(gray >= 255):.1f} %  -> {out}')
    if not n:
        sys.exit(f'{path}: no complete "P2" block found')


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


# ---------------- whole-grid scan (M2 demo): frame -> cards -> windows ----------------

def fabric_luma(bgr):
    """capture_384.v's grey: (77R + 150G + 29B) >> 8, from an 8-bit BGR image."""
    b, g, r = (bgr[:, :, i].astype(np.int64) for i in range(3))
    return (77 * r + 150 * g + 29 * b) >> 8


def fabric_red(bgr):
    """capture_384.v's is_red: R > 64, R > 1.5 G, R > 1.5 B (integer halves)."""
    b, g, r = (bgr[:, :, i].astype(np.int64) for i in range(3))
    return (r > 64) & (r > g + (g >> 1)) & (r > b + (b >> 1))


def window_map(frame, x0, y0, step, flags=0, clip=None):
    """conv1's window mapper (conv_layer.v WIN_MAP): the DIM x DIM virtual input
    read out of the stored grey frame.

    Virtual pixel (iy, ix): u = DIM-1-ix if flip-x else ix, v likewise from iy
    with flip-y; su = (u*STEP)>>8, sv = (v*STEP)>>8 (STEP is Q8.8); the frame
    pixel is (col, row) = (X0+sv, Y0+su) with transpose, else (X0+su, Y0+sv).
    Taps outside the clip rectangle (inclusive) or the frame read 0 -- the zero
    conv1's padding reads. flags: 1 transpose, 2 flip x, 4 flip y."""
    fh, fw = frame.shape
    i = np.arange(DIM)
    su = (((DIM - 1 - i) if flags & 2 else i) * step) >> 8      # from ix
    sv = (((DIM - 1 - i) if flags & 4 else i) * step) >> 8      # from iy
    if flags & 1:
        col = np.broadcast_to(x0 + sv[:, None], (DIM, DIM))
        row = np.broadcast_to(y0 + su[None, :], (DIM, DIM))
    else:
        col = np.broadcast_to(x0 + su[None, :], (DIM, DIM))
        row = np.broadcast_to(y0 + sv[:, None], (DIM, DIM))
    cx0, cy0, cx1, cy1 = clip if clip is not None else (0, 0, fw - 1, fh - 1)
    ok = ((col >= max(cx0, 0)) & (col <= min(cx1, fw - 1)) &
          (row >= max(cy0, 0)) & (row <= min(cy1, fh - 1)))
    out = np.zeros((DIM, DIM), dtype=np.int64)
    out[ok] = frame[row[ok], col[ok]]
    return out


# app_rtos.c VARIANTS, the same table: the vote's windows as (dx, dy, zoom per
# mille, FLAGS toggle). The first SCAN_READS are read for every card; a card
# whose identity turns up twice in the grid is read with all of them.
VARIANTS = [(0, 0, 1000, 0),        # as is
            (0, 0, 1000, 6),        # turned 180 deg (flip x and y)
            (3, 0, 1000, 0),        # shifted x + 3 px
            (0, -3, 1000, 6),       # shifted y - 3 px, turned
            (0, 0, 1050, 0),        # zoomed out 5 %
            (-3, 0, 1000, 6),       # duplicates only: shifted x - 3 px, turned
            (0, 0, 950, 0)]         # duplicates only: zoomed in 5 %
SCAN_READS = 5                      # app_config.h


def _ident(r):
    return 'JOKER' if r[3] else r[0] + r[1]


def _read_variants(gray, card, base, is_red, n):
    """The card through the first n VARIANTS windows, as prvReadCard reads it:
    ([(rank, suit, score, joker), ...], the as-is window)."""
    x0, y0, step, flags, clip = base
    reads, first = [], None
    for dx, dy, zoom, flip in VARIANTS[:n]:
        vx, vy, vs = x0, y0, step
        if zoom != 1000:                      # rescale, centred on the card again
            vs = min(511, max(1, (step * zoom + 500) // 1000))
            span = ((DIM - 1) * vs) >> 8
            vx = (card.x0 + card.x1 - span + 1) // 2
            vy = (card.y0 + card.y1 - span + 1) // 2
        win = window_map(gray, vx + dx, vy + dy, vs, flags ^ flip, clip)
        first = win if first is None else first
        reads.append(infer(to_q(win), is_red))
    return reads, first


def _vote(reads):
    """prvReadCard's vote: the most frequent identity, ties to the higher summed
    rank logit, then the earlier one. Returns (that identity's highest-logit
    read -- the earliest on a tie -- and how many reads agreed)."""
    best, best_n, best_s = None, 0, 0
    for i in dict.fromkeys(_ident(r) for r in reads):      # in order of first appearance
        members = [r for r in reads if _ident(r) == i]
        n, s = len(members), sum(r[2] for r in members)
        if n > best_n or (n == best_n and s > best_s):
            best, best_n, best_s = i, n, s
    return max((r for r in reads if _ident(r) == best), key=lambda r: r[2]), best_n


def read_cards(gray, red, orient=(False, False, False), reads=SCAN_READS):
    """The board's scan in software: locate the cards, then `reads` bit-exact
    CNN reads per card through window_map and the vote, then the duplicate
    re-read -- exactly app_rtos.c prvScan. Returns [(card, (rank, suit, score,
    joker), red_fraction, as-is window, is_red, reads taken, reads agreeing,
    the as-is read alone)] in (row, col) order; is_red is locate's integer rule
    (red px x RED_PER > index-box px), as on the board."""
    import locate
    fh, fw = gray.shape
    rows = []
    for card in locate.locate(gray.astype(np.uint8), orient=orient):
        base = locate.window_params(card, fw, fh, orient=orient)
        n, area = locate.index_red(red, card)
        is_red = n * locate.RED_PER > area
        rs, win = _read_variants(gray, card, base, is_red, reads)
        chosen, agree = _vote(rs)
        rows.append([card, chosen, n / area, win, is_red, reads, agree, rs[0], base])
    # one deck, so the same card twice is a misread: read both with every window
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a = rows[i][1]
            if a[3] or _ident(a) != _ident(rows[j][1]):
                continue
            for m in (i, j):
                if rows[m][5] >= len(VARIANTS):
                    continue
                rs, _ = _read_variants(gray, rows[m][0], rows[m][8], rows[m][4], len(VARIANTS))
                rows[m][1], rows[m][6] = _vote(rs)
                rows[m][5] = len(VARIANTS)
    return [tuple(r[:8]) for r in rows]


def pair_lines(reads):
    """'7: 7H (1,1) + 7D (2,2)'-style lines from [(row, col, rank, suit, joker)]."""
    import locate
    lines = []
    for key, members in locate.rank_groups(reads):
        cards = ' + '.join(f'{"JOKER" if key == "JOKER" else r + s[0]} ({row},{col})'
                           for row, col, r, s in members)
        tag = '' if len(members) == 2 else f'   [{len(members)} of a kind]'
        lines.append(f'{key}: {cards}{tag}')
    return lines


# cv2.rotate code and locate() orient for a photo turned by `rotate` degrees
# clockwise: the camera rotated 90 deg on the rig. Turned clockwise, the rows
# run right to left along frame x and the columns top to bottom along y.
ROTATIONS = {0: (None, (False, False, False)),
             90: ('ROTATE_90_CLOCKWISE', (True, True, False)),
             270: ('ROTATE_90_COUNTERCLOCKWISE', (True, False, True))}


def photos(json_path, outdir=None, rotate=0):
    """Score the whole-grid pipeline on labelled photos of real layouts.

    Each photo is rescaled so its cards are about 95 and 65 px wide as well as
    left native (never enlarged: that would invent detail), with smooth (area)
    and aliased (nearest) resampling -- the D8M's 4x sub-sampling sits between
    the two. For every version: are all cards found at the right (row, col)?
    Is each read right with the detected colour, and with the true one? Does the
    pair report match the truth? Phone photos are sharper than the D8M, so
    treat the numbers as optimistic. rotate=90/270 turns every photo first, so
    the cards lie sideways the way the rotated camera sees them: that exercises
    the window's transpose+flip and locate()'s orient."""
    import cv2
    import locate
    rot_code, orient = ROTATIONS[rotate]
    spec = json.load(open(json_path))
    base = os.path.join(os.path.dirname(os.path.abspath(json_path)), spec.get('dir', '.'))
    rows = {}                     # version -> counters
    reds, blacks = [], []         # index red fractions by true colour
    misses = {}
    for name, grid in spec['photos'].items():
        bgr0 = cv2.imread(os.path.join(base, name))
        if bgr0 is None:
            sys.exit(f'cannot read {os.path.join(base, name)}')
        native = int(np.median([c.w for c in locate.locate(fabric_luma(bgr0).astype(np.uint8))]))
        truth = {(r + 1, c + 1): parse_truth(code) + (code,)
                 for r, line in enumerate(grid) for c, code in enumerate(line)}
        want_pairs = pair_lines([(r, c, t[0], t[1], t[0] is None) for (r, c), t in truth.items()])
        for target in (None, 95, 65):
            s = 1.0 if target is None else target / native
            if s > 1.05:
                continue
            for interp, tag in ((cv2.INTER_AREA, 'area'), (cv2.INTER_NEAREST, 'nearest')):
                if target is None and tag == 'nearest':
                    continue
                bgr = bgr0 if target is None else cv2.resize(bgr0, None, fx=s, fy=s, interpolation=interp)
                if rot_code:
                    bgr = cv2.rotate(bgr, getattr(cv2, rot_code))
                gray, red = fabric_luma(bgr), fabric_red(bgr)
                reads = read_cards(gray, red, orient)
                ver = f'{"native" if target is None else "~%dpx" % target} {tag}'
                k = rows.setdefault(ver, dict(photos=0, grid=0, cards=0, ok=0, one=0, ok_true=0, colour=0, pairs=0))
                k['photos'] += 1
                k['grid'] += sorted((c.row, c.col) for c, *_ in reads) == sorted(truth)
                got = []
                for card, (r, su, score, j), rf, win, is_red, _nread, _agree, one in reads:
                    t = truth.get((card.row, card.col))
                    if t is None:
                        continue
                    t_rank, t_suit, t_red, code = t
                    k['cards'] += 1
                    (reds if t_red else blacks).append(rf)
                    k['colour'] += is_red == bool(t_red)
                    hit = (not j) and r == t_rank and su == t_suit
                    k['ok'] += hit
                    k['one'] += (not one[3]) and one[0] == t_rank and one[1] == t_suit
                    if not hit:
                        misses.setdefault(ver, []).append(
                            f'{name[-13:-5]} ({card.row},{card.col}) {code}->{"JOKER" if j else r + su[0]}')
                    if is_red != bool(t_red):
                        rt, st, _, jt = infer(to_q(win), t_red)
                        k['ok_true'] += (not jt) and rt == t_rank and st == t_suit
                    else:
                        k['ok_true'] += hit
                    got.append((card.row, card.col, r, su, j))
                k['pairs'] += pair_lines(got) == want_pairs
                if outdir:
                    os.makedirs(outdir, exist_ok=True)
                    from PIL import Image
                    sheet = Image.new('L', (3 * 192, 3 * 192))
                    for card, _res, _rf, win, *_ in reads:
                        if 1 <= card.row <= 3 and 1 <= card.col <= 3:
                            tile = Image.fromarray(win.astype(np.uint8)).resize((192, 192), Image.NEAREST)
                            sheet.paste(tile, ((card.col - 1) * 192, (card.row - 1) * 192))
                    sheet.save(os.path.join(outdir, f'{name[-13:-5].replace(".", "")}_{ver.replace(" ", "_").replace("~", "")}.png'))
                print(f'{name[-13:-5]}  {ver:<14} cards {len(reads)}  '
                      + '  '.join(f'({c.row},{c.col}){"J" if res[3] else res[0] + res[1][0]}'
                                  for c, res, *_ in reads), flush=True)
        print('   truth pairs:', ' | '.join(want_pairs))

    print(f'\n{"version":<16}{"grid ok":>9}{"cards":>7}{"1 read":>9}{"voted":>11}{"(true colour)":>15}'
          f'{"colour ok":>11}{"pairs ok":>10}')
    tot = dict(cards=0, one=0, ok=0)
    for ver, k in rows.items():
        n = max(k['cards'], 1)
        for key in tot:
            tot[key] += k[key]
        print(f'{ver:<16}{k["grid"]:>4}/{k["photos"]:<4}{k["cards"]:>7}{k["one"]:>9}'
              f'{k["ok"]:>5} {100 * k["ok"] / n:3.0f}%{k["ok_true"]:>9} {100 * k["ok_true"] / n:3.0f}%'
              f'{k["colour"]:>7}/{k["cards"]:<3}{k["pairs"]:>6}/{k["photos"]}')
    print(f'{"all":<16}{"":>9}{tot["cards"]:>7}{tot["one"]:>9}{tot["ok"]:>5}   '
          f'(voted: {SCAN_READS} reads per card, duplicates {len(VARIANTS)})')
    if reds and blacks:
        print(f'\nindex red fraction: red cards {min(reds):.3f}-{max(reds):.3f}, '
              f'black cards {min(blacks):.3f}-{max(blacks):.3f}  (red when > 1/{locate.RED_PER})')
    for ver, m in misses.items():
        print(f'misses, {ver}: ' + '; '.join(m))


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
    ap.add_argument('--photos', metavar='JSON',
                    help='whole-grid scan on labelled photos of real layouts (photos_3x3.json): '
                         'card finding, (row, col), reads and pairs at several card sizes')
    ap.add_argument('--outdir', metavar='DIR',
                    help='with --photos: write each version\'s 3x3 sheet of CNN windows here')
    ap.add_argument('--rotate', type=int, choices=sorted(ROTATIONS), default=0,
                    help='with --photos: turn each photo this many degrees clockwise first, as the '
                         'rotated camera sees the table')
    ap.add_argument('--scan', metavar='DUMP',
                    help='replay every board grid scan in a file of SW1 whole-frame dumps (maxval 1023): '
                         'one paste or a whole PuTTY log')
    ap.add_argument('--frames', metavar='LOG',
                    help='every frame dump in a file (a paste or a whole PuTTY log) -> <file>_<n>.png, '
                         'red-test bits tinted red, plus one exposure line each')
    ap.add_argument('--orient', type=int, default=0, choices=range(len(SCAN_ORIENTS)),
                    help='with --scan: the board\'s orientation (UART o): 0 upright, 1 cam CW, 2 cam CCW, 3 cam 180')
    a = ap.parse_args()

    if a.frames:
        frames(a.frames)
        sys.exit(0)

    if a.scan:
        scan(a.scan, a.orient)
        sys.exit(0)

    if a.photos:
        photos(a.photos, a.outdir, a.rotate)
        sys.exit(0)

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
    if p.shape == (384, 512):
        # a whole-frame dump (29 Sep bitstream): what guided mode's default
        # window reads, the old centred crop at 1:1
        p = window_map(p, 64, 0, 256)
        print('  512x384 frame dump: using the default window (frame x 64..447)')
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
