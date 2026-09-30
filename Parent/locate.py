"""Find the face-up cards in a grid frame and number them by (row, col).

Reference for M2/locate.c: the board runs the same steps on CPU1 over the frame
it reads back through snapshot_addr/data. Written as plain loops a C port can
follow, not as clever numpy:

  1. sample the frame every STEP px
  2. Otsu threshold on the sample histogram -- cards are near-white on a dark mat
     (the 28 Sep phone photos: mat grey 55-67, cards 170-220)
  3. 4-connected components -> bounding box and pixel count
  4. keep card-like ones:
       - not touching the sample border (desk, sockets, clothes past the mat do)
       - pieces of one card whose boxes overlap heavily are merged into one box:
         at 50-80 px card width the thin white border of court and many-pip
         cards breaks, and the card comes out as two overlapping blobs.
         (A 3x3 closing healed that too but bridged 5 mm gaps at 50 px; plain
         threshold + merge held on all 7 photos at every scale down to 50 px.)
       - card-shaped box, card-sized relative to the median card
       - not nested inside another card's box (court-card artwork fragments)
       - a blob about twice the median area is two touching cards: split it
  5. rows from the gaps between centres along one axis, columns along the other
  6. colour from the red bits in the two index corners, sampled from the card's
     own corners (its blob's extreme points) so tilt and perspective can't move
     them off the index. (30 Sep, the first D8M frame: cards ~6 deg off square
     read K-spades red and 8-diamonds black with bounding-box corners.)

Frame coordinates are the board's: (0, 0) is the top-left of the stored frame,
x to the right. `orient` maps them onto the professor's (row, col): row 1 is
the top row and column 1 the left, as the professor sees the table.
"""
import numpy as np

# Where the rank+suit index sits, as (u0, v0, u1, v1) percent of the card's short
# and long side, measured from a card corner along the card's own edges; the
# opposite corner holds the other index. Tighter than build_warped_dataset's
# INDEX_BOX (2, 3, 21, 34): anchored on the real corner it needn't allow for
# tilt, and it keeps clear of the card edge (the D8M's false-red fringe) and of
# court artwork. INDEX_NU x INDEX_NV samples per corner, all integer so
# locate.c samples the identical points.
INDEX_BOX_PCT = (4, 4, 18, 28)
INDEX_NU, INDEX_NV = 12, 20
RED_PER = 16          # red when red samples x RED_PER > samples, both corners together
                      # (6.25 %; 30 Sep, D8M + 729 photo cards: black <= 2.3 %, red >= 15 %)

STEP = 2              # sample every 2nd frame pixel (~49 k bridge reads at 576x340)
MIN_BOX = 100         # sample px; smaller boxes are specks, never cards
# Shape and size tests, in integers so locate.c matches exactly:
#   card-shaped    box w/h (or h/w) within 0.50 .. 0.90, tilt included
#   card-sized     box area 0.5 .. 1.6 x the median card box
#   two cards      box area 1.7 .. 2.4 x the median: split it
# a new row/column starts where sorted centres jump by more than half the
# median card extent along that axis
MERGE_OVERLAP = 0.25  # merge two boxes sharing > this fraction of the smaller one:
                      # pieces of one card overlap 50-65 %, tilted neighbours only in slivers

# Window around a card for the CNN (conv1's window mapper, card_cnn_core.v).
# Square side = 1.15 x the card's longer side; clip = card box grown by 4 % of
# it. All integer arithmetic (see window_params) so locate.c matches exactly.
VIRT = 384            # conv1's virtual input
FLAG_T, FLAG_FX, FLAG_FY = 1, 2, 4   # window FLAGS bits: transpose, flip x, flip y


class Card:
    __slots__ = ('x0', 'y0', 'x1', 'y1', 'row', 'col', 'corners')

    def __init__(self, x0, y0, x1, y1, corners=None):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.row = self.col = 0
        # (top-left, top-right, bottom-left, bottom-right) card corners, frame px
        self.corners = corners or ((x0, y0), (x1, y0), (x0, y1), (x1, y1))

    w = property(lambda s: s.x1 - s.x0 + 1)
    h = property(lambda s: s.y1 - s.y0 + 1)
    cx = property(lambda s: (s.x0 + s.x1) / 2.0)
    cy = property(lambda s: (s.y0 + s.y1) / 2.0)
    landscape = property(lambda s: s.w > s.h)

    def __repr__(self):
        return f'Card(({self.row},{self.col}) x{self.x0}-{self.x1} y{self.y0}-{self.y1})'


def otsu(hist):
    """Threshold maximising the between-class variance; ties keep the lowest."""
    total = int(sum(hist))
    sum_all = float(sum(i * int(hist[i]) for i in range(256)))
    w_b, sum_b, best, thr = 0, 0.0, -1.0, 0
    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * int(hist[t])
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        d = m_b - m_f
        between = float(w_b) * float(w_f) * (d * d)          # d*d, as locate.c
        if between > best:
            best, thr = between, t
    return thr


# A card's corners are its blob's extreme points: top-left minimises x + y,
# bottom-right maximises it, top-right maximises x - y, bottom-left minimises it
# (for any tilt under 45 deg). Ties go to the upper point for the top corners and
# the lower one for the bottom corners, so the pick doesn't depend on the order
# the flood fill meets pixels in, and locate.c's beyond() picks the same one.
def _ksum(p):
    return (p[0] + p[1], p[1])


def _kdiff(p):
    return (p[0] - p[1], -p[1])


def components(b):
    """4-connected components of a boolean map ->
    [(x0, y0, x1, y1, count, (tl, tr, bl, br))], corners as (x, y).

    Flood fill with an explicit stack, as locate.c does."""
    h, w = b.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for y in range(h):
        row = b[y]
        for x in np.flatnonzero(row & ~seen[y]):
            if seen[y, x]:
                continue
            x0 = x1 = x
            y0 = y1 = y
            n = 0
            tl = tr = bl = br = (int(x), int(y))
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                n += 1
                x0, x1 = min(x0, cx), max(x1, cx)
                y0, y1 = min(y0, cy), max(y1, cy)
                p = (int(cx), int(cy))
                if _ksum(p) < _ksum(tl):
                    tl = p
                if _kdiff(p) > _kdiff(tr):
                    tr = p
                if _kdiff(p) < _kdiff(bl):
                    bl = p
                if _ksum(p) > _ksum(br):
                    br = p
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and b[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append((int(x0), int(y0), int(x1), int(y1), n, (tl, tr, bl, br)))
    return out


def _merge_corners(a, b):
    """Corners of two blobs' union: the more extreme of each pair."""
    return (min(a[0], b[0], key=_ksum), max(a[1], b[1], key=_kdiff),
            min(a[2], b[2], key=_kdiff), max(a[3], b[3], key=_ksum))


def _box_corners(x0, y0, x1, y1):
    """A split half's corners: its box is all there is to go on."""
    return ((x0, y0), (x1, y0), (x0, y1), (x1, y1))


def _merge_overlapping(boxes):
    """Union boxes that overlap by more than MERGE_OVERLAP of the smaller, until
    none do. Boxes are [x0, y0, x1, y1, corners], inclusive."""
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            a = boxes[i]
            for j in range(i + 1, len(boxes)):
                b = boxes[j]
                iw = min(a[2], b[2]) - max(a[0], b[0]) + 1
                ih = min(a[3], b[3]) - max(a[1], b[1]) + 1
                if iw <= 0 or ih <= 0:
                    continue
                small = min((a[2] - a[0] + 1) * (a[3] - a[1] + 1), (b[2] - b[0] + 1) * (b[3] - b[1] + 1))
                if iw * ih > MERGE_OVERLAP * small:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]),
                                _merge_corners(a[4], b[4])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def _shaped(w, h):
    return (2 * w >= h and 10 * w <= 9 * h) or (2 * h >= w and 10 * h <= 9 * w)


def _cluster(values, gap):
    """Index of each value's group, groups split where sorted values jump by > gap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    idx = [0] * len(values)
    g = 0
    for k, i in enumerate(order):
        if k and values[i] - values[order[k - 1]] > gap:
            g += 1
        idx[i] = g
    return idx, g + 1


def locate(gray, step=STEP, orient=(False, False, False), info=None):
    """Cards in a grey frame (H x W uint8), sorted by (row, col).

    orient = (swap, flip_rows, flip_cols): swap takes rows along frame x (the
    camera rotated 90 deg); the flips number rows/columns from the other end.
    info, if a dict, receives the threshold and reject counts for debugging."""
    s = np.asarray(gray)[::step, ::step]
    sh, sw = s.shape
    thr = otsu(np.bincount(s.ravel(), minlength=256))
    b = s > thr

    cand = []
    rej_edge = rej_small = 0
    for x0, y0, x1, y1, _n, cor in components(b):
        if x0 == 0 or y0 == 0 or x1 == sw - 1 or y1 == sh - 1:
            rej_edge += 1
            continue
        if (x1 - x0 + 1) * (y1 - y0 + 1) < MIN_BOX:
            rej_small += 1
            continue
        cand.append((x0, y0, x1, y1, cor))
    n_comp = len(cand) + rej_edge + rej_small
    cand = [tuple(c) for c in _merge_overlapping(cand)]

    shaped = [c for c in cand if _shaped(c[2] - c[0] + 1, c[3] - c[1] + 1)]
    cards = []
    if shaped:
        # The size reference counts only boxes at least a quarter of the largest
        # card-shaped one: cards are the biggest card-shaped things on the mat.
        # With 1-2 cards, card-shaped slivers of other cards at the frame edge
        # otherwise dragged the median down and got the real card rejected as
        # too big (29 Sep: a lone 7D read as a sliver, AD).
        top = max((c[2] - c[0] + 1) * (c[3] - c[1] + 1) for c in shaped)
        shaped = [c for c in shaped if 4 * (c[2] - c[0] + 1) * (c[3] - c[1] + 1) >= top]
        areas = sorted((c[2] - c[0] + 1) * (c[3] - c[1] + 1) for c in shaped)
        med = areas[len(areas) // 2]
        ws = sorted(c[2] - c[0] + 1 for c in shaped)
        hs = sorted(c[3] - c[1] + 1 for c in shaped)
        mw, mh = ws[len(ws) // 2], hs[len(hs) // 2]
        for x0, y0, x1, y1, cor in cand:
            w, h = x1 - x0 + 1, y1 - y0 + 1
            a = w * h
            if 2 * a >= med and 10 * a <= 16 * med and _shaped(w, h):
                cards.append([x0, y0, x1, y1, cor])
            elif 17 * med <= 10 * a <= 24 * med:
                # two touching cards: split across the axis that is doubled
                if w * mh >= h * mw:
                    xm = x0 + w // 2
                    halves = ([x0, y0, xm - 1, y1], [xm, y0, x1, y1])
                else:
                    ym = y0 + h // 2
                    halves = ([x0, y0, x1, ym - 1], [x0, ym, x1, y1])
                cards.extend(hv + [_box_corners(*hv)] for hv in halves
                             if _shaped(hv[2] - hv[0] + 1, hv[3] - hv[1] + 1))

    # drop boxes inside another card's box (bright fragments of court artwork)
    kept = []
    for i, c in enumerate(cards):
        inside = any(j != i and o[0] <= c[0] and o[1] <= c[1] and c[2] <= o[2] and c[3] <= o[3]
                     and (o[2] - o[0]) * (o[3] - o[1]) > (c[2] - c[0]) * (c[3] - c[1])
                     for j, o in enumerate(cards))
        if not inside:
            kept.append(c)

    H, W = np.asarray(gray).shape
    e = step - 1                      # a sample stands for step x step frame px
    out = [Card(x0 * step, y0 * step, min(W - 1, x1 * step + e), min(H - 1, y1 * step + e),
                ((tl[0] * step, tl[1] * step), (min(W - 1, tr[0] * step + e), tr[1] * step),
                 (bl[0] * step, min(H - 1, bl[1] * step + e)),
                 (min(W - 1, br[0] * step + e), min(H - 1, br[1] * step + e))))
           for x0, y0, x1, y1, (tl, tr, bl, br) in kept]
    assign_grid(out, orient)
    out.sort(key=lambda c: (c.row, c.col))
    if info is not None:
        info.update(threshold=thr, components=n_comp,
                    rejected_edge=rej_edge, rejected_small=rej_small, cards=len(out))
    return out


def assign_grid(cards, orient=(False, False, False)):
    """Set .row and .col (1-based) from the gaps between card centres."""
    if not cards:
        return 0, 0
    swap, flip_r, flip_c = orient
    ra = [c.cx if swap else c.cy for c in cards]         # position along the row axis
    ca = [c.cy if swap else c.cx for c in cards]
    er = sorted(c.w if swap else c.h for c in cards)     # card extent along it
    ec = sorted(c.h if swap else c.w for c in cards)
    ri, nr = _cluster(ra, er[len(er) // 2] / 2.0)       # ROW_GAP 0.5 x the extent
    ci, nc = _cluster(ca, ec[len(ec) // 2] / 2.0)
    for c, r, k in zip(cards, ri, ci):
        c.row = (nr - r) if flip_r else r + 1
        c.col = (nc - k) if flip_c else k + 1
    return nr, nc


def turn_flags(orient):
    """FLAGS that turn a sideways card upright for this camera orientation.

    Transpose plus ONE flip is a 90 deg rotation (transpose alone would mirror
    the card). Which flip decides upright versus upside down, and the photo test
    of 29 Sep read upside-down cards worse (A-hearts as A-diamonds), so it follows
    the camera: rotated clockwise (rows numbered from the right, orient[1]) needs
    flip y, anticlockwise needs flip x."""
    return FLAG_T | (FLAG_FY if orient[1] else FLAG_FX)


def window_params(card, frame_w, frame_h, orient=(False, False, False)):
    """conv1 window registers for one card: (X0, Y0, STEP, FLAGS, CLIP).

    A square window 1.15 x the card's longer side, centred on the card, scaled
    onto the 384x384 virtual input (STEP is Q8.8 frame px per virtual px). The
    clip rectangle -- the card box grown by 4 % of the longer side -- blacks
    out neighbours, so the network sees one card on black, the way it was
    trained. A card lying sideways is turned upright by turn_flags(orient).
    Rounding is floor(x + 0.5) in integers throughout, as in locate.c."""
    long_side = max(card.w, card.h)
    step = max(1, min(511, (23 * long_side + 15) // 30))     # 1.15 * long * 256/384
    span = ((VIRT - 1) * step) >> 8                         # frame px the window covers
    x0 = (card.x0 + card.x1 - span + 1) // 2                # centre - span/2
    y0 = (card.y0 + card.y1 - span + 1) // 2
    m = max(2, (4 * long_side + 50) // 100)
    clip = (max(0, card.x0 - m), max(0, card.y0 - m),
            min(frame_w - 1, card.x1 + m), min(frame_h - 1, card.y1 + m))
    flags = turn_flags(orient) if card.landscape else 0
    return x0, y0, step, flags, clip


def index_samples(card):
    """The sample points of the card's two index corners, [[(x, y), ...] x 2].

    Upright or upside down, the indices sit on one diagonal: top-left and
    bottom-right of a portrait card, top-right and bottom-left of one lying
    sideways. Each corner's patch runs INDEX_BOX_PCT along the card's own two
    edges from that corner -- u along the short side, v along the long -- so it
    follows the card through tilt and perspective. Nearest pixel, in integers,
    exactly as locate.c's index_red()."""
    U0, V0, U1, V1 = INDEX_BOX_PCT
    nu, nv = INDEX_NU, INDEX_NV
    tl, tr, bl, br = card.corners
    # (corner, its neighbour along the short side, along the long side)
    patches = ((tr, br, tl), (bl, tl, br)) if card.landscape else ((tl, tr, bl), (br, bl, tr))
    d = 100 * (nu - 1) * (nv - 1)
    out = []
    for c, a, b in patches:
        pts = []
        for i in range(nu):
            fu = (U0 * (nu - 1) + (U1 - U0) * i) * (nv - 1)
            for j in range(nv):
                fv = (V0 * (nv - 1) + (V1 - V0) * j) * (nu - 1)
                nx = fu * (a[0] - c[0]) + fv * (b[0] - c[0])
                ny = fu * (a[1] - c[1]) + fv * (b[1] - c[1])
                pts.append((c[0] + (2 * nx + d) // (2 * d), c[1] + (2 * ny + d) // (2 * d)))
        out.append(pts)
    return out


def index_red(red, card):
    """(red samples, samples) over both index corners; the card is red when
    red samples x RED_PER > samples.

    Only the indices: court cards carry red and gold artwork whatever the suit,
    which is what fooled the old whole-window count. Points off the frame
    don't count."""
    h, w = red.shape
    n = m = 0
    for pts in index_samples(card):
        for x, y in pts:
            if 0 <= x < w and 0 <= y < h:
                m += 1
                n += int(red[y, x])
    return n, max(m, 1)


def rank_groups(reads):
    """reads: [(row, col, rank, suit, joker)] -> [(key, [(row, col, rank, suit), ...])]
    for every rank seen at least twice, in reading order. A pair is two cards of
    the same rank; two jokers pair too. Three of a kind is reported as a group of
    three rather than a pair plus a silently dropped card."""
    groups = {}
    for r, c, rank, suit, joker in sorted(reads, key=lambda t: (t[0], t[1])):
        key = 'JOKER' if joker else rank
        groups.setdefault(key, []).append((r, c, rank, suit))
    return [(k, v) for k, v in groups.items() if len(v) >= 2]
