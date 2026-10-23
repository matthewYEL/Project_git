"""M3 read-policy check: how many CNN reads a card needs, and what accuracy each
read policy keeps (bit-exact network, sim_card_cnn.py).

Every card is read once through all 7 VARIANTS windows (the board's order), so
each policy is just a different prefix of the same reads -- no policy gets
luckier windows than another. Policies:

  one    the as-is window only
  full   M2's prvReadCard: 5-read vote, then the club look on every black card
         (all 7 windows, CLUB_MIN) -- app_rtos.c READ_FULL
  fast   M3's READ_FAST (app_rtos.c): V0+V1 agree (and are confident) -> done
         for a red card; a black card also needs V5 to agree; a joker read
         asks V2 (joker if 2 of 3); any doubt -> the full vote

Sets: the 28 Sep phone photos (held out; native / ~95 / ~65 px, as --photos)
and the deck at the M3 board's size (~56-68 px wide, blurred and noisy like
the D8M; these scans trained the model, so read them as a policy comparison,
not as an accuracy claim). The deck set is the only one with jokers.

    OPENBLAS_NUM_THREADS=1 python read_policy_check.py [--logit-min N ...]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

import locate
import sim_card_cnn as S

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, 'friends files', 'trainings', 'image_data', 'deck_labeled')
FW, FH = 512, 384
DECK_W = [56, 68]                   # card width on the board, px (--deck-w)


def ident(r):
    return 'JOKER' if r[3] else r[0] + r[1][0]


def full(rs, is_red):
    chosen, agree = S._vote(rs[:S.SCAN_READS])
    _, chosen, _ = S._club_look(None, None, None, is_red, rs, chosen, agree)
    return ident(chosen), (S.SCAN_READS if is_red or chosen[3] else len(S.VARIANTS))


def fast(rs, is_red, logit_min, third=2):
    """app_rtos.c READ_FAST, on the 7 reads in VARIANTS order. `third` is the
    window a black card's confirming read uses (2 keeps every read a prefix)."""
    a, b = rs[0], rs[1]
    if a[3] or b[3]:                                    # a joker read: a third look
        if sum(r[3] for r in rs[:3]) >= 2:
            return 'JOKER', 3
        return full(rs, is_red)
    if ident(a) == ident(b) and min(a[2], b[2]) >= logit_min:
        if is_red:
            return ident(a), 2
        if ident(rs[third]) == ident(a):                # black: a third window agrees
            return ident(a), 3
    return full(rs, is_red)


def photo_cards():
    """(set, truth, 7 reads, is_red) for every labelled photo card."""
    spec = json.load(open(os.path.join(HERE, 'photos_3x3.json')))
    base = os.path.join(HERE, spec.get('dir', '.'))
    for name, grid in spec['photos'].items():
        bgr0 = cv2.imread(os.path.join(base, name))
        native = int(np.median([c.w for c in locate.locate(S.fabric_luma(bgr0).astype(np.uint8))]))
        truth = {(r + 1, c + 1): code for r, line in enumerate(grid) for c, code in enumerate(line)}
        for target, interp, tag in ((None, None, 'photo native'), (95, cv2.INTER_AREA, 'photo ~95 area'),
                                    (65, cv2.INTER_AREA, 'photo ~65 area'), (65, cv2.INTER_NEAREST, 'photo ~65 nearest')):
            s = 1.0 if target is None else target / native
            if s > 1.05:
                continue
            bgr = bgr0 if target is None else cv2.resize(bgr0, None, fx=s, fy=s, interpolation=interp)
            gray, red = S.fabric_luma(bgr), S.fabric_red(bgr)
            for card in locate.locate(gray.astype(np.uint8)):
                code = truth.get((card.row, card.col))
                if code is None:
                    continue
                n, area = locate.index_red(red, card)
                is_red = n * locate.RED_PER > area
                base_w = locate.window_params(card, gray.shape[1], gray.shape[0])
                rs, _ = S._read_variants(gray, card, base_w, is_red, len(S.VARIANTS))
                t = S.parse_truth(code)
                yield tag, ('JOKER' if t[0] is None else t[0] + t[1][0]), rs, is_red
        print(f'  {name}: done', flush=True)


def deck_frame(card, rng, w):
    """One deck card at the M3 board's size on a dark mat, blurred like the D8M
    (finetune_fcs_blur.frame, with the size pinned)."""
    if rng.random() < 0.5:
        card = cv2.rotate(card, cv2.ROTATE_180)
    h = round(w * card.shape[0] / card.shape[1])
    small = cv2.resize(cv2.resize(card, (4 * w, 4 * h), interpolation=cv2.INTER_AREA), (w, h),
                       interpolation=cv2.INTER_NEAREST)
    mat = np.full((FH, FW, 3), rng.uniform(35, 75), np.float32) + rng.normal(0, 6, (FH, FW, 3))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-5, 5), 1.0)
    M[0, 2] += FW / 2 + rng.uniform(-150, 150) - w / 2
    M[1, 2] += FH / 2 + rng.uniform(-100, 100) - h / 2
    f = cv2.warpAffine(small.astype(np.float32), M, (FW, FH))
    m = cv2.warpAffine(np.ones((h, w), np.float32), M, (FW, FH))[..., None]
    f = f * m + mat * (1 - m)
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.7, 1.4))
    f = f * rng.uniform(0.5, 1.0) + rng.normal(0, rng.uniform(2, 5), f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def deck_cards(per_card, seed):
    rng = np.random.default_rng(seed)
    for cls in sorted(os.listdir(DECK)):
        for f in sorted(os.listdir(os.path.join(DECK, cls))):
            im = cv2.imread(os.path.join(DECK, cls, f))
            im = cv2.resize(im, (400, round(400 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
            truth = 'JOKER' if cls == 'JOKER' else cls[:-1] + cls[-1]
            done = tries = 0
            while done < per_card and tries < 4 * per_card:
                tries += 1
                bgr = deck_frame(im, rng, int(rng.uniform(*DECK_W)))
                gray, red = S.fabric_luma(bgr), S.fabric_red(bgr)
                cs = locate.locate(gray.astype(np.uint8))
                if len(cs) != 1:
                    continue
                n, area = locate.index_red(red, cs[0])
                is_red = n * locate.RED_PER > area
                base_w = locate.window_params(cs[0], FW, FH)
                rs, _ = S._read_variants(gray, cs[0], base_w, is_red, len(S.VARIANTS))
                done += 1
                yield f'deck {DECK_W[0]}-{DECK_W[1]} px', truth, rs, is_red
        print(f'  deck {cls}: done', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logit-min', type=int, nargs='*', default=[0, 1000, 1500, 2000])
    ap.add_argument('--per-card', type=int, default=3, help='deck frames per card')
    ap.add_argument('--skip-photos', action='store_true')
    ap.add_argument('--skip-deck', action='store_true')
    ap.add_argument('--cached', action='store_true', help='re-score the reads saved by the last run')
    ap.add_argument('--deck-w', type=int, nargs=2, default=list(DECK_W), metavar=('MIN', 'MAX'),
                    help='deck card width range on the board, px')
    args = ap.parse_args()
    DECK_W[:] = args.deck_w

    import pickle
    cache = os.path.join(HERE, 'dumps', f'read_policy_reads{"_nophotos" if args.skip_photos else ""}'
                         f'{"_nodeck" if args.skip_deck else "_deck%d-%d" % tuple(DECK_W)}.pkl')
    if args.cached:
        rows = pickle.load(open(cache, 'rb'))
    else:
        rows = []
        if not args.skip_photos:
            rows += list(photo_cards())
        if not args.skip_deck:
            rows += list(deck_cards(args.per_card, seed=21))
        pickle.dump(rows, open(cache, 'wb'))

    sets = list(dict.fromkeys(r[0] for r in rows)) + ['ALL']
    pols = [('one', lambda rs, red: (ident(rs[0]), 1)), ('full', full)]
    pols += [(f'fast L{L}', (lambda L: lambda rs, red: fast(rs, red, L))(L)) for L in args.logit_min]
    pols += [('fast V5', lambda rs, red: fast(rs, red, args.logit_min[0], 5))]
    print(f'\n{"set":<20}{"cards":>6}' + ''.join(f'{p:>16}' for p, _ in pols))
    for s in sets:
        sel = [r for r in rows if s == 'ALL' or r[0] == s]
        line = f'{s:<20}{len(sel):>6}'
        for _, fn in pols:
            res = [fn(rs, red) for _, _, rs, red in sel]
            ok = sum(got == t for (got, _), (_, t, _, _) in zip(res, sel))
            nr = sum(n for _, n in res) / max(len(sel), 1)
            line += f'{ok:>6}/{len(sel):<3}{nr:4.1f}r '
        print(line)
    print('(ok/cards, then mean CNN reads per card; 1.73 s per read on the board)')

    # first-read confidence of right vs wrong as-is reads, for READ_LOGIT_MIN
    good = sorted(rs[0][2] for _, t, rs, _ in rows if ident(rs[0]) == t and not rs[0][3])
    bad = sorted(rs[0][2] for _, t, rs, _ in rows if ident(rs[0]) != t and not rs[0][3])
    if good:
        print(f'\nas-is rank logit, right reads: min {good[0]}  p5 {good[len(good) // 20]}  median {good[len(good) // 2]}')
    if bad:
        print(f'as-is rank logit, wrong reads: {bad}')
    misses = [(s, t, [ident(r) for r in rs], red) for s, t, rs, red in rows
              if fast(rs, red, args.logit_min[0])[0] != t]
    for m in misses:
        print('fast miss:', m)


if __name__ == '__main__':
    sys.exit(main())
