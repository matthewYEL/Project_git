"""M3 look check: how far apart sortgame.c's card looks (sort_sig) land for
the same card seen twice versus a different card -- a joker laid exactly over
it -- so app_config.h SORT_SIG_DIFF sits between the two.

Each deck card is drawn at the M3 board's size (~56-68 px) on a mat twice
with independent noise, blur, gain and a 0-2 px shift (the same card, two
frames), and once with the joker drawn in its place (same box), and with
another random card in its place.

    python sig_check.py
"""
import os

import cv2
import numpy as np

import locate
import sim_card_cnn as S

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, 'friends files', 'trainings', 'image_data', 'deck_labeled')
FW, FH = 512, 384


def sig(gray, c):
    """sortgame.c sort_sig, integer for integer."""
    w, h = c.x1 - c.x0 + 1, c.y1 - c.y0 + 1
    ix, iy, iw, ih = c.x0 + w // 10, c.y0 + h // 10, w - 2 * (w // 10), h - 2 * (h // 10)
    v = []
    for gy in range(4):
        for gx in range(4):
            v.append(sum(int(gray[iy + ih * (8 * gy + 2 + 4 * (q >> 1)) // 32, ix + iw * (8 * gx + 2 + 4 * (q & 1)) // 32])
                         for q in range(4)))
    top = max(max(v), 1)
    return [x * 255 // top for x in v]


def diff(a, b):
    """sort_sig_diff: the closer of as-is and turned 180 deg, mean per cell."""
    d0 = sum(abs(x - y) for x, y in zip(a, b))
    d1 = sum(abs(x - y) for x, y in zip(a, b[::-1]))
    return min(d0, d1) // 16


def draw(card, w, x, y, turn, rng, shift=0):
    h = round(w * card.shape[0] / card.shape[1])
    if turn:
        card = cv2.rotate(card, cv2.ROTATE_180)
    small = cv2.resize(cv2.resize(card, (4 * w, 4 * h), interpolation=cv2.INTER_AREA), (w, h),
                       interpolation=cv2.INTER_NEAREST)
    f = np.full((FH, FW, 3), 50, np.float32) + rng.normal(0, 6, (FH, FW, 3))
    x += shift * rng.integers(-1, 2)
    y += shift * rng.integers(-1, 2)
    f[y:y + h, x:x + w] = small
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.7, 1.3))
    f = f * rng.uniform(0.6, 1.0) + rng.normal(0, rng.uniform(2, 5), f.shape)
    return S.fabric_luma(np.clip(f, 0, 255).astype(np.uint8)).astype(np.uint8)


def main():
    rng = np.random.default_rng(5)
    cards = []
    for cls in sorted(os.listdir(DECK)):
        for f in sorted(os.listdir(os.path.join(DECK, cls))):
            im = cv2.imread(os.path.join(DECK, cls, f))
            cards.append((cls, cv2.resize(im, (300, round(300 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)))
    jokers = [im for cls, im in cards if cls == 'JOKER']
    same, joker, other = [], [], []
    for cls, im in cards:
        if cls == 'JOKER':
            continue
        for _ in range(3):
            w, x, y, turn = int(rng.uniform(56, 68)), int(rng.uniform(40, 380)), int(rng.uniform(20, 260)), rng.random() < 0.5
            g1 = draw(im, w, x, y, turn, rng)
            c1 = locate.locate(g1)
            if len(c1) != 1:
                continue
            s1 = sig(g1, c1[0])
            g2 = draw(im, w, x, y, turn, rng, shift=2)
            c2 = locate.locate(g2)
            if len(c2) == 1:
                same.append(diff(s1, sig(g2, c2[0])))
            gj = draw(jokers[rng.integers(len(jokers))], w, x, y, rng.random() < 0.5, rng)
            cj = locate.locate(gj)
            if len(cj) == 1:
                joker.append(diff(s1, sig(gj, cj[0])))
            o = cards[rng.integers(len(cards))]
            if o[0] != cls:
                go = draw(o[1], w, x, y, rng.random() < 0.5, rng)
                co = locate.locate(go)
                if len(co) == 1:
                    other.append(diff(s1, sig(go, co[0])))

    for name, d in (('same card', same), ('joker over it', joker), ('another card', other)):
        d = sorted(d)
        print(f'{name:<14} n {len(d):4d}  min {d[0]:3d}  p5 {d[len(d) // 20]:3d}  median {d[len(d) // 2]:3d}'
              f'  p95 {d[len(d) * 19 // 20]:3d}  max {d[-1]:3d}')
    for t in (20, 25, 30, 35, 40, 50):
        print(f'threshold {t}: same card re-read {sum(x > t for x in same)}/{len(same)}, '
              f'joker missed {sum(x <= t for x in joker)}/{len(joker)}, other card missed {sum(x <= t for x in other)}/{len(other)}')


if __name__ == '__main__':
    main()
