"""Check M2/locate.c against locate.py, card for card, on rig-like frames.

Each 28 Sep photo is cropped to its card grid, optionally turned 90 deg (the
camera as mounted on the rig), scaled to fit the board's 512x384 frame and
converted the fabric's way ({gray, 0, red} words). Both implementations then
find the cards, and the (row, col), box, colour and conv1 window registers
must agree exactly. The frames are also a realistic end-to-end test: this
prints how many cards each finds and whether the grid is the full 3x3. The
board's own SW1 frame dumps in dumps/*.log are compared too, when present.

  python locate_check.py [path/to/locate_host_test.exe]
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

import locate
import sim_card_cnn as S

HERE = os.path.dirname(os.path.abspath(__file__))
M2 = os.path.join(HERE, 'LAB_5', 'fpga_cnn', 'workspace_lab3', 'M2')
FW, FH = 512, 384
# (name, cv2 rotation, locate orient, ORIENT_* bits)
ORIENTS = [('upright', None, (False, False, False), 0),
           ('cw', cv2.ROTATE_90_CLOCKWISE, (True, True, False), 1 | 2),
           ('ccw', cv2.ROTATE_90_COUNTERCLOCKWISE, (True, False, True), 1 | 4)]


def rig_frame(bgr, rot):
    """Crop to the grid (+ a margin), turn, fit into 512x384 on black."""
    g = S.fabric_luma(bgr).astype(np.uint8)
    cards = locate.locate(g)
    x0 = min(c.x0 for c in cards); x1 = max(c.x1 for c in cards)
    y0 = min(c.y0 for c in cards); y1 = max(c.y1 for c in cards)
    m = int(0.25 * np.median([c.w for c in cards]))
    crop = bgr[max(0, y0 - m):y1 + m + 1, max(0, x0 - m):x1 + m + 1]
    if rot is not None:
        crop = cv2.rotate(crop, rot)
    s = min((FW - 8) / crop.shape[1], (FH - 8) / crop.shape[0])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    out = np.zeros((FH, FW, 3), np.uint8)
    oy, ox = (FH - crop.shape[0]) // 2, (FW - crop.shape[1]) // 2
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out


def python_lines(words, orient):
    """locate.py's lines for a frame of board words ({gray, 0, red})."""
    gray = (words >> 2).astype(np.uint8)
    red = (words & 1).astype(bool)
    info = {}
    cards = locate.locate(gray, orient=orient, info=info)
    nr = max((c.row for c in cards), default=0)
    nc = max((c.col for c in cards), default=0)
    lines = [f'thr {info["threshold"]} n {len(cards)} rows {nr} cols {nc}']
    for c in cards:
        n, area = locate.index_red(red, c)
        x0, y0, step, flags, clip = locate.window_params(c, FW, FH, orient=orient)
        lines.append(f'({c.row},{c.col}) {c.x0} {c.y0} {c.x1} {c.y1} red {int(n * locate.RED_PER > area)} '
                     f'win {x0} {y0} {step} {flags} clip {clip[0]} {clip[1]} {clip[2]} {clip[3]}')
    return lines


def few_card_crops(bgr0):
    """1-2 card layouts cut from a 3x3 photo, with a margin that takes in
    slivers of the neighbouring cards -- the case where card-shaped clutter
    once outnumbered the real cards (29 Sep)."""
    g = S.fabric_luma(bgr0).astype(np.uint8)
    cards = locate.locate(g)
    picks = [('1 card (2,2)', lambda c: (c.row, c.col) == (2, 2)),
             ('2 side by side', lambda c: c.row == 1 and c.col <= 2),
             ('2 stacked', lambda c: c.col == 1 and c.row <= 2)]
    for label, keep in picks:
        sel = [c for c in cards if keep(c)]
        for pad in (10, 40):
            x0 = max(0, min(c.x0 for c in sel) - pad); x1 = max(c.x1 for c in sel) + pad
            y0 = max(0, min(c.y0 for c in sel) - pad); y1 = max(c.y1 for c in sel) + pad
            yield f'{label}, margin {pad}', bgr0[y0:y1 + 1, x0:x1 + 1], len(sel)


def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else os.path.join(M2, 'locate_host_test.exe')
    spec = json.load(open(os.path.join(HERE, 'photos_3x3.json')))
    tmp = tempfile.mkdtemp()
    ok = total = missed = 0
    cases = []
    for name in spec['photos']:
        bgr0 = cv2.imread(os.path.join(HERE, spec['dir'], name))
        cases.append((name[-13:-5], bgr0, 9))
    for name in ('WhatsApp Image 2026-09-28 at 20.32.26.jpeg', 'WhatsApp Image 2026-09-28 at 20.40.56.jpeg'):
        bgr0 = cv2.imread(os.path.join(HERE, spec['dir'], name))
        cases += [(f'{name[-13:-5]} {label}', crop, n) for label, crop, n in few_card_crops(bgr0)]
    frames = []
    for tag, bgr0, n_want in cases:
        for oname, rot, orient, bits in ORIENTS:
            bgr = rig_frame(bgr0, rot)
            words = (S.fabric_luma(bgr).astype(np.uint16) << 2) | S.fabric_red(bgr).astype(np.uint16)
            frames.append((tag, oname, words, orient, bits, n_want))
    # the board's own SW1 frame dumps, if any: real D8M pixels, tilt and the
    # colour fringe included. No card count is asked of them -- cards cut by
    # the frame edge are rightly left out.
    for log in sorted(glob.glob(os.path.join(HERE, 'dumps', '*.log'))):
        for k, w, h, maxval, words in S._pgm_blocks(log):
            if maxval == 1023 and (w, h) == (FW, FH):
                frames.append((f'{os.path.basename(log)} dump {k}', 'upright', words.astype(np.uint16),
                               ORIENTS[0][2], ORIENTS[0][3], None))
    for tag, oname, words, orient, bits, n_want in frames:
        path = os.path.join(tmp, 'frame.bin')
        with open(path, 'wb') as f:
            f.write(np.array([FW, FH], '<u2').tobytes())
            f.write(words.astype('<u2').tobytes())
        want = python_lines(words, orient)
        got = subprocess.run([exe, path, str(bits)], capture_output=True, text=True).stdout.split('\n')
        got = [l for l in got if l.strip()]
        total += 1
        cw = np.median([int(l.split()[3]) - int(l.split()[1]) + 1 for l in want[1:]]) if len(want) > 1 else 0
        found = len(want) - 1
        short = '' if n_want is None or found == n_want else f'   ! found {found} of {n_want}'
        missed += bool(short)
        if want == got:
            ok += 1
            print(f'{tag:<34} {oname:7s} match  {want[0]}  card w~{cw:.0f}px{short}')
        else:
            print(f'{tag:<34} {oname:7s} DIFFER{short}')
            for a, b in zip(want, got + [''] * (len(want) - len(got))):
                if a != b:
                    print('   py:', a, '\n    c:', b)
    print(f'\n{ok}/{total} frames identical between locate.c and locate.py; '
          f'{total - missed}/{total} found every card')
    sys.exit(0 if ok == total and not missed else 1)


if __name__ == '__main__':
    main()
