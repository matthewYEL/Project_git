"""Check conv_layer.v's window mapper (WIN_MAP) against sim_card_cnn.window_map.

No RTL simulator runs on this machine, so this transliterates the Verilog
expressions -- same operand widths, same order -- and compares what conv1 would
read for every one of its 384x384 taps with the simulator's model, over random
windows (every flag combination, negative origins, clips at the frame edges).
It also checks that ghrd_top.v's reset values reproduce the old 384x384 crop
exactly, which is what keeps guided mode unchanged.

  python win_map_check.py        # prints PASS or the first mismatch
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', '..')))   # lab 5/sim_card_cnn.py
import sim_card_cnn as S                                                     # noqa: E402

DIM, FRAME_W, FRAME_H, COLB = 384, 512, 384, 9


def rtl_read(frame, x0, y0, step, flags, clip):
    """conv1's view per conv_layer.v: for each virtual tap (iy, ix) in bounds,
    tap_ok and rd_next -> the word read, or 0 when the tap is masked."""
    cx0, cy0, cx1, cy1 = clip
    ix = np.arange(DIM)[None, :].repeat(DIM, 0)
    iy = np.arange(DIM)[:, None].repeat(DIM, 1)
    m9 = 0x1FF
    wu = ((DIM - 1 - (ix & m9)) & m9) if flags & 2 else (ix & m9)       # [8:0]
    wv = ((DIM - 1 - (iy & m9)) & m9) if flags & 4 else (iy & m9)
    pu = (wu * step) & 0x3FFFF                                           # [17:0]
    pv = (wv * step) & 0x3FFFF
    su, sv = pu >> 8, pv >> 8                                            # pu[17:8]
    def s12(v):                                                          # signed [11:0]
        v = v & 0xFFF
        return np.where(v >= 0x800, v - 0x1000, v)
    fcol = s12(x0 + (sv if flags & 1 else su))
    frow = s12(y0 + (su if flags & 1 else sv))
    in_window = ((fcol >= cx0) & (fcol <= cx1) & (frow >= cy0) & (frow <= cy1)
                 & (fcol >= 0) & (fcol < FRAME_W) & (frow >= 0) & (frow < FRAME_H))
    addr = ((frow & 0x1FF) << COLB) | (fcol & 0x1FF)                     # {frow[8:0], fcol[8:0]}
    flat = frame.reshape(-1)
    return np.where(in_window, flat[np.where(in_window, addr, 0)], 0)


def main():
    rng = np.random.default_rng(29)
    frame = rng.integers(0, 256, (FRAME_H, FRAME_W), dtype=np.int64)

    # reset values = the old crop, 1:1: camera x 128..511, y 48..431
    got = S.window_map(frame, 64, 0, 256, 0, (0, 0, FRAME_W - 1, FRAME_H - 1))
    rtl = rtl_read(frame, 64, 0, 256, 0, (0, 0, FRAME_W - 1, FRAME_H - 1))
    assert np.array_equal(got, frame[0:384, 64:448]), 'sim default window is not the old crop'
    assert np.array_equal(rtl, frame[0:384, 64:448]), 'RTL default window is not the old crop'

    for n in range(3000):
        x0 = int(rng.integers(-1024, 1024))
        y0 = int(rng.integers(-1024, 1024))
        if n % 3:                               # most trials near the frame, where cards are
            x0 = int(rng.integers(-150, FRAME_W))
            y0 = int(rng.integers(-150, FRAME_H))
        step = int(rng.integers(1, 512))
        flags = int(rng.integers(0, 8))
        a, b = sorted(rng.integers(0, 1024, 2))
        c, d = sorted(rng.integers(0, 1024, 2))
        clip = (int(a), int(c), int(b), int(d))
        if n % 4 == 0:
            clip = (0, 0, FRAME_W - 1, FRAME_H - 1)
        want = S.window_map(frame, x0, y0, step, flags, clip)
        got = rtl_read(frame, x0, y0, step, flags, clip)
        if not np.array_equal(want, got):
            bad = np.argwhere(want != got)[0]
            sys.exit(f'MISMATCH trial {n}: x0={x0} y0={y0} step={step} flags={flags} clip={clip} '
                     f'at (iy, ix)={tuple(bad)}: sim {want[tuple(bad)]} rtl {got[tuple(bad)]}')
    print('PASS: window mapper, 3000 random windows + the reset window, bit-identical to window_map')


if __name__ == '__main__':
    main()
