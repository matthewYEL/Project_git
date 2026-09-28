"""Off-board tooling for the HDMI text layer (text_overlay.v).

  font      build font8x16.hex, the layer's CP437 font ROM, from a Linux console
            font (run it where the font lives -- WSL):
                wsl python3 text_layer.py font [--psf F.psf.gz] [--sheet sheet.png]
  render    draw a text-RAM dump (2400 hex words, one per line, as
            M2/gui_host_test.c writes it) the way the fabric composites it:
                python text_layer.py render ram.hex gui.png
  selftest  replay text_overlay.v's 4-stage pipeline clock by clock over a full
            VGA_Controller frame of random cells and check it against a direct
            render (the RTL check -- there is no licensed simulator here):
                python text_layer.py selftest

The ROM is 256 glyphs x 16 rows, one byte per row, bit 7 = leftmost pixel, in
CP437 order, so the C code writes standard CP437 codes (0x03 = heart ...).
The default source is console-setup's FullGreek-VGA16, the classic VGA 8x16
design; it has every glyph the GUI draws.
"""
import argparse
import gzip
import random
import struct
import sys

COLS, ROWS = 80, 30            # 640x480 raster in 8x16 cells
CELLS = COLS * ROWS            # text RAM depth
FONT_HEX = "font8x16.hex"

# camera_capture.v's preview window and CNN crop box (PV_X/PV_Y there)
PV_X, PV_Y, PV_W, PV_H = 8, 32, 320, 240
BOX_X0, BOX_X1 = PV_X + 64, PV_X + 255      # inclusive edges
BOX_Y0, BOX_Y1 = PV_Y + 24, PV_Y + 215

# 16-colour CGA palette -- same order as text_overlay.v and hdmi_gui.h
PALETTE = [0x000000, 0x0000AA, 0x00AA00, 0x00AAAA, 0xAA0000, 0xAA00AA, 0xAA5500, 0xAAAAAA,
           0x555555, 0x5555FF, 0x55FF55, 0x55FFFF, 0xFF5555, 0xFF55FF, 0xFFFF55, 0xFFFFFF]

# CP437 0x01-0x1F are drawn glyphs, not controls; Python's codec maps them to
# controls, so take them from the standard table.
CP437_LOW = [0x0000, 0x263A, 0x263B, 0x2665, 0x2666, 0x2663, 0x2660, 0x2022,
             0x25D8, 0x25CB, 0x25D9, 0x2642, 0x2640, 0x266A, 0x266B, 0x263C,
             0x25BA, 0x25C4, 0x2195, 0x203C, 0x00B6, 0x00A7, 0x25AC, 0x21A8,
             0x2191, 0x2193, 0x2192, 0x2190, 0x221F, 0x2194, 0x25B2, 0x25BC]
FALLBACK = {0x25BA: 0x25B6, 0x25C4: 0x25C0}      # the font draws the pointers as triangles

# every code the GUI (M2/hdmi_gui.c) draws; each must have a non-blank glyph
USED = (list(range(0x21, 0x7F)) + [0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x0A, 0x1F, 0xFA, 0xFE] +
        [0xB1, 0xDC, 0xDF] +
        [0xB3, 0xB4, 0xBF, 0xC0, 0xC3, 0xC4, 0xD9, 0xDA])


def cp437_unicode(code):
    if code < 0x20:
        return CP437_LOW[code]
    if code == 0x7F:
        return 0x2302
    return ord(bytes([code]).decode("cp437"))


def load_psf(path):
    """-> (glyph bytes list, {unicode: glyph index})"""
    d = gzip.open(path).read() if path.endswith(".gz") else open(path, "rb").read()
    uni = {}
    if d[:2] == b"\x36\x04":                          # PSF1: always 8 wide
        mode, height = d[2], d[3]
        n = 512 if mode & 1 else 256
        glyphs = [d[4 + i * height: 4 + (i + 1) * height] for i in range(n)]
        tab, i, g = d[4 + n * height:], 0, 0
        while mode & 6 and g < n and i + 1 < len(tab):
            u = struct.unpack_from("<H", tab, i)[0]
            i += 2
            if u == 0xFFFF:
                g += 1
            elif u != 0xFFFE:
                uni.setdefault(u, g)
        width = 8
    elif d[:4] == b"\x72\xb5\x4a\x86":                # PSF2
        hsize, flags, n, bpg, height, width = struct.unpack_from("<6I", d, 8)
        glyphs = [d[hsize + i * bpg: hsize + (i + 1) * bpg] for i in range(n)]
        if flags & 1:
            for g, ent in enumerate(d[hsize + n * bpg:].split(b"\xff")[:n]):
                for ch in ent.split(b"\xfe")[0].decode("utf8", "ignore"):
                    uni.setdefault(ord(ch), g)
    else:
        sys.exit(f"{path}: not a PSF font")
    if (width, height) != (8, 16):
        sys.exit(f"{path}: glyphs are {width}x{height}, the ROM needs 8x16")
    return glyphs, uni


def cmd_font(a):
    glyphs, uni = load_psf(a.psf)
    rom, missing = [], []
    for code in range(256):
        u = cp437_unicode(code)
        g = uni.get(u, uni.get(FALLBACK.get(u, -1)))
        if code == 0 or g is None:
            if code:
                missing.append(code)
            rom += [0] * 16
        else:
            rom += list(glyphs[g])
    blank = [c for c in USED if not any(rom[c * 16:(c + 1) * 16])]
    assert not blank, "GUI glyphs missing from the font: " + " ".join("%02X" % c for c in blank)
    assert len(rom) == 4096
    with open(a.out, "w") as f:
        f.write("".join("%02x\n" % b for b in rom))
    print(f"{a.out}: 256 glyphs x 16 rows from {a.psf}")
    if missing:
        print("  blank (not in the font, unused by the GUI):", " ".join("%02X" % c for c in missing))
    if a.sheet:
        from PIL import Image
        img = Image.new("RGB", (16 * 10 * 3, 16 * 18 * 3), (40, 40, 40))
        px = img.load()
        for code in range(256):
            ox, oy = (code % 16) * 10 * 3 + 3, (code // 16) * 18 * 3 + 3
            for r in range(16):
                for c in range(8):
                    on = rom[code * 16 + r] >> (7 - c) & 1
                    for dy in range(3):
                        for dx in range(3):
                            px[ox + c * 3 + dx, oy + r * 3 + dy] = (255, 255, 255) if on else (0, 0, 0)
        img.save(a.sheet)
        print(f"  glyph sheet -> {a.sheet} (row = high nibble, column = low nibble)")


def read_hex(path, n):
    words = [int(t, 16) for t in open(path).read().split()]
    assert len(words) == n, f"{path}: {len(words)} words, expected {n}"
    return words


def pixel(ram, font, x, y):
    """What the HDMI mux (camera_capture.v) shows at raster (x, y); the preview's
    video is drawn as a flat grey stand-in."""
    cell = ram[(y >> 4) * COLS + (x >> 3)]
    fg_on = font[(cell & 0xFF) * 16 + (y & 15)] >> (7 - (x & 7)) & 1
    in_win = PV_X <= x < PV_X + PV_W and PV_Y <= y < PV_Y + PV_H
    edge = in_win and (((x in (BOX_X0, BOX_X1)) and BOX_Y0 <= y <= BOX_Y1) or
                       ((y in (BOX_Y0, BOX_Y1)) and BOX_X0 <= x <= BOX_X1))
    if edge:
        return 0x00FF00
    if fg_on:
        return PALETTE[(cell >> 8) & 15]
    if in_win:
        return 0x3C3C46
    return PALETTE[(cell >> 12) & 15]


def cmd_render(a):
    from PIL import Image
    font = read_hex(a.font, 4096)
    ram = read_hex(a.ram, CELLS)
    img = Image.new("RGB", (640, 480))
    img.putdata([((p >> 16) & 255, (p >> 8) & 255, p & 255)
                 for y in range(480) for x in range(640) for p in [pixel(ram, font, x, y)]])
    img.save(a.png)
    print(f"{a.png}: 640x480 from {a.ram}")


def cmd_selftest(a):
    """text_overlay.v, register for register, driven by VGA_Controller.v's
    counters (801 clocks per line, 525 lines -- its H_Cont<H_TOTAL off-by-one)."""
    font = read_hex(a.font, 4096)
    rnd = random.Random(1)
    ram = [rnd.getrandbits(16) for _ in range(CELLS)]
    H_BLANK, H_TOTAL, V_BLANK, V_TOTAL = 160, 800, 44, 524

    def rd(addr):                          # past DEPTH only in blanking: don't care
        return ram[addr] if addr < CELLS else 0

    # pipeline registers, names as in text_overlay.v
    a1 = yr1 = xb1 = 0
    cell_q = xb2 = yr2 = 0
    bits3 = attr3 = xb3 = 0
    fg_on = fg = bg = 0
    hist = []                              # (request, x, y) per clock, for the 4-clock lag
    checked = 0
    for v in range(V_TOTAL + 1):
        for h in range(H_TOTAL + 1):
            req = H_BLANK <= h < H_TOTAL and V_BLANK <= v < V_TOTAL
            x = h - H_BLANK if h >= H_BLANK else 0
            y = v - V_BLANK if v >= V_BLANK else 0
            # the outputs present during this clock describe the pixel from 4 clocks ago
            if len(hist) >= 4 and hist[-4][0]:
                _, px_, py_ = hist[-4]
                cell_d = ram[(py_ >> 4) * COLS + (px_ >> 3)]
                exp_on = font[(cell_d & 0xFF) * 16 + (py_ & 15)] >> (7 - (px_ & 7)) & 1
                got = (fg_on, fg, bg)
                exp = (exp_on, PALETTE[(cell_d >> 8) & 15], PALETTE[(cell_d >> 12) & 15])
                assert got == exp, f"pixel ({px_},{py_}): pipeline {got} != direct {exp}"
                checked += 1
            hist.append((req, x, y))
            # one rising edge of VGA_CLK: every stage samples the old values
            fg_on, fg, bg = (bits3 >> (7 - xb3)) & 1, PALETTE[attr3 & 15], PALETTE[(attr3 >> 4) & 15]
            bits3, attr3, xb3 = font[(cell_q & 0xFF) * 16 + yr2], cell_q >> 8, xb2
            cell_q, xb2, yr2 = rd(a1), xb1, yr1
            a1, yr1, xb1 = ((y >> 4) << 6) + ((y >> 4) << 4) + (x >> 3), y & 15, x & 7
    assert checked == 640 * 480, checked
    print(f"selftest OK: {checked} pixels, pipeline == direct render at a 4-clock lag")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    s = p.add_subparsers(dest="cmd", required=True)
    f = s.add_parser("font")
    f.add_argument("--psf", default="/usr/share/consolefonts/FullGreek-VGA16.psf.gz")
    f.add_argument("--out", default=FONT_HEX)
    f.add_argument("--sheet", help="also write a PNG of all 256 glyphs")
    r = s.add_parser("render")
    r.add_argument("ram")
    r.add_argument("png")
    r.add_argument("--font", default=FONT_HEX)
    t = s.add_parser("selftest")
    t.add_argument("--font", default=FONT_HEX)
    a = p.parse_args()
    {"font": cmd_font, "render": cmd_render, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
