"""Export a DualHeadCardCNN checkpoint as the 14 fixed-point hex files the FPGA
RTL and sim_card_cnn.py read.

FORMAT: the RTL ($readmemh into `reg signed [15:0]`, conv_layer.v / fc_layer.v)
treats every weight and bias as Q6.10 -- a 16-bit two's-complement integer equal
to round(value * 1024). One value per line, four lowercase hex digits, nothing
else in the file. sim_card_cnn.py parses the same files with int(line, 16) and
asserts the line count equals the tensor's element count.

This is NOT export_weights_hex.py. That script writes INT8 bytes with a
per-tensor float scale, which the fabric has no way to apply -- it exists for
the C-header path. Feeding its output to the RTL produces a network of noise.

FILENAMES are the RTL's own (fcs_w, fcrank_w, ...), written directly, so the
historical rename-by-hand step and the trap that came with it are gone.

ORDER within a file is PyTorch's row-major flatten of the parameter tensor:
conv [out][in][ky][kx], linear [out][in]. That is exactly the RTL's
    w_addr = ((f*IN_CH + c)*K + ky)*K + kx      (conv_layer.v)
    w_addr = o*N_IN + k                         (fc_layer.v)
Column 2304 of fcs_w is the colour-bit weight (fc_layer.v APPEND_COLOUR).

    python export_weights_rtl.py --checkpoint checkpoints_13/last_dualhead_model.pt \
        --out_dir ../../weights ../../LAB_5/fpga_cnn \
        --probe_dir image_data/deck_corners_384_final/val --fit_range 16 \
        --hps_header ../../LAB_5/fpga_cnn/workspace_lab3/Atlas-Blinking-LED-Baremetal-GNU/fcs_w_ddr.h

fcs_w goes to the board through --hps_header (the HPS copies it into DDR3);
fcs_w.hex is still written because sim_card_cnn.py reads it.

Refuses to write anything if any layer saturates the 16-bit range. Prints a
per-layer rounding table, re-reads every file with the simulator's parser to
prove it round-trips, and drops a weights_manifest.json beside the files.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# the model class lives in the trainer; import it rather than keep a 7th copy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from card_cnn_model import (  # noqa: E402
    DualHeadCardCNN, get_transforms, parse_class_name, CONV1_CH_DEFAULT, FEAT_DIM,
    QUANT_BITS_DEFAULT, img_dw_for,
)

# (rtl file stem, module attribute, parameter, expected shape) -- the order and
# shapes mirror sim_card_cnn.py's read_hex table and card_cnn_core.v's WFILE/BFILE
def layer_table(conv1_ch: int):
    """The (rtl stem, attribute, parameter, expected shape) table.

    conv1_ch is a parameter because it is the design's main M10K lever, so the
    shapes of conv1_w/conv1_b/conv2_w move with it."""
    return [
        ("conv1_w",  "conv1",     "weight", (conv1_ch, 1, 5, 5)),
        ("conv1_b",  "conv1",     "bias",   (conv1_ch,)),
        ("conv2_w",  "conv2",     "weight", (16, conv1_ch, 5, 5)),
        ("conv2_b",  "conv2",     "bias",   (16,)),
        ("conv3_w",  "conv3",     "weight", (16, 16, 5, 5)),
        ("conv3_b",  "conv3",     "bias",   (16,)),
        ("conv4_w",  "conv4",     "weight", (16, 16, 5, 5)),
        ("conv4_b",  "conv4",     "bias",   (16,)),
        ("fcs_w",    "fc_shared", "weight", (64, 2305)),
        ("fcs_b",    "fc_shared", "bias",   (64,)),
        ("fcrank_w", "fc_rank",   "weight", (13, 64)),
        ("fcrank_b", "fc_rank",   "bias",   (13,)),
        ("fcsuit_w", "fc_suit",   "weight", (4, 64)),
        ("fcsuit_b", "fc_suit",   "bias",   (4,)),
        ("fcjoker_w", "fc_joker", "weight", (2, 64)),
        ("fcjoker_b", "fc_joker", "bias",   (2,)),
    ]

LAYERS = layer_table(CONV1_CH_DEFAULT)

INT16_MIN, INT16_MAX = -32768, 32767

# ROMs the RTL reads through a narrowed port. The .hex FILE FORMAT DOES NOT
# CHANGE -- still four hex digits of a 16-bit two's-complement Q6.10 value.
# Only the Verilog `reg signed [9:0] w_mem` is narrower, and $readmemh of a
# 4-digit value into a 10-bit reg keeps the low 10 bits, which is the correct
# two's-complement value provided it fits. This table is what enforces "fits".
#
# Why bother: an M10K holds 512 words at x16 but 1024 at x10, so a narrowed ROM
# costs half the blocks. fcs_w is NOT here any more: it lives in HPS DDR3 at
# full 16 bits (card_cnn_core.v, see --hps_header), so it has no ROM to narrow.
NARROW_ROM_BITS = {
    "conv3_w": 10,
    "conv4_w": 10,
}

# fcs_w is written with 14 fractional bits (Q1.14), not Q6.10. Its weights are
# tiny -- max |w| ~0.1 on the 384x384 model, 98 LSB in Q6.10 with ~16% at 0 or
# +/-1 -- and that rounding flipped 10S to 8S in the fixed-point model while the
# float model read it at probability 1.00. --fit_range cannot help: conv4's
# scale cancels out of fc_shared's weights, so they land at the same magnitude
# whatever the target. The DDR3 path stores fcs_w at full 16 bits anyway, so
# the extra resolution is free; fc_layer.v shifts that layer by W_FRAC=14
# (card_cnn_core.v FCS_W_FRAC) and sim_card_cnn.py by the manifest's
# "fcs_w_frac". Q1.14 holds |w| < 2.0.
FCS_W_FRAC = 14


def write_hps_header(q: np.ndarray, path: Path):
    """fcs_w as a C array for atlas_main.c, which copies it into DDR3 for the
    fabric (one value per 32-bit word at FCS_W_DDR_BASE). Same quantised values
    as fcs_w.hex, so the board, the .hex and sim_card_cnn.py cannot disagree."""
    lines = [
        "/* GENERATED by export_weights_rtl.py -- do not edit.",
        f" * fc_shared weights, Q{16 - FCS_W_FRAC}.{FCS_W_FRAC} (NOT Q6.10), PyTorch [out][in] order (in = 2304 features +",
        " * the colour bit). atlas_main.c copies these into HPS DDR3 at startup;",
        " * card_cnn_core.v reads them back through f2h_sdram0. */",
        "#ifndef FCS_W_DDR_H",
        "#define FCS_W_DDR_H",
        "#include <stdint.h>",
        f"#define FCS_W_COUNT {q.size}u",
        "static const int16_t FCS_W_Q[FCS_W_COUNT] = {",
    ]
    vals = q.tolist()
    for i in range(0, len(vals), 16):
        lines.append("    " + ", ".join(str(v) for v in vals[i:i + 16]) + ",")
    lines += ["};", "#endif", ""]
    path.write_text("\n".join(lines), newline="\n")


def load_model(ckpt: Path) -> DualHeadCardCNN:
    # the trainer saves a plain state_dict, which torch 2.7's weights_only=True
    # default loads fine; a whole-model pickle would need weights_only=False
    state = torch.load(ckpt, map_location="cpu")
    # conv1 width varies between builds; constructing the default and loading
    # blind fails with a shape error that reads like a corrupt checkpoint
    model = DualHeadCardCNN.from_state_dict(state)   # strict: drift fails here
    model.eval()
    return model


def quantise(t: torch.Tensor, frac_bits: int):
    """Q<16-frac>.<frac> of a float tensor. Returns (flat int64 numpy, stats)."""
    w = t.detach().cpu().double().contiguous().flatten().numpy()
    scale = float(1 << frac_bits)
    q = np.round(w * scale)
    n_sat = int(((q > INT16_MAX) | (q < INT16_MIN)).sum())
    q = np.clip(q, INT16_MIN, INT16_MAX).astype(np.int64)
    back = q / scale
    rms_w = float(np.sqrt(np.mean(w * w))) or 1e-12
    stats = {
        "count": int(w.size),
        "max_abs": float(np.abs(w).max()),
        "max_abs_q": int(np.abs(q).max()),
        "saturated": n_sat,
        "zeros_pct": 100.0 * float((q == 0).mean()),
        "tiny_pct": 100.0 * float((np.abs(q) <= 1).mean()),
        "rms_err_pct": 100.0 * float(np.sqrt(np.mean((w - back) ** 2))) / rms_w,
    }
    return q, stats


def write_hex(q: np.ndarray, path: Path):
    # explicit LF: both $readmemh and the sim's int(l,16) accept LF or CRLF, and
    # LF keeps the output byte-identical across OSes and free of autocrlf churn
    with open(path, "w", newline="\n") as f:
        for v in q.tolist():
            f.write(f"{v & 0xFFFF:04x}\n")


def read_hex_like_sim(path: Path) -> np.ndarray:
    """sim_card_cnn.py read_hex, verbatim semantics."""
    v = [int(l, 16) for l in open(path) if l.strip()]
    return np.array([x - 0x10000 if x >= 0x8000 else x for x in v], dtype=np.int64)


CHAIN = ("conv1", "conv2", "conv3", "conv4", "fc_shared")  # ReLU layers, forward order
HEADS = ("fc_rank", "fc_suit", "fc_joker")
N_FEAT = FEAT_DIM   # fc_shared inputs that are activations; col 2304 is the colour bit


def load_probe_images(probe_dir: Path, limit: int, quant_bits):
    """(stacked image tensor, colour flags) from a class-labelled folder, using
    the trainer's own eval transform and its folder-name parser -- quantised the
    way the model was trained, so the measured peaks are the ones the RTL sees."""
    tf = get_transforms(train=False, quant_bits=quant_bits)
    from PIL import Image
    xs, cols = [], []
    for class_dir in sorted(d for d in probe_dir.iterdir() if d.is_dir()):
        _r, _s, colour, _j = parse_class_name(class_dir.name)
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            xs.append(tf(Image.open(p).convert("RGB")))
            cols.append(float(colour))
            if len(xs) >= limit:
                break
        if len(xs) >= limit:
            break
    if not xs:
        sys.exit(f"probe: no images under {probe_dir}")
    return torch.stack(xs), torch.tensor(cols)


def measure_peaks(model: DualHeadCardCNN, x: torch.Tensor, cols: torch.Tensor) -> dict:
    """Max post-ReLU activation per CHAIN layer over the probe images. This is
    what saturates in the RTL: conv_layer.v / fc_layer.v clip every output at
    32767/1024 = 32.0 after ReLU, silently. Weight statistics cannot show it."""
    peaks = {}

    def hook(name):
        def fn(_m, _i, out):
            peaks[name] = max(peaks.get(name, 0.0), float(out.clamp(min=0).max()))
        return fn

    # The heads' logits saturate too -- fc_layer.v clips them at +/-32.0 -- and
    # they are not ReLU'd, so their peak is the largest MAGNITUDE. This is the
    # check that was missing: the 384x384 model's rank logits reach ~55, both
    # 10 and 8 clipped to 31.999 on a 10S, and the tie went to 8.
    def head_hook(name):
        def fn(_m, _i, out):
            peaks[name] = max(peaks.get(name, 0.0), float(out.abs().max()))
        return fn

    handles = [getattr(model, n).register_forward_hook(hook(n)) for n in CHAIN] + \
              [getattr(model, n).register_forward_hook(head_hook(n)) for n in HEADS]
    with torch.no_grad():
        model(x, cols)
    for h in handles:
        h.remove()
    return peaks


def print_peaks(peaks: dict, title: str):
    ceiling = INT16_MAX / 1024.0
    print(f"\n{title}")
    print(f"  {'layer':<10} {'max relu out':>12}   Q6.10 ceiling {ceiling:.2f}, 1-bit headroom 16.00"
          "   (heads: max |logit|)")
    worst = 0.0
    for n in CHAIN + HEADS:
        v = peaks.get(n, 0.0)
        worst = max(worst, v)
        flag = "  SATURATES" if v >= ceiling else ("  <1 bit headroom" if v >= 16.0 else "")
        print(f"  {n:<10} {v:12.3f}{flag}")
    if worst >= ceiling:
        print("  !! at least one layer exceeds the Q6.10 range on real inputs; the RTL will clip it")
    return worst


def fit_range(model: DualHeadCardCNN, peaks: dict, target: float) -> dict:
    """Shrink each CHAIN layer's output so its measured peak lands at `target`,
    and push the inverse into the next layer so every logit is unchanged.

    Exact in float, not an approximation: the net is ReLU/maxpool/linear, all
    positively homogeneous, so relu(s*x) == s*relu(x) and the next layer being
    linear in its input absorbs 1/s. The colour bit (fc_shared column 2304) is a
    constant 1.0 in the RTL, not an activation, so conv3's compensation must not
    touch it. Head weights absorb fc_shared's 1/s, which brings every logit back
    to the checkpoint's value; then each head is itself shrunk (weights AND
    bias) so its peak |logit| lands at `target`. That scales all of a head's
    logits by one positive factor, so every argmax -- and the joker comparison --
    is identical to the checkpoint's; only rank_score's units change.

    What changes is only WHERE the fixed-point noise lands: smaller activations
    are exactly what Q6.10 needs, at the price of smaller (coarser) weights in
    the shrunk layers. The sim selftest is the arbiter of that trade.
    """
    scales, carry = {}, 1.0
    with torch.no_grad():
        for name in CHAIN:
            layer = getattr(model, name)
            # undo the previous layer's shrink on this layer's input
            if name == "fc_shared":
                layer.weight[:, :N_FEAT] /= carry
            else:
                layer.weight /= carry
            # this layer's output is now back at its measured scale; shrink it
            s = min(1.0, target / peaks[name]) if peaks.get(name, 0.0) > 0 else 1.0
            layer.weight *= s
            layer.bias *= s
            scales[name] = s
            carry = s
        for name in HEADS:
            head = getattr(model, name)
            head.weight /= carry            # logits back at the checkpoint's values...
            s = min(1.0, target / peaks[name]) if peaks.get(name, 0.0) > 0 else 1.0
            head.weight *= s                # ...then the whole head shrunk into range
            head.bias *= s
            scales[name] = s
    return scales


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", nargs="+", required=True,
                    help="one or more directories; each gets the 14 .hex files + weights_manifest.json")
    ap.add_argument("--frac_bits", type=int, default=10,
                    help="fractional bits; MUST equal FRAC_BITS in conv_layer.v / fc_layer.v (default 10 = Q6.10)")
    ap.add_argument("--probe_dir", default=None,
                    help="class-labelled image folder (e.g. the val split) for the activation-headroom probe")
    ap.add_argument("--fit_range", type=float, default=None, metavar="PEAK",
                    help="rescale layers so no measured activation exceeds PEAK (needs --probe_dir). Exact "
                         "in float -- logits are unchanged -- it only moves where the fixed-point noise lands. "
                         "16.0 leaves one bit of headroom under the 32.0 ceiling for inputs the probe never saw")
    ap.add_argument("--probe_limit", type=int, default=400)
    ap.add_argument("--hps_header", default=None, metavar="PATH",
                    help="also write fcs_w as a C header (fcs_w_ddr.h) for the Arm DS app, which "
                         "copies it into DDR3 for the fabric. Pass the Atlas-Blinking-LED-"
                         "Baremetal-GNU project's fcs_w_ddr.h -- the board needs it.")
    ap.add_argument("--quant_bits", type=int, default=QUANT_BITS_DEFAULT,
                    help=f"input grey-level bits the checkpoint was TRAINED with (default "
                         f"{QUANT_BITS_DEFAULT}; the trainer's --quant_bits). Used for the probe "
                         "and recorded as img_dw in the manifest -- ghrd_top.v's IMG_DW "
                         "and sim_card_cnn.py both follow it.")
    a = ap.parse_args()
    if a.fit_range is not None and not a.probe_dir:
        ap.error("--fit_range needs --probe_dir to measure the activations it is fitting")

    ckpt = Path(a.checkpoint)
    print(f"export_weights_rtl: {ckpt}  ->  Q{16 - a.frac_bits}.{a.frac_bits}  (frac_bits={a.frac_bits})")
    model = load_model(ckpt)

    scales, peaks_after = None, None
    if a.probe_dir:
        x, cols = load_probe_images(Path(a.probe_dir), a.probe_limit, a.quant_bits)
        peaks = measure_peaks(model, x, cols)
        print_peaks(peaks, f"activation headroom over {len(x)} images from {a.probe_dir} (as trained)")
        if a.fit_range is not None:
            scales = fit_range(model, peaks, a.fit_range)
            peaks_after = measure_peaks(model, x, cols)
            print("\n  --fit_range %.2f: layer scales " % a.fit_range
                  + ", ".join(f"{k} x{v:.4f}" for k, v in scales.items())
                  + " (each head shrunk as a whole: every argmax unchanged)")
            print_peaks(peaks_after, "activation headroom after fitting (this is what the RTL sees)")

    # quantise everything first; write nothing unless all of it is representable
    layers = layer_table(model.conv1_ch)
    exported, any_sat, warn, too_wide = [], False, [], []
    print(f"\n  {'file':<10} {'shape':<14} {'count':>7} {'max|w|':>8} {'max|q|':>7} {'sat':>4} {'zero%':>6} {'tiny%':>6} {'rmsErr%':>8} {'rom':>6}")
    for stem, attr, param, shape in layers:
        t = getattr(getattr(model, attr), param)
        assert tuple(t.shape) == shape, f"{stem}: checkpoint shape {tuple(t.shape)} != RTL shape {shape}"
        # fcs_w alone gets FCS_W_FRAC fractional bits -- see the constant
        q, s = quantise(t, FCS_W_FRAC if stem == "fcs_w" else a.frac_bits)
        exported.append((stem, q, s))
        any_sat |= s["saturated"] > 0
        if param == "weight" and (s["zeros_pct"] > 5 or s["tiny_pct"] > 15 or s["rms_err_pct"] > 5):
            warn.append(stem)

        # narrowed-ROM range gate -- see NARROW_ROM_BITS
        bits = NARROW_ROM_BITS.get(stem)
        if bits is None:
            rom = "16"
        else:
            limit = (1 << (bits - 1)) - 1
            rom = f"{bits}{'!' if s['max_abs_q'] > limit else ''}"
            if s["max_abs_q"] > limit:
                too_wide.append((stem, bits, s["max_abs_q"], limit))
        print(f"  {stem:<10} {str(shape):<14} {s['count']:>7} {s['max_abs']:8.4f} {s['max_abs_q']:>7} "
              f"{s['saturated']:>4} {s['zeros_pct']:6.2f} {s['tiny_pct']:6.2f} {s['rms_err_pct']:8.3f} {rom:>6}")

    if any_sat:
        print("\nFAIL: a layer exceeds the 16-bit range at this frac_bits. Nothing written.")
        print("      Q6.10 spans +/-32.0; the shipped model peaks at 0.58, so this is a pathological checkpoint.")
        sys.exit(1)
    if too_wide:
        print("\nFAIL: a narrowed ROM does not fit its Verilog port width. Nothing written.")
        for stem, bits, got, limit in too_wide:
            print(f"      {stem}.hex needs {got.bit_length() + 1} signed bits, "
                  f"but the RTL reads it {bits} bits wide (max |q| {limit}).")
        print("      The .hex file would be written correctly and then read back TRUNCATED by")
        print("      $readmemh into the narrow reg -- silently wrong weights, not a build error.")
        print("      Fix: lower --fit_range so the preceding layer's compensating scale shrinks")
        print("      these weights, or widen the port in conv_layer.v / fc_layer.v and re-fit.")
        sys.exit(1)
    if warn:
        print(f"\nWARN: coarse rounding on {', '.join(warn)} (zeros>5% / tiny>15% / rmsErr>5%). "
              f"Shipped baseline for fcs_w: zeros 1.8%, tiny 5.4%, rmsErr ~1.3%. Not fatal; watch the selftest.")

    ck_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    hps_md5 = None
    if a.hps_header:
        fcs_q = next(q for stem, q, _ in exported if stem == "fcs_w")
        hp = Path(a.hps_header)
        write_hps_header(fcs_q, hp)
        hps_md5 = hashlib.md5(hp.read_bytes()).hexdigest()
        print(f"\nwrote {hp}  ({fcs_q.size} fc_shared weights for the HPS to copy into DDR3)")
    else:
        print("\nWARN: no --hps_header -- the board reads fc_shared's weights from DDR3, copied "
              "there from fcs_w_ddr.h; without regenerating it the board runs OLD weights.")
    for out in a.out_dir:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"checkpoint": str(ckpt), "checkpoint_sha256": ck_sha, "frac_bits": a.frac_bits,
                    "fit_range": a.fit_range, "layer_scales": scales,
                    "fcs_w_frac": FCS_W_FRAC,   # card_cnn_core.v FCS_W_FRAC must match
                    "peak_activations_after_fit": peaks_after,
                    # the RTL must be built to match these or the ROMs are read
                    # at the wrong width, conv1 at the wrong channel count, or
                    # the image at the wrong depth. ghrd_top.v: CONV1_CH, IMG_DW.
                    "conv1_ch": model.conv1_ch,
                    "img_dw": img_dw_for(a.quant_bits),
                    "narrow_rom_bits": NARROW_ROM_BITS,
                    # fcs_w reaches the board through this header, not a ROM
                    "hps_header": ({"path": str(a.hps_header), "md5": hps_md5}
                                   if a.hps_header else None),
                    "written": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}}
        for stem, q, s in exported:
            p = out / f"{stem}.hex"
            write_hex(q, p)
            back = read_hex_like_sim(p)
            assert back.size == q.size and np.array_equal(back, q), f"{p}: round-trip through the sim parser failed"
            manifest["files"][p.name] = {"count": int(q.size), "md5": hashlib.md5(p.read_bytes()).hexdigest()}
        (out / "weights_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nwrote {len(exported)} files + weights_manifest.json to {out}/  (each re-read and verified)")
    print(f"\nghrd_top.v must say:  CONV1_CH = {model.conv1_ch};  IMG_DW = {img_dw_for(a.quant_bits)};"
          "   (sim_card_cnn.py reads both from the manifest)")


if __name__ == "__main__":
    main()
