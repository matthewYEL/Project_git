#!/usr/bin/env python3
"""check_weights.py -- verify the .hex weights match the RTL topology.

$readmemh loads a wrong-sized or missing file without a single warning, and the
bitstream builds cleanly into a network of garbage. This parses the layer
instantiations out of card_cnn/card_cnn_core.v, works out how many values each
one should read, and compares against the files Quartus will actually pick up
from the project root.

    python check_weights.py

Exits non-zero on any mismatch. Run it before every Quartus compile.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(ROOT, "card_cnn", "card_cnn_core.v")


def params(text):
    """{name: int} from a `#(...)` parameter list, ignoring string params."""
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r"\.(\w+)\s*\(\s*(\d+)\s*\)", text)}


def files(text):
    w = re.search(r'\.WFILE\s*\(\s*"([^"]+)"', text)
    b = re.search(r'\.BFILE\s*\(\s*"([^"]+)"', text)
    return (w.group(1) if w else None), (b.group(1) if b else None)


def instantiations(src, kind):
    """Each `kind #( ... ) name (` block, as raw parameter text."""
    out = []
    for m in re.finditer(kind + r"\s*#\((.*?)\)\s*\n?\s*(\w+)\s*\(", src, re.S):
        out.append((m.group(2), m.group(1)))
    return out


def hex_values(path):
    """($readmemh-visible value count, list of complaints)."""
    n, bad = 0, []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            tok = line.split("//")[0].strip()
            if not tok:
                continue
            for word in tok.split():
                n += 1
                try:
                    v = int(word, 16)
                except ValueError:
                    bad.append("line %d: %r is not hex" % (lineno, word))
                    continue
                if len(word) != 4 or not 0 <= v <= 0xFFFF:
                    bad.append("line %d: %r is not a 4-digit 16-bit word"
                               % (lineno, word))
    return n, bad


def main():
    src = open(CORE).read()
    errors = []
    expected = {}   # filename -> (count, why)

    convs = instantiations(src, "conv_layer")
    fcs = instantiations(src, "fc_layer")
    if not convs or not fcs:
        sys.exit("could not parse layer instantiations from %s" % CORE)

    for name, body in convs:
        p = params(body)
        wf, bf = files(body)
        nw = p["IN_CH"] * p["OUT_CH"] * 25          # 5x5 kernels
        expected[wf] = (nw, "%s: %d in x %d out x 5x5" %
                        (name, p["IN_CH"], p["OUT_CH"]))
        expected[bf] = (p["OUT_CH"], "%s: one per output channel" % name)

    for name, body in fcs:
        p = params(body)
        wf, bf = files(body)
        expected[wf] = (p["N_IN"] * p["N_OUT"],
                        "%s: %d in x %d out" % (name, p["N_IN"], p["N_OUT"]))
        expected[bf] = (p["N_OUT"], "%s: one per output" % name)

    print("RTL topology from %s:" % os.path.relpath(CORE, ROOT))
    for name, body in convs:
        p = params(body)
        print("  conv  %-8s %2d -> %-2d  %dx%d -> %dx%d (pooled)"
              % (name, p["IN_CH"], p["OUT_CH"], p["DIM"], p["DIM"],
                 p["DIM"] // 2, p["DIM"] // 2))
    for name, body in fcs:
        p = params(body)
        print("  fc    %-8s %4d -> %d" % (name, p["N_IN"], p["N_OUT"]))
    print()

    for fname in sorted(expected):
        want, why = expected[fname]
        path = os.path.join(ROOT, fname)
        if not os.path.exists(path):
            errors.append("%s MISSING -- $readmemh will leave this layer at X/0"
                          % fname)
            continue
        got, bad = hex_values(path)
        status = "ok" if got == want else "MISMATCH"
        print("  %-14s %7d values, want %7d  %-8s  (%s)"
              % (fname, got, want, status, why))
        if got != want:
            errors.append("%s has %d values, RTL reads %d (%s)"
                          % (fname, got, want, why))
        errors.extend("%s %s" % (fname, b) for b in bad[:5])

    # Weight files in this model's naming family that nothing reads: the
    # stale-mix trap. Scoped to the current scheme on purpose -- the root also
    # holds conv_weight/conv_bias/fc_weight/fc_bias from the old lab CNN, which
    # are dead but not part of this network and not this script's business.
    family = re.compile(r"^(conv\d+|fcs|fcrank|fcsuit|fcjoker)_[wb]\.hex$")
    referenced = set(expected)
    for fname in sorted(os.listdir(ROOT)):
        if family.match(fname) and fname not in referenced:
            errors.append("%s is in the project root but no layer reads it -- "
                          "delete it so a stale mix cannot happen" % fname)

    print()
    if errors:
        print("FAIL (%d):" % len(errors))
        for e in errors:
            print("  - " + e)
        return 1
    print("PASS -- every layer's weights are present and correctly sized.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
