"""Fine-tune fc_shared.weight -- the one layer the board keeps in DDR3 -- on
blurred cards (2 Oct, demo day).

The D8M's lens is never focused (the VCM rests at far focus): a real frame's
card edges rise over 2.3 px (Gaussian sigma ~0.9 px). At sigma 1.2-2.0 the
checkpoint-16 network reads clubs as spades -- the three lobes melt into a
spade-like blob -- while spades never turn into clubs. Everything except
fc_shared's weights is a ROM in the bitstream, so only those may change:

  * the conv trunk is frozen, so its features are computed once and cached;
  * fc_shared's bias and the three heads stay fixed (they are ROMs too);
  * the export reuses checkpoint 16's layer scales, so fcs_w alone changes and
    every other .hex stays byte-identical (checked against the manifest).

The training cards are the 54 deck scans pushed through the board's own path
(sim_card_cnn.fabric_luma -> locate.locate -> window_params -> the 7 VARIANTS
windows via window_map) at 50-95 px card widths, tilted, upside down, smooth or
pixelated, blurred (sigma 0.5-2.0), dim and noisy; mixed with deck_mix_384_final
so nothing already learned is lost.

  python finetune_fcs_blur.py check                 # ckpt16 re-exported == the deployed fcs_w.hex
  python finetune_fcs_blur.py cache  CACHE_DIR      # synthetic windows + deck_mix -> trunk features
  python finetune_fcs_blur.py train  CACHE_DIR RUN.pt [--lr --l2 --epochs --suit_weight]
  python finetune_fcs_blur.py deploy RUN.pt         # fcs_w.hex + manifest + M2/fcs_w_ddr.h, backups first
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.normpath(os.path.join(HERE, '..', '..'))                  # lab 5
sys.path.insert(0, HERE)
sys.path.insert(0, LAB)
from card_cnn_model import DualHeadCardCNN, POOLS, parse_class_name  # noqa: E402
import export_weights_rtl as X  # noqa: E402
import sim_card_cnn as S  # noqa: E402
import locate as L  # noqa: E402

CKPT = os.path.join(HERE, 'checkpoints_16', 'best_dualhead_model.pt')
DECK = os.path.join(HERE, 'image_data', 'deck_labeled')
MIX = os.path.join(HERE, 'image_data', 'deck_mix_384_final')
WEIGHTS = os.path.join(LAB, 'weights')
HPS = os.path.join(LAB, 'LAB_5', 'fpga_cnn', 'workspace_lab3', 'M2', 'fcs_w_ddr.h')
FW, FH = 512, 384
torch.set_num_threads(4)


def load_ckpt():
    model = DualHeadCardCNN.from_state_dict(torch.load(CKPT, map_location='cpu'))
    return model.eval()


def trunk(model, x):
    """conv1..conv4 as the model's forward, flattened: (n, 2304)."""
    with torch.no_grad():
        for conv, k in zip((model.conv1, model.conv2, model.conv3, model.conv4), POOLS):
            x = F.max_pool2d(F.relu(conv(x)), k)
        return torch.flatten(x, 1)


# ---- export with checkpoint 16's own scales -------------------------------

def export_q(W):
    """fc_shared.weight (float, the checkpoint's domain) -> the int16 Q1.14 the
    board stores: export_weights_rtl.fit_range's transform in its own float32
    order (feature columns / conv4's scale, then all * fc_shared's), then its
    quantise(). Returns ((64, 2305) int64, stats)."""
    man = json.load(open(os.path.join(WEIGHTS, 'weights_manifest.json')))
    s4, sfc = man['layer_scales']['conv4'], man['layer_scales']['fc_shared']
    W = W.detach().clone().float()
    W[:, :X.N_FEAT] /= s4
    W *= sfc
    q, stats = X.quantise(W, X.FCS_W_FRAC)
    return q.reshape(64, 2305), stats


def check():
    q, st = export_q(load_ckpt().fc_shared.weight)
    ok = np.array_equal(q, S.fsw)
    print(f'ckpt16 fc_shared re-exported with the manifest scales: {"IDENTICAL to" if ok else "DIFFERS from"} '
          f'weights/fcs_w.hex ({int((q != S.fsw).sum())} of {q.size} differ, saturated {st["saturated"]})')
    return ok


# ---- synthetic blurred cards through the board's path ---------------------

def scans():
    out = []
    for cls in sorted(os.listdir(DECK)):
        for f in sorted(glob.glob(os.path.join(DECK, cls, '*'))):
            im = cv2.imread(f)
            out.append((cls, cv2.resize(im, (400, round(400 * im.shape[0] / im.shape[1])),
                                        interpolation=cv2.INTER_AREA)))
    return out


def frame(card, rng):
    """One card on a mat in the board's 512x384 frame, as a blurred D8M sees it."""
    w = int(rng.uniform(50, 95))
    if rng.random() < 0.5:
        card = cv2.rotate(card, cv2.ROTATE_180)
    h = round(w * card.shape[0] / card.shape[1])
    if rng.random() < 0.5:                       # sharp at 4x, then every 4th pixel
        small = cv2.resize(cv2.resize(card, (4 * w, 4 * h), interpolation=cv2.INTER_AREA), (w, h),
                           interpolation=cv2.INTER_NEAREST)
    else:
        small = cv2.resize(card, (w, h), interpolation=cv2.INTER_AREA)
    mat = np.full((FH, FW, 3), rng.uniform(35, 75), np.float32) + rng.normal(0, 6, (FH, FW, 3))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-6, 6), 1.0)
    M[0, 2] += FW / 2 + rng.uniform(-120, 120) - w / 2
    M[1, 2] += FH / 2 + rng.uniform(-50, 50) - h / 2
    f = cv2.warpAffine(small.astype(np.float32), M, (FW, FH))
    m = cv2.warpAffine(np.ones((h, w), np.float32), M, (FW, FH))[..., None]
    f = f * m + mat * (1 - m)
    if rng.random() < 0.85:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 2.0))
    f = f * rng.uniform(0.4, 1.0) + rng.normal(0, rng.uniform(2, 6), f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def windows(gray, card):
    """The 7 VARIANTS windows the board reads -- sim_card_cnn._read_variants'
    window arithmetic, without the inference."""
    x0, y0, step, flags, clip = L.window_params(card, FW, FH)
    out = []
    for dx, dy, zoom, flip in S.VARIANTS:
        vx, vy, vs = x0, y0, step
        if zoom != 1000:
            vs = min(511, max(1, (step * zoom + 500) // 1000))
            span = ((S.DIM - 1) * vs) >> 8
            vx = (card.x0 + card.x1 - span + 1) // 2
            vy = (card.y0 + card.y1 - span + 1) // 2
        out.append(S.window_map(gray, vx + dx, vy + dy, vs, flags ^ flip, clip))
    return out


def synth(model, cards, seed, n_black, n_other, n_joker):
    rng = np.random.default_rng(seed)
    feats, labels, skipped = [], [], 0
    for cls, card in cards:
        lab = parse_class_name(cls)
        n = n_joker if lab[3] else n_black if cls[-1] in 'SC' else n_other
        done = tries = 0
        while done < n and tries < 4 * n:
            tries += 1
            g = S.fabric_luma(frame(card, rng))
            cs = L.locate(g.astype(np.uint8))
            if len(cs) != 1:
                skipped += 1
                continue
            x = np.stack(windows(g, cs[0])).astype(np.float32) / 255.0
            feats.append(trunk(model, torch.from_numpy(x)[:, None]).half())
            labels += [lab] * len(x)
            done += 1
    print(f'  synthetic seed {seed}: {len(labels)} windows ({skipped} frames skipped: not one card found)', flush=True)
    return torch.cat(feats), torch.tensor(labels)


def mix(model, split):
    from PIL import Image
    items = [(f, parse_class_name(cls)) for cls in sorted(os.listdir(os.path.join(MIX, split)))
             for f in sorted(glob.glob(os.path.join(MIX, split, cls, '*')))]
    feats = []
    for i in range(0, len(items), 32):
        x = np.stack([np.asarray(Image.open(f).convert('L').resize((384, 384)), np.float32) / 255.0
                      for f, _ in items[i:i + 32]])
        feats.append(trunk(model, torch.from_numpy(x)[:, None]).half())
        if i % 3200 == 0:
            print(f'  deck_mix {split}: {i}/{len(items)}', flush=True)
    return torch.cat(feats), torch.tensor([lab for _, lab in items])


def cache(out):
    os.makedirs(out, exist_ok=True)
    model, cards, t0 = load_ckpt(), scans(), time.time()
    for name, fn in (('synth_train', lambda: synth(model, cards, 1, 25, 15, 10)),
                     ('synth_hold', lambda: synth(model, cards, 2, 4, 4, 4)),
                     ('mix_val', lambda: mix(model, 'val')),
                     ('mix_train', lambda: mix(model, 'train'))):
        x, y = fn()
        torch.save({'x': x, 'y': y}, os.path.join(out, name + '.pt'))
        print(f'{name}: {tuple(x.shape)}  {time.time() - t0:.0f}s', flush=True)


# ---- fine-tune fc_shared.weight on the cached features --------------------

def heads(model):
    return [p.detach() for p in (model.fc_shared.bias, model.fc_rank.weight, model.fc_rank.bias,
                                 model.fc_suit.weight, model.fc_suit.bias, model.fc_joker.weight,
                                 model.fc_joker.bias)]


def forward(W, H, x, colour):
    b, Wr, br, Wu, bu, Wj, bj = H
    h = F.relu(torch.cat([x, colour[:, None]], 1) @ W.T + b)
    return h @ Wr.T + br, h @ Wu.T + bu, h @ Wj.T + bj


def score(W, H, data):
    """(cards right, cards) with the fabric's colour-constrained suit, plus the
    black cards' suit (right, total) -- what the clubs-as-spades fault is about."""
    x, y = data['x'].float(), data['y']
    with torch.no_grad():
        r, s, j = forward(W, H, x, y[:, 2].float())
    allowed = torch.zeros_like(s, dtype=torch.bool)
    allowed[y[:, 2] == 0, :2] = True
    allowed[y[:, 2] == 1, 2:] = True
    sp = s.masked_fill(~allowed, float('-inf')).argmax(1)
    jp = j.argmax(1)
    nj = y[:, 3] == 0
    right = torch.where(nj, (r.argmax(1) == y[:, 0]) & (sp == y[:, 1]) & (jp == 0), jp == 1)
    black = nj & (y[:, 2] == 0)
    return int(right.sum()), len(y), int((sp == y[:, 1])[black].sum()), int(black.sum())


def train(cache_dir, run, lr, l2, epochs, suit_weight, mix_per_epoch, seed):
    torch.manual_seed(seed)
    model = load_ckpt()
    W0, H = model.fc_shared.weight.detach().clone(), heads(model)
    d = {k: torch.load(os.path.join(cache_dir, k + '.pt')) for k in ('synth_train', 'synth_hold', 'mix_val', 'mix_train')}
    W = W0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=0.0)

    def report(tag):
        hv = score(W.detach(), H, d['mix_val']); hs = score(W.detach(), H, d['synth_hold'])
        print(f'{tag:>9}: deck_mix val {hv[0]}/{hv[1]}   blur hold-out {hs[0]}/{hs[1]} cards, '
              f'black suits {hs[2]}/{hs[3]}', flush=True)
        return hv, hs

    base = report('ckpt16')
    for ep in range(1, epochs + 1):
        pick = torch.randperm(len(d['mix_train']['y']))[:mix_per_epoch]
        xs = torch.cat([d['synth_train']['x'], d['mix_train']['x'][pick]])
        ys = torch.cat([d['synth_train']['y'], d['mix_train']['y'][pick]])
        order = torch.randperm(len(ys))
        for i in range(0, len(ys), 256):
            b = order[i:i + 256]
            x, y = xs[b].float(), ys[b]
            r, s, j = forward(W, H, x, y[:, 2].float())
            nj = y[:, 3] == 0
            loss = F.cross_entropy(j, y[:, 3])
            if nj.any():
                loss = loss + F.cross_entropy(r[nj], y[nj, 0]) + suit_weight * F.cross_entropy(s[nj], y[nj, 1])
            loss = loss + l2 * ((W - W0) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        last = report(f'epoch {ep}')
    q, st = export_q(W.detach())
    torch.save({'W': W.detach(), 'q': q, 'base': base, 'last': last,
                'args': dict(lr=lr, l2=l2, epochs=epochs, suit_weight=suit_weight)}, run)
    print(f'saved {run}: Q1.14 max|q| {st["max_abs_q"]}, saturated {st["saturated"]}, '
          f'max |W - W0| {float((W.detach() - W0).abs().max()):.4f}')


# ---- deploy -----------------------------------------------------------------

def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def deploy(run):
    q = torch.load(run, weights_only=False)['q']
    man_path = os.path.join(WEIGHTS, 'weights_manifest.json')
    man = json.load(open(man_path))
    for name, info in man['files'].items():             # the ROMs must be exactly what the .sof has
        if name != 'fcs_w.hex':
            assert md5(os.path.join(WEIGHTS, name)) == info['md5'], f'{name} changed -- not deploying'
    bak = WEIGHTS + '_ckpt16_bak'
    if not os.path.exists(bak):
        shutil.copytree(WEIGHTS, bak)
    if not os.path.exists(HPS + '.pre-blur.bak'):
        shutil.copy2(HPS, HPS + '.pre-blur.bak')
    X.write_hex(q.flatten(), Path(WEIGHTS) / 'fcs_w.hex')
    X.write_hps_header(q.flatten(), Path(HPS))
    assert np.array_equal(X.read_hex_like_sim(Path(WEIGHTS) / 'fcs_w.hex').reshape(64, 2305), q)
    man['files']['fcs_w.hex']['md5'] = md5(os.path.join(WEIGHTS, 'fcs_w.hex'))
    man['hps_header']['md5'] = md5(HPS)
    man['fcs_w_finetune'] = {'script': 'finetune_fcs_blur.py', 'run': os.path.basename(run),
                             'written': time.strftime('%Y-%m-%d %H:%M:%S'),
                             'note': 'fc_shared.weight only, blur-augmented; every other file is checkpoint 16'}
    with open(man_path, 'w', newline='\n') as f:
        json.dump(man, f, indent=2)
    print(f'deployed {run}: weights/fcs_w.hex + manifest, {HPS}; backups {bak}, fcs_w_ddr.h.pre-blur.bak')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('check', 'cache', 'train', 'deploy'))
    ap.add_argument('paths', nargs='*')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--l2', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--suit_weight', type=float, default=2.0)
    ap.add_argument('--mix_per_epoch', type=int, default=15000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.cmd == 'check':
        sys.exit(0 if check() else 1)
    if a.cmd == 'cache':
        cache(a.paths[0])
    elif a.cmd == 'train':
        train(a.paths[0], a.paths[1], a.lr, a.l2, a.epochs, a.suit_weight, a.mix_per_epoch, a.seed)
    else:
        deploy(a.paths[0])
