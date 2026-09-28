"""
train_card_cnn_dualhead.py

Self-contained dual-head + Joker-aware version of the card CNN.

  - Shared conv backbone operates on the grayscale corner-crop, 384x384.
    Geometry and the model class live in card_cnn_model.py -- see its header
    for why conv1 pools by 4 and everything after it by 2.
  - A "color flag" bit (red=1/black=0 -- computed cheaply in FPGA fabric
    from raw RGB, or from ground-truth suit at training time) is
    concatenated into the feature vector.
  - THREE heads branch off the shared features:
      rank head    -> 13 classes (2,3,4,5,6,7,8,9,10,J,Q,K,A)
      suit head    -> 4 classes, absolute: 0=Spade, 1=Club, 2=Heart, 3=Diamond.
                      The colour flag restricts the argmax to the right pair at
                      DECODE time rather than being one input among 2305.
      joker head   -> 2 classes: 0=not Joker, 1=Joker

  Jokers have no meaningful rank/suit, so their rank/suit loss is MASKED
  OUT during training -- only the joker head learns from Joker samples.
  Normal cards train all three heads (their joker-head target is simply 0).

Dataset: an ImageFolder-style directory, one folder per class (e.g. "10H",
"AS", "KD", "JOKER"). Labels are derived by PARSING THE FOLDER NAME --
check parse_class_name() and adjust the regex if your dataset's naming
convention differs.

Usage:
  python train_card_cnn_dualhead.py --data_dir ./data --epochs 30 --out_dir ./checkpoints
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image

# Geometry, labels, transforms and the model all live in card_cnn_model.py so
# that the six scripts which used to carry their own copy cannot drift apart
# again. Re-exported here because callers (export_weights_rtl.py, sim_card_cnn.py)
# have historically imported them from this module.
from card_cnn_model import (  # noqa: F401
    IMG_SIZE, POOLS, FEAT_DIM, CONV1_CH_DEFAULT, QUANT_BITS_DEFAULT,
    RANKS, SUITS, SUIT_TO_IDX, BLACK_SUIT_IDX, RED_SUIT_IDX,
    parse_class_name, get_transforms, QuantizeInput, DualHeadCardCNN,
)


class DualHeadCardDataset(Dataset):
    """Wraps an ImageFolder-style directory, deriving (rank, suit_in_color,
    color, is_joker) labels per image instead of a single flat class index."""

    def __init__(self, root, transform):
        base = datasets.ImageFolder(root)
        self.transform = transform
        self.samples = []
        for path, class_idx in base.samples:
            class_name = base.classes[class_idx]
            parsed = parse_class_name(class_name)
            self.samples.append((path, *parsed))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, rank_idx, suit_idx, color, is_joker = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, rank_idx, suit_idx, color, is_joker


def collate_with_color(batch):
    imgs, ranks, suits, colors, jokers = zip(*batch)
    imgs = torch.stack(imgs)
    ranks = torch.tensor(ranks, dtype=torch.long)
    suits = torch.tensor(suits, dtype=torch.long)
    colors = torch.tensor(colors, dtype=torch.float32)
    jokers = torch.tensor(jokers, dtype=torch.long)
    return imgs, ranks, suits, colors, jokers


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, device, epochs, lr, out_dir,
          rank_weight=1.0, suit_weight=1.0, joker_weight=1.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, ranks, suits, colors, jokers in train_loader:
            imgs, ranks, suits, colors, jokers = (
                imgs.to(device), ranks.to(device), suits.to(device),
                colors.to(device), jokers.to(device)
            )
            optimizer.zero_grad()
            rank_logits, suit_logits, joker_logits = model(imgs, colors)

            joker_loss = criterion(joker_logits, jokers)

            non_joker_mask = jokers == 0
            if non_joker_mask.any():
                rank_loss = criterion(rank_logits[non_joker_mask], ranks[non_joker_mask])
                suit_loss = criterion(suit_logits[non_joker_mask], suits[non_joker_mask])
            else:
                rank_loss = torch.tensor(0.0, device=device)
                suit_loss = torch.tensor(0.0, device=device)

            loss = joker_weight * joker_loss + rank_weight * rank_loss + suit_weight * suit_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        rank_acc, suit_acc, full_acc, joker_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch+1:3d}/{epochs}  loss={train_loss:.4f}  "
              f"rank_acc={rank_acc:.4f}  suit_acc={suit_acc:.4f}  "
              f"full_card_acc={full_acc:.4f}  joker_acc={joker_acc:.4f}")

        # combine full-card accuracy and joker accuracy for checkpoint selection
        combined = (full_acc + joker_acc) / 2
        if combined > best_acc:
            best_acc = combined
            torch.save(model.state_dict(), Path(out_dir) / "best_dualhead_model.pt")

    # The strict '>' above keeps the FIRST epoch to reach the best val score. When
    # val is augmented copies of the training photos it saturates within a few
    # epochs, so "best" is an early, barely-fitted model and everything the
    # cosine schedule did afterwards is discarded. Save the final epoch too, so
    # the fully annealed weights are available to export for the FPGA.
    torch.save(model.state_dict(), Path(out_dir) / "last_dualhead_model.pt")

    print(f"best combined val score: {best_acc:.4f}")
    return best_acc


@torch.no_grad()
def constrain_suit_by_color(suit_logits, colors):
    """Restricts the 4-way suit prediction to the pair matching the detected
    colour: {S,C} when black, {H,D} when red.

    This is how inference decodes the head, so evaluation must score it the
    same way -- an unconstrained argmax would report a number that the real
    pipeline never produces."""
    is_red = (colors >= 0.5).unsqueeze(1)
    allowed = torch.zeros_like(suit_logits, dtype=torch.bool)
    allowed[:, BLACK_SUIT_IDX] = ~is_red
    allowed[:, RED_SUIT_IDX] = is_red
    return suit_logits.masked_fill(~allowed, float("-inf")).argmax(dim=1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rank_correct, suit_correct, suit_raw_correct, full_correct, joker_correct = 0, 0, 0, 0, 0
    non_joker_total, total = 0, 0

    for imgs, ranks, suits, colors, jokers in loader:
        imgs, ranks, suits, colors, jokers = (
            imgs.to(device), ranks.to(device), suits.to(device),
            colors.to(device), jokers.to(device)
        )
        rank_logits, suit_logits, joker_logits = model(imgs, colors)
        rank_pred = rank_logits.argmax(dim=1)
        suit_pred_raw = suit_logits.argmax(dim=1)                    # head on its own
        suit_pred = constrain_suit_by_color(suit_logits, colors)     # as inference decodes it
        joker_pred = joker_logits.argmax(dim=1)

        joker_correct += (joker_pred == jokers).sum().item()
        total += jokers.size(0)

        non_joker_mask = jokers == 0
        if non_joker_mask.any():
            rank_correct += (rank_pred[non_joker_mask] == ranks[non_joker_mask]).sum().item()
            suit_raw_correct += (suit_pred_raw[non_joker_mask] == suits[non_joker_mask]).sum().item()
            suit_correct += (suit_pred[non_joker_mask] == suits[non_joker_mask]).sum().item()
            full_correct += ((rank_pred[non_joker_mask] == ranks[non_joker_mask]) &
                              (suit_pred[non_joker_mask] == suits[non_joker_mask])).sum().item()
            non_joker_total += non_joker_mask.sum().item()

    rank_acc = rank_correct / non_joker_total if non_joker_total else 0.0
    suit_acc = suit_correct / non_joker_total if non_joker_total else 0.0
    full_acc = full_correct / non_joker_total if non_joker_total else 0.0
    joker_acc = joker_correct / total if total else 0.0
    return rank_acc, suit_acc, full_acc, joker_acc


# ---------------------------------------------------------------------------
# INT8 export for FPGA deployment
# ---------------------------------------------------------------------------

def quantize_tensor(w: torch.Tensor):
    max_val = w.abs().max().item()
    scale = max_val / 127.0 if max_val > 0 else 1.0
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale


def export_int8_header(model: DualHeadCardCNN, out_path: str):
    """Writes an INT8 C header with weights, biases and scales for each
    layer, including all three heads."""
    layers = {
        "conv1_w": model.conv1.weight.data,
        "conv1_b": model.conv1.bias.data,
        "conv2_w": model.conv2.weight.data,
        "conv2_b": model.conv2.bias.data,
        "conv3_w": model.conv3.weight.data,
        "conv3_b": model.conv3.bias.data,
        "conv4_w": model.conv4.weight.data,
        "conv4_b": model.conv4.bias.data,
        "fc_shared_w": model.fc_shared.weight.data,
        "fc_shared_b": model.fc_shared.bias.data,
        "fc_rank_w": model.fc_rank.weight.data,
        "fc_rank_b": model.fc_rank.bias.data,
        "fc_suit_w": model.fc_suit.weight.data,
        "fc_suit_b": model.fc_suit.bias.data,
        "fc_joker_w": model.fc_joker.weight.data,
        "fc_joker_b": model.fc_joker.bias.data,
    }

    lines = ["// Auto-generated INT8 weights for DualHeadCardCNN (with Joker head)",
             "#ifndef CARD_CNN_DUALHEAD_WEIGHTS_H",
             "#define CARD_CNN_DUALHEAD_WEIGHTS_H", ""]
    lines.append(f"#define NUM_RANKS {len(RANKS)}")
    lines.append(f"#define NUM_SUIT_OPTIONS {len(SUITS)}  // absolute: S, C, H, D")
    lines.append("#define NUM_JOKER_OPTIONS 2  // 0=not joker, 1=joker")
    lines.append("")

    scales = {}
    for name, tensor in layers.items():
        q, scale = quantize_tensor(tensor)
        scales[name] = scale
        flat = q.flatten().tolist()
        lines.append(f"// shape: {list(tensor.shape)}, scale: {scale:.8f}")
        lines.append(f"static const signed char {name}[{len(flat)}] = {{")
        lines.append(", ".join(str(v) for v in flat))
        lines.append("};")
        lines.append("")

    lines.append("#endif // CARD_CNN_DUALHEAD_WEIGHTS_H")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    with open(out_path.replace(".h", "_scales.json"), "w") as f:
        json.dump(scales, f, indent=2)

    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                         help="warm-start from an existing checkpoint instead of random init -- "
                              "use this for fast targeted fine-tuning on corrected/new data "
                              "rather than a full retrain from scratch")
    parser.add_argument("--rank_weight", type=float, default=1.0,
                         help="loss weight for the rank head (default 1.0)")
    parser.add_argument("--suit_weight", type=float, default=1.0,
                         help="loss weight for the suit head -- raise this if suit accuracy "
                              "lags behind rank accuracy (default 1.0)")
    parser.add_argument("--joker_weight", type=float, default=1.0,
                         help="loss weight for the joker head (default 1.0)")
    # The two M10K budget levers. At 384x384 the design does not fit with both
    # at their generous setting, so the defaults are the cheapest pair; train
    # a more generous one only if the selftest says accuracy needs it. See
    # card_cnn_model.py. ghrd_top.v's CONV1_CH / IMG_DW must match whatever is
    # trained -- export_weights_rtl.py records both in weights_manifest.json.
    parser.add_argument("--conv1_ch", type=int, default=CONV1_CH_DEFAULT,
                         help=f"conv1 output channels (default {CONV1_CH_DEFAULT}). Each "
                              "channel costs 18 M10K blocks in conv1's pooled output.")
    parser.add_argument("--quant_bits", type=int, default=QUANT_BITS_DEFAULT,
                         help=f"input grey-level bits (default {QUANT_BITS_DEFAULT}), matching "
                              "what the fabric stores in the image RAM: 5 costs 72 M10K "
                              "blocks, 8 (full grey levels, IMG_DW 10) costs 144.")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    print(f"IMG_SIZE={IMG_SIZE}  pools={POOLS}  feat_dim={FEAT_DIM}  "
          f"conv1_ch={args.conv1_ch}  quant_bits={args.quant_bits or 8}")

    train_ds = DualHeadCardDataset(Path(args.data_dir) / "train",
                                    transform=get_transforms(train=True, quant_bits=args.quant_bits))
    val_ds = DualHeadCardDataset(Path(args.data_dir) / "val",
                                  transform=get_transforms(train=False, quant_bits=args.quant_bits))
    n_jokers_train = sum(1 for s in train_ds.samples if s[4] == 1)
    n_jokers_val = sum(1 for s in val_ds.samples if s[4] == 1)
    print(f"train samples: {len(train_ds)} ({n_jokers_train} jokers)  "
          f"val samples: {len(val_ds)} ({n_jokers_val} jokers)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, collate_fn=collate_with_color)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, collate_fn=collate_with_color)

    model = DualHeadCardCNN(conv1_ch=args.conv1_ch).to(device)
    if args.init_checkpoint:
        print(f"warm-starting from {args.init_checkpoint}")
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        model_sd = model.state_dict()

        # Skip any layer whose shape no longer matches. fc_suit changed from
        # 2-way (colour-relative) to 4-way (absolute S/C/H/D), so its weights
        # can't carry over -- but the conv backbone and every other head can,
        # which is most of the value of warm starting.
        compatible = {k: v for k, v in ckpt.items()
                      if k in model_sd and v.shape == model_sd[k].shape}
        skipped = [k for k in ckpt if k not in compatible]
        model_sd.update(compatible)
        model.load_state_dict(model_sd)

        if skipped:
            print(f"  reinitialised from scratch (shape changed): {', '.join(skipped)}")
        print(f"  loaded {len(compatible)}/{len(ckpt)} tensors from the checkpoint")
        print("NOTE: for fine-tuning, consider a lower --lr (e.g. 1e-4) than the default 1e-3, "
              "so you nudge existing weights rather than overwrite what they've already learned")
    train(model, train_loader, val_loader, device, args.epochs, args.lr, args.out_dir,
          rank_weight=args.rank_weight, suit_weight=args.suit_weight, joker_weight=args.joker_weight)

    # reload best checkpoint before export
    model.load_state_dict(torch.load(Path(args.out_dir) / "best_dualhead_model.pt"))
    export_int8_header(model, str(Path(args.out_dir) / "card_cnn_dualhead_weights.h"))


if __name__ == "__main__":
    main()