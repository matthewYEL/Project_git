"""Single source of truth for the card CNN's geometry, labels and architecture.

Everything that used to be copy-pasted into six files lives here now:
train_card_cnn_dualhead.py, auto_label_and_sort.py, recognize_card.py,
export_weights_hex.py, webcam_recognize.py and webcam_recognize_detect.py all
import from this module. That duplication is not hypothetical harm -- one copy
(export_weights_hex.py) kept a 2-way suit head long after the model went 4-way
and silently wrote a wrong-shaped fcsuit_w.hex, and webcam_recognize.py still
had the stale 2-way head when this module was written.

GEOMETRY. IMG_SIZE is 384, up from 96, because the CNN input is now sampled 1:1
from the camera's 640x480 frame instead of being read back out of the
2x2-decimated 320x240 display buffer. The old 96x96 window covered 192 x 384
camera pixels; the new one covers 384 x 384. Per unit of card that is 4x the
vertical sampling and 2x the horizontal, plus a 2x wider field of view.

POOLING. Four conv stages with pools (4, 2, 2, 2) -- product 32, so
384 -> 96 -> 48 -> 24 -> 12 and the flattened vector stays 16*12*12 = 2304.
fc_shared therefore stays 2305 -> 64 and its weight ROM does not grow; at 289
M10K blocks in 16-bit form it is already the largest single memory in the
design.

conv1's pool is 4 and not 2 because of a hard device limit, not a modelling
preference. conv1's pooled output at 8 channels costs:

    pool 2 -> 8 x 192 x 192 = 294,912 words x 16 bit = 576 M10K blocks
    pool 4 -> 8 x  96 x  96 =  73,728 words x 16 bit = 144 M10K blocks

The 5CSEBA6U23I7DK has 553 blocks in total, so pool 2 does not fit even if the
rest of the design were deleted. 96x96 feature maps are the ceiling this chip
can carry, and conv1 pool 4 is how you reach it.
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

IMG_SIZE = 384

# conv1 -> conv2 -> conv3 -> conv4. Product must divide IMG_SIZE down to 12.
POOLS = (4, 2, 2, 2)
POOL_TOTAL = POOLS[0] * POOLS[1] * POOLS[2] * POOLS[3]  # 32

# THE TWO M10K BUDGET LEVERS.
#   conv1 channels: 18 blocks each in conv1's pooled output (x16 RAM)
#   input bits:     8 -> the image RAM is 144 blocks (x10); 5 -> 72 (x5)
# With fc_shared's weights moved to HPS DDR3 (card_cnn_core.v; they were 145-289
# of the 553 blocks) the generous pair fits at ~465 blocks, so the defaults are
# 8 filters and full 8-bit grey levels. Before the DDR3 port they had to be
# 4 filters and 5 bits. Whatever is trained here, ghrd_top.v's CONV1_CH / IMG_DW
# must match it -- export_weights_rtl.py records both in weights_manifest.json.
CONV1_CH_DEFAULT = 8
QUANT_BITS_DEFAULT = 8   # 8 = full grey levels (IMG_DW 10 in the RTL)
CONV_CH = 16             # conv2/conv3/conv4 output channels
SHARED_OUT = 64

FEAT_DIM = CONV_CH * (IMG_SIZE // POOL_TOTAL) ** 2   # 16 * 12 * 12 = 2304

assert IMG_SIZE % POOL_TOTAL == 0, "IMG_SIZE must divide evenly by every pool"
assert FEAT_DIM == 2304, (
    f"FEAT_DIM is {FEAT_DIM}, not 2304 -- fc_shared and fcs_w.hex would change "
    "shape and the FPGA weight ROM would no longer fit"
)


# ---------------------------------------------------------------------------
# Input quantisation (M10K budget trim 5b)
# ---------------------------------------------------------------------------

class QuantizeInput:
    """Round the input to `bits` grey levels, the way the fabric stores it.

    The capture path computes an 8-bit luminance and writes the top `bits` of
    it into the image RAM. The CNN then reads that back as Q6.10. Storing 5
    bits instead of 10 halves the image RAM from 144 to 72 M10K blocks (M10K
    holds 1024 words at x8/x10 but 2048 at x4/x5), which is one of the two ways
    to make the 384x384 design fit.

    Apply AFTER ToTensor, so the input is [0,1]. bits=None is a no-op.
    """

    def __init__(self, bits=None):
        self.bits = bits

    def __call__(self, x):
        if self.bits is None or self.bits >= 8:
            return x
        drop = 2 ** (8 - self.bits)          # 8-bit gray >> drop
        levels = 2 ** self.bits
        return torch.floor(x * 255.0 / drop).clamp_(0, levels - 1) / levels

    def __repr__(self):
        return f"{type(self).__name__}(bits={self.bits})"


def img_dw_for(quant_bits):
    """The RTL's IMG_DW for a given input depth: 10 stores the full 8-bit gray
    (as gray<<2), anything under 8 stores exactly that many bits."""
    return 10 if quant_bits is None or quant_bits >= 8 else quant_bits


def get_transforms(train: bool, quant_bits=QUANT_BITS_DEFAULT):
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]
    if train:
        ops += [
            transforms.RandomRotation(8),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    ops += [
        transforms.ToTensor(),  # -> [0,1], shape (1, H, W)
    ]
    if quant_bits is not None:
        ops.append(QuantizeInput(quant_bits))
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

# Absolute 4-way suit encoding. This REPLACED an earlier scheme where the suit
# head was 2-way and its meaning depended on the colour flag ({S:0, C:1} when
# black, {H:0, D:1} when red).
#
# That scheme asked the head to learn a single binary decision where class 0
# meant "Spade OR Heart" and class 1 meant "Club OR Diamond" -- not coherent
# visual categories. The colour flag was supposed to disambiguate, but it
# reached the network as one scalar among 2305 inputs to fc_shared, so the
# model largely ignored it and settled on a default class. The symptom was a
# strictly one-directional error pattern: Heart->Diamond and Spade->Club (both
# 0->1), never the reverse.
#
# With an absolute encoding the head learns four distinct shapes, and the
# colour flag is applied at DECODE time to restrict the choice to the correct
# pair -- where it is decisive rather than advisory. card_cnn_core.v:191-234
# implements exactly that constraint in the argmax.
SUITS = ["S", "C", "H", "D"]
SUIT_TO_IDX = {s: i for i, s in enumerate(SUITS)}
BLACK_SUIT_IDX = [0, 1]   # S, C
RED_SUIT_IDX = [2, 3]     # H, D


def parse_class_name(name: str):
    """Parse a folder/class name into (rank_idx, suit_idx, color_flag, is_joker).

    suit_idx is ABSOLUTE (0=S, 1=C, 2=H, 3=D), not relative to colour.

    Jokers: rank_idx/suit_idx/color are placeholder 0s (unused -- masked out of
    the loss) with is_joker=1. Normal cards: real rank/suit/color, is_joker=0.

    ADJUST THIS REGEX if your dataset's class names don't match the
    '<rank><suit-letter>' pattern (e.g. '10H', 'AS', 'KD') or if Joker folders
    use a different name than 'JOKER'."""
    name = name.strip().upper()

    if "JOKER" in name:
        return 0, 0, 0, 1  # placeholder rank/suit/color, is_joker=1

    m = re.match(r"^(10|[2-9]|[JQKA])([SCHD])$", name)
    if not m:
        raise ValueError(
            f"Could not parse class name '{name}' -- update parse_class_name() regex"
        )

    rank_str, suit_letter = m.groups()
    rank_idx = RANKS.index(rank_str)
    suit_idx = SUIT_TO_IDX[suit_letter]
    color = 1 if suit_letter in ("H", "D") else 0
    return rank_idx, suit_idx, color, 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DualHeadCardCNN(nn.Module):
    """384x384 -> rank(13) / suit(4) / joker(2), sharing one 64-wide hidden layer.

    conv1_ch is a parameter because it is one of the M10K budget's two levers:
    conv1's pooled output is conv1_ch x 96 x 96, i.e. 18 M10K blocks per
    channel. It is stored in no checkpoint field -- infer it from
    conv1.weight.shape[0] when loading (see from_state_dict)."""

    def __init__(self, conv1_ch=CONV1_CH_DEFAULT):
        super().__init__()
        self.conv1_ch = conv1_ch

        # shared backbone
        self.conv1 = nn.Conv2d(1, conv1_ch, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(conv1_ch, CONV_CH, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(CONV_CH, CONV_CH, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(CONV_CH, CONV_CH, kernel_size=5, padding=2)

        # +1 input for the color flag, concatenated after flattening
        self.fc_shared = nn.Linear(FEAT_DIM + 1, SHARED_OUT)

        self.fc_rank = nn.Linear(SHARED_OUT, len(RANKS))   # 13-way
        self.fc_suit = nn.Linear(SHARED_OUT, len(SUITS))   # 4-way: S, C, H, D
        self.fc_joker = nn.Linear(SHARED_OUT, 2)           # not-joker / joker

    def forward(self, x, color_flag):
        # color_flag: shape (batch,) float tensor of 0.0/1.0
        x = F.max_pool2d(F.relu(self.conv1(x)), POOLS[0])   # 384 -> 96
        x = F.max_pool2d(F.relu(self.conv2(x)), POOLS[1])   #  96 -> 48
        x = F.max_pool2d(F.relu(self.conv3(x)), POOLS[2])   #  48 -> 24
        x = F.max_pool2d(F.relu(self.conv4(x)), POOLS[3])   #  24 -> 12
        x = torch.flatten(x, 1)
        x = torch.cat([x, color_flag.unsqueeze(1)], dim=1)
        shared = F.relu(self.fc_shared(x))
        rank_logits = self.fc_rank(shared)
        suit_logits = self.fc_suit(shared)
        joker_logits = self.fc_joker(shared)
        return rank_logits, suit_logits, joker_logits

    @classmethod
    def from_state_dict(cls, state, strict=True):
        """Build a model whose conv1 width matches the checkpoint, then load it.

        conv1_ch is a build-time choice that varies between experiments, so
        constructing the default and calling load_state_dict blind fails with a
        shape error that reads like corruption. Infer it instead."""
        conv1_ch = state["conv1.weight"].shape[0]
        model = cls(conv1_ch=conv1_ch)
        model.load_state_dict(state, strict=strict)
        return model


if __name__ == "__main__":
    # Runnable self-check: shapes must survive the whole chain, and the
    # flattened vector must still be the 2304 the FPGA's fcs_w.hex expects.
    for ch in (4, 8):
        m = DualHeadCardCNN(conv1_ch=ch).eval()
        x = torch.zeros(2, 1, IMG_SIZE, IMG_SIZE)
        colour = torch.tensor([0.0, 1.0])
        with torch.no_grad():
            r, s, j = m(x, colour)
        assert r.shape == (2, 13), r.shape
        assert s.shape == (2, 4), s.shape
        assert j.shape == (2, 2), j.shape
        assert m.fc_shared.weight.shape == (64, 2305), m.fc_shared.weight.shape
        assert m.conv2.weight.shape == (16, ch, 5, 5), m.conv2.weight.shape

        # round-trip through a state_dict the way the exporter loads one
        again = DualHeadCardCNN.from_state_dict(m.state_dict())
        assert again.conv1_ch == ch

    # quantiser: 5-bit must collapse 256 grey levels to 32 and stay in [0,1)
    q = QuantizeInput(5)
    vals = q(torch.arange(256).float().div(255.0))
    assert vals.unique().numel() == 32, vals.unique().numel()
    assert float(vals.max()) < 1.0 and float(vals.min()) == 0.0
    assert QuantizeInput(None)(vals).equal(vals)

    print(f"OK  IMG_SIZE={IMG_SIZE}  FEAT_DIM={FEAT_DIM}  pools={POOLS}")
