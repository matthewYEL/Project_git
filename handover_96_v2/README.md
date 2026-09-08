# Handover — what to copy where

## quartus_project/ → the folder holding your .qpf

Six Verilog files. Overwrite the existing ones.

| file | what changed |
|---|---|
| `card_cnn_avalon.v` | **trailing comma fixed** after removing `colour_flag_hw` and `irq`; **`done` now latched**; **ID register added at +0x18** |
| `card_cnn_core.v` | RAMs right-sized (`DEPTH` parameter) — saves 55 M10K blocks |
| `conv_layer.v` | 2-stage pipeline; round-to-nearest; `FRAC_BITS` parameter |
| `fc_layer.v` | 2-stage pipeline; **write address zero-extends** instead of part-selecting |
| `maxpool_layer.v` | 2-stage pipeline; `ram_dp` gained `DEPTH` |
| `ghrd_top.v` | camera removed, orphaned PIO inputs tied off, **no conduit line** |

## The twelve .hex weights — INCLUDED, in quartus_project/

From `checkpoints_13`. They sit alongside the `.v` files and go to the same
place: the **project root**, beside the `.qpf`, not a subfolder. `$readmemh`
resolves relative to where Quartus runs, and missing files produce a bitstream
that builds cleanly and outputs garbage with no error.

Delete any older `.hex` files from the project first so a stale mix cannot
happen.

Line counts, if you want to check nothing truncated in transit:

| file | lines | why |
|---|---:|---|
| `conv1_w.hex` | 200 | 8 filters x 1 channel x 5x5 |
| `conv1_b.hex` | 8 | |
| `conv2_w.hex` | 3,200 | 16 x 8 x 5x5 |
| `conv2_b.hex` | 16 | |
| `fcs_w.hex` | 147,520 | 64 x 2305 (2304 features + colour flag) |
| `fcs_b.hex` | 64 | |
| `fcrank_w.hex` | 832 | 13 x 64 |
| `fcrank_b.hex` | 13 | |
| `fcsuit_w.hex` | 256 | 4 x 64 |
| `fcsuit_b.hex` | 4 | |
| `fcjoker_w.hex` | 128 | 2 x 64 |
| `fcjoker_b.hex` | 2 | |

Format is Q6.10 — `stored = round(real * 1024)` — chosen by measurement, not
convention. Q4.12 saturated `fc_shared` badly enough to cost 6.6% accuracy.
`FRAC_BITS` in `conv_layer.v` and `fc_layer.v` must stay at 10 to match.

## model/ — the trained checkpoint

`best_dualhead_model.pt` from `checkpoints_13`, plus the per-tensor
quantisation scales. Not needed to build or run the hardware (the `.hex` files
already carry the weights) — included so the exact model behind this build is
recoverable, and so `recognize_card.py` can be run against it on a laptop to
cross-check hardware results.

Architecture, which the RTL depends on:

```
input 48x48 grayscale (rank/suit index corner, 1:2 crop squashed square)
  conv1      8 filters 5x5 pad 2 -> ReLU -> maxpool 2x2
  conv2     16 filters 5x5 pad 2 -> ReLU -> maxpool 2x2
  flatten   16*12*12 = 2304  +  1 colour flag  =  2305
  fc_shared 2305 -> 64 ReLU
  fc_rank 64->13 | fc_suit 64->4 | fc_joker 64->2
decode: rank = argmax; suit = argmax restricted to {S,C} or {H,D} by colour
```

## Alignment with downsample_96x96.v

The teammate's hardware downsampler and this model are only PARTLY aligned.

**Aligned:** the crop aspect. The hardware crops 96 columns by 192 rows (1:2)
and squashes it square; `checkpoints_13` was trained on 0.21 x 0.30 corner
crops, also 1:2. That was the whole point of the retrain.

**Three gaps, all handled on the host by `board/snapshot_to_accel.c`:**

| | hardware gives | accelerator wants |
|---|---|---|
| size | 9,216 cell sums (96x96) | 2,304 pixels (48x48) |
| scale | raw sums, 0..510 per cell | Q6.10, white = 1024 |
| colour | `colour_hw` from a red-pixel count vs `RED_THRESH` | a flag on CONTROL bit 2 |

A 96x96 model does not fit: the flatten becomes 9,216, so `fc_shared` needs
1,153 M10K blocks against 553. It would need a third conv+pool stage
(96 -> 48 -> 24 -> 12) and inference would go from ~50 ms to ~258 ms. Worth
doing eventually; averaging 2x2 in software costs nothing now and is still
better than the path it replaces, which stretched 29 real columns to 48.

**`RED_THRESH` needs its own calibration.** It counts pixels exceeding a
redness threshold; the software path takes the 98th percentile of a
white-balanced chromaticity ratio. Different statistics — the software's value
of 99 does NOT transfer. Use `snapshot_red_count()` on several known red and
known black cards and set `RED_THRESH` to the midpoint of the gap.

**Do not invert grayscale.** `downsample_28x28.v` computes `(255*3)-(r+g+b)`
because MNIST is white-on-black. This model trained on white cards with dark
ink. Inverting feeds it the negative of everything it learned, and it fails
completely while looking like a wiring fault.

## python/ — inference scripts, run on a laptop

The three scripts that load the checkpoint. All are self-contained (each
duplicates the model definition rather than importing it), so **a change to
the architecture means editing all three plus the two under `fpga/scripts/`**.

| file | use |
|---|---|
| `recognize_card.py` | single image, batch, or labelled eval with confusion matrices |
| `auto_label_and_sort.py` | semi-automated labelling of new photos |
| `webcam_recognize_detect.py` | live camera demo, card detection + majority vote |

Settings that must agree with the hardware, and currently do:

- `IMG_SIZE = 48`
- colour threshold **99** — from calibrating on augmented data, where black
  tops at 82.9 and red starts at 115.9. Calibrating on originals gives a
  different gap (82.6 to 155.5) and a threshold that costs two Diamonds,
  because rotation and brightness jitter pull marginal red cards down.
  Calibrate on the same distribution you evaluate on.
- corner crop `0.21 x 0.30` — a 1:2 aspect, matching the 96-column by 192-row
  hardware crop in `downsample_96x96.v`
- 4-way absolute suit encoding {S,C,H,D}, with the colour flag restricting the
  argmax to the matching pair at decode time

Not included, but in your tree: `build_warped_dataset.py`,
`train_card_cnn_dualhead.py`, `augment_dataset.py`,
`split_augmented_train_val.py`, `combine_datasets.py`,
`calibrate_color_threshold.py`.

## board/ → compile in WSL, scp to the board

```bash
arm-linux-gnueabihf-gcc -O2 -static -o probe_cnn   probe_cnn.c
arm-linux-gnueabihf-gcc -O2 -static -o card_cnn    card_cnn.c
arm-linux-gnueabihf-gcc -O2 -static -o bridge_test bridge_test.c
scp probe_cnn card_cnn bridge_test root@192.168.1.30:~/
```

`-static` matters: WSL's glibc is newer than the board's, and a dynamically
linked binary fails with a GLIBC version error.

- `card_cnn.c` — `CNN_BASE 0x20000`, `IMAGE_BASE` **corrected to 0x4000**
  (the RTL selects the image window with `avs_address[12]`, which is word
  0x1000 = byte 0x4000, not 0x2000)
- `probe_cnn.c` — reads all seven registers, checks the ID magic, separates
  `busy` from `done`
- `bridge_test.c` — LED walk, proves the bridge works

---

## Order of operations

**1. Delete the duplicate component description.** This is the one that was
breaking everything: two `_hw.tcl` files both declaring a module named
`card_cnn_avalon`, so Platform Designer picked one arbitrarily and the stale
one didn't match the current Verilog.

```powershell
Remove-Item .\new_component_hw.tcl
```

**2. Copy in the six `.v` files and the twelve `.hex` files.**

**3. Re-do the Component Editor** on `card_cnn_avalon.v`. The saved
`card_cnn_avalon_hw.tcl` was analysed against a version that still had
`colour_flag_hw` and `irq`, so it no longer matches.

- Files tab → `card_cnn_avalon.v`, top-level, **Analyze Synthesis Files**
- Signals tab → seven signals only:

| port | interface | type |
|---|---|---|
| `clk` | clock | `clk` |
| `reset_n` | reset | `reset_n` |
| `avs_address` | avalon_slave_0 | `address` |
| `avs_read` | avalon_slave_0 | `read` |
| `avs_readdata` | avalon_slave_0 | `readdata` |
| `avs_write` | avalon_slave_0 | `write` |
| `avs_writedata` | avalon_slave_0 | `writedata` |

- Interfaces tab → associated clock `clock`, reset `reset`, **address units
  WORDS**, **Read Wait 1**
- **Finish** — and confirm the file was actually written:

```powershell
Get-ChildItem -Recurse -Filter "*_hw.tcl" | Select-Object FullName
```

Exactly one `card_cnn_avalon_hw.tcl`. If `new_component_hw.tcl` is back,
delete it again.

**4. Platform Designer:** remove `card_cnn_avalon_0`, re-add from the IP
Catalog, reconnect slave → `hps_0.h2f_lw_axi_master` plus clock and reset,
base `0x20000`, lock it, Generate HDL.

**5. Compile and program.** Program the `.sof` **after** Linux has finished
booting — the SD card's `soc_system.rbf` is loaded by U-Boot at every boot and
overwrites anything programmed earlier.

**6. Test, in this order:**

```
./bridge_test 0x10040     # LEDs walk = your bitstream is live (0x3000 = factory)
./probe_cnn 0x20000       # +0x18 must return 0xCA5D0001
./card_cnn scan_img_052_tl.pgm
```

`0xCA5D0001` is the gate. It returns a constant regardless of what the core is
doing, so if it comes back wrong the read path is broken and every other
register value is meaningless.

---

## Where things stood

Verified in simulation against bit-exact reference vectors: conv1 matching all
18,432 outputs, full chain correct on both colour paths.

Three RTL bugs were found that way — a pipeline one cycle short, a head-select
mux driven by pulses instead of latched flags, and a width-mismatched write
address that silently discarded the suit and joker heads while rank kept
working perfectly.

A fourth only appeared on hardware and **could not** have been found in
simulation: `done` was a one-cycle pulse, invisible to a driver polling at
millisecond intervals. A testbench polls every cycle and always catches it.

Model is `checkpoints_13`, trained from scratch at the 1:2 crop aspect that
`downsample_96x96.v` produces, 60 epochs with label smoothing 0.05.

The colour threshold must be set from that run's calibration step — take the
MIDPOINT of the black-max / red-min gap, not the number the script prints (it
reports the lowest cutoff that works, which leaves no margin beneath it). It
goes in three places: `recognize_card.py`, `auto_label_and_sort.py` and
`webcam_recognize_detect.py`.

Worth recording: calibrating on originals while evaluating on augmented copies
cost two Diamonds at one point. Rotation and brightness jitter pull marginal
red cards down, so calibration has to run on the same distribution as the
evaluation — hence `--include_augmented`.

Resource use ~428 of 553 M10K blocks with the camera removed. The camera and
accelerator do not both fit — the frame buffer alone is ~410 blocks, and it
exists for the HDMI passthrough rather than for the CNN. Moving it to DDR3, or
dropping the passthrough in favour of a generated dashboard, is the way back.
