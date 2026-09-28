# Card CNN Bench Handbook

DE10-Nano (5CSEBA6U23I7DK) · Quartus 25.1std.0 Lite · top `ghrd_top`
Bitstream built 2026-09-06 04:16, fitter successful.

Web version: https://claude.ai/code/artifact/5d80e075-81ac-45fe-ae69-eff863c78e6e

A playing card goes in front of the D8M camera; a three-head CNN in Cyclone V
fabric reads rank, suit and joker; a bare-metal Cortex-A9 app prints the answer.

---

## 1. What the system does

```
D8M sensor  ->  frame buffer  ->  crop+downsample  ->  CNN fabric  ->  HPS app
  640x480        320x240            48x48             3 heads         printf
raw Bayer      2x2 Bayer skip    centre 240x240    rank13/suit4/    PIO poll,
               ON_CHIP_FRAM.v    5x5 luma cells    joker2, Q6.10    App Console
```

The green box on HDMI is exactly the 240x240 the network sees
(screen x 200..439, y 120..359).

No Linux, no SD card image. Weights are baked into M10K via `$readmemh`, so the
`.sof` is self-contained.

### Why the display got worse and the network got better

The frame buffer was cut 640x480 -> 320x240 to free ~300 M10K blocks, which paid
for the bigger conv layers. HDMI preview is therefore half resolution, centred in
a 640x480 frame with black borders. What the network sees went the other way:

| Stage            | Old (28x28 lab CNN) | Now (48x48 card CNN)              |
|------------------|---------------------|-----------------------------------|
| Sensor output    | 640x480             | 640x480 — unchanged               |
| Frame buffer     | 640x480             | 320x240                           |
| Region sampled   | 476x476 buffer px   | 240x240 buffer px = 480x480 sensor|
| Cell size        | 17x17               | 5x5                               |
| Network input    | 28x28               | 48x48 — 2.9x the pixels           |
| Sensor : network | 17 : 1              | 10 : 1                            |

Field of view barely moved. The real detail gain came from framing (§3): whole
card across the crop was ~1.9 mm/cell, the index corner is ~0.33 mm/cell.

---

## 2. Run the prebuilt bitstream

Needs no Quartus, no Python, no weights.

**You need:** DE10-Nano + D8M on GPIO0, HDMI display, USB-Blaster + Quartus
Programmer, Arm DS with Intel SoC EDS started from the Embedded Command Shell
(so `SOCEDS_DEST_ROOT` is set).

1. **Program the FPGA.** Quartus Programmer → Auto Detect → 5CSEBA6 →
   `DE10-Nano.sof` → Start. Repeat after every power cycle; the `.sof` is volatile.
2. **Close Quartus.** It and Arm DS contend for the USB-Blaster.
3. **Import the project.** Arm DS → File → Import → Existing Projects into
   Workspace → `Atlas-Blinking-LED-Baremetal-GNU/`. The name is a leftover from
   the lab template; `atlas_main.c` is the CNN app.
4. **Launch** `Atlas-Blinking-LED-Baremetal-Debug`. It runs `debug-hosted.ds`:
   reset → load `u-boot-spl.axf` (brings up DDR3) → load `atlas_main.axf`.
   HDMI stays dark until this runs — `hps_fpga_reset_n` gates the camera and the
   preloader releases it.
5. **Watch the App Console** (semihosting). Ships with `LIVE 1`, one line per
   inference:

   ```
   [   42]   3 of Hearts     logit  418  red  9210->red    min  310 max 5980  vs+2
   ```

**Before blaming the model:** the app prints `camera: mipi_cfg=.. cam_cfg=.. vs=..
pixclk=..` on startup. If `vs` does not advance, no MIPI frames are arriving and
every prediction is the network's answer to a blank image. Reseat the D8M ribbon.

---

## 3. Framing is the whole ball game

The model was trained on the card's **rank/suit index corner** — the top-left
25% x 30% of a perspective-corrected 250x350 card, squashed to 48x48. It has
never seen a whole card.

| Corner fills box | Whole card in box | Card area needed | 5% slip costs |
|------------------|-------------------|------------------|---------------|
| 54/54            | 4/54              | ~16 x 26 mm      | ~75 points    |

Hold the card so the **index corner alone fills the green box** — about 4x closer
than instinct says. Whole-card framing drops to near chance and looks like a
broken model rather than a framing mistake.

### Tuning ZOOM for your rig

The captured box is square; the model wants 0.6 aspect. `ZOOM_*` trims it.

| Define | Ships | Meaning                     | Sensitivity                |
|--------|-------|-----------------------------|----------------------------|
| ZOOM_X | 9     | Left edge of crop, in cells | Tight — retune per rig     |
| ZOOM_Y | 0     | Top edge of crop, in cells  | Tight — retune per rig     |
| ZOOM_W | 29    | Crop width; 29/48 = 0.60    | Forgiving — 24..32 = 54/54 |
| ZOOM_H | 48    | Crop height, full           | Leave alone                |

Procedure: keep `DUMP_PGM 1`, capture, read the printed ASCII PGM to see where
the corner landed, adjust `ZOOM_X`/`ZOOM_Y`, rebuild (Project → Build), relaunch.
While aiming set `PREVIEW_EVERY` to ~10 — semihosting `putchar` is slow enough
that a full 48x48 dump every frame dominates the loop.

---

## 4. The training pipeline

Only needed to retrain on your own deck. Each stage consumes the previous one's
output folder.

1. **Shoot the deck in a known order.** 54 photos: 13 Diamonds, 13 Clubs,
   13 Hearts, 13 Spades, then 2 Jokers. Each suit runs 2,3,…,10,J,Q,K,A.
   Name them so they sort: `img_001.png` … `img_054.png`.
   (`img_1, img_2, … img_10` does NOT sort — 10 comes before 2.)

2. **Sort into class folders.**
   ```
   python build_ordered_deck.py --input_dir ./image_data/my_deck --out_dir ./image_data/deck2
   ```
   No model, no manual confirmation — labels assigned by position.

3. **Warp and corner-crop.**
   ```
   python build_warped_dataset.py
   ```
   Runs the same card detector the live pipeline uses, perspective-corrects each
   card, takes the index corner. This is what makes training and inference
   compositions identical. Some detection failures are expected (cards clipped by
   the frame edge, low-contrast backgrounds); per-class success rate is printed.

4. **Augment.**
   ```
   python augment_dataset.py
   ```
   Rotation, perspective skew, brightness/contrast jitter, random crop-in, and
   random RGB channel gains. The channel gains matter most — they force suit to be
   learned by shape, since colour arrives separately as a hardware flag.
   Rotation is capped well below 180° on purpose: a 6 rotated 180° is a 9, so
   aggressive rotation would silently generate wrong labels.

5. **Split train/val.**
   ```
   python split_augmented_train_val.py --data_dir ... --out_dir ... --val_count 3
   ```
   Guarantees every class lands in both splits.

6. **Train.**
   ```
   python train_card_cnn_dualhead.py
   ```
   Shared conv backbone, colour flag bit concatenated into the feature vector,
   three heads. Joker samples have rank/suit loss masked out — only the joker head
   learns from them.

7. **Calibrate the colour threshold.**
   ```
   python calibrate_color_threshold.py --data_dir ./image_data/deck2_final/val
   ```
   Prints 98th-percentile redness for red- and black-suited cards separately.
   Feeds `RED_THRESH_SW` in the app and `RED_THRESH` in `downsample_48x48.v`.

8. **Export INT8 weights.**
   ```
   python export_weights_hex.py
   ```
   One `$readmemh`-ready file per layer plus `weight_summary.json`.

   > **RENAME REQUIRED.** The exporter writes `fc_shared_w.hex`, `fc_rank_w.hex`,
   > `fc_suit_w.hex`, `fc_joker_w.hex`. The Verilog reads `fcs_w.hex`,
   > `fcrank_w.hex`, `fcsuit_w.hex`, `fcjoker_w.hex`. Biases too. Nothing checks —
   > a missed rename compiles cleanly into a network of zeros.

9. **Verify offline, then compile.** Run `sim_card_cnn.py` on a saved capture
   before spending 40 minutes in Quartus (§8). Then compile and check the fit
   report: RAM blocks against the 553 budget, and that no weight ROM fell into
   logic.

**Honesty note:** when every image in a class derives from one photo via
augmentation, train and val are highly correlated. Val accuracy tells you "it
memorised this photo's augmented variants", not that it generalises. The board is
the real test set.

---

## 5. Network reference

Input 48x48 luminance, Q6.10 fixed point (1.0 = 1024). Convs are 5x5 pad 2, so
spatial size survives each conv and halves at each pool: 48 → 24 → 12.

| Layer     | Shape             | Weight file    | Weights | Biases |
|-----------|-------------------|----------------|--------:|-------:|
| conv1     | 1 → 8, 5x5 pad 2  | conv1_w.hex    |     200 |      8 |
| pool1     | 2x2 max, 48 → 24  | —              |       — |      — |
| conv2     | 8 → 16, 5x5 pad 2 | conv2_w.hex    |   3,200 |     16 |
| pool2     | 2x2 max, 24 → 12  | —              |       — |      — |
| fc_shared | 2305 → 64         | fcs_w.hex      | 147,520 |     64 |
| fc_rank   | 64 → 13           | fcrank_w.hex   |     832 |     13 |
| fc_suit   | 64 → 4            | fcsuit_w.hex   |     256 |      4 |
| fc_joker  | 64 → 2            | fcjoker_w.hex  |     128 |      2 |

The 2305 input to `fc_shared` is 16 x 12 x 12 = 2304 features plus one **colour
bit**. That bit is computed in fabric from raw RGB, not learned: a pixel is red
when R > 1.5·G and R > 1.5·B with R above `RED_MIN` (64); the frame is red when
the count clears `RED_THRESH`. Final argmax is colour-constrained, so a
red-flagged frame can only resolve to Hearts or Diamonds.

**Index order.** Rank 0..12 = 2,3,4,5,6,7,8,9,10,J,Q,K,A.
Suit 0..3 = Spades, Clubs, Hearts, Diamonds.

**Fixed point.** The downsampler stores a raw 5x5 luminance sum per cell, ITU-R
601 weighted as `(77R + 150G + 29B) >> 8`, max 25*255 = 6375. Software converts
to Q6.10 as `sum * 1024 / 6375`. Luminance not R+G+B, because training used PIL
`convert("L")` — the same weighting.

**Current build cost:** ALMs 60%, RAM blocks 471/553 (85%), DSP 9/112, block
memory bits 68%. `fcs_w` alone is 147,520 INT8 values and dominates M10K.

---

## 6. Register map

Lightweight HPS-to-FPGA bridge at `0xFF200000`. Plain PIOs — `card_cnn_avalon.v`
exists in the tree but is deliberately not instantiated.

| Offset | Name           | Bits   | Meaning                                              |
|--------|----------------|--------|------------------------------------------------------|
| 0x00   | img_wr_ctrl    | [3:0]  | [0] wr_en, [2] colour override en, [3] override val  |
| 0x10   | img_wr_data    | [15:0] | Pixel, Q6.10                                         |
| 0x20   | img_wr_addr    | [11:0] | 0..2303 as y*48 + x                                  |
| 0x30   | camera_trigger | [0]    | Request a snapshot                                   |
| 0x40   | snapshot_data  | [15:0] | Raw 5x5 luminance sum at current address             |
| 0x50   | snapshot_addr  | [11:0] | Cell index, plus diagnostics below                   |
| 0x60   | cnn_start      | [0]    | Edge-triggered start                                 |
| 0x70   | cnn_result     | [31:0] | See below                                            |

**cnn_result bit layout**

| Bits    | Field         | Notes                                       |
|---------|---------------|---------------------------------------------|
| [3:0]   | rank          | 0..12                                       |
| [5:4]   | suit          | 0..3                                        |
| [6]     | joker         | Overrides rank and suit when set            |
| [7]     | done          | Sticky                                      |
| [8]     | colour used   | Which branch the argmax was constrained to  |
| [9]     | snapshot_done |                                             |
| [31:16] | rank_score    | Winning rank logit, Q6.10                   |

**Diagnostic addresses on snapshot_addr** — reading past the 2304 image cells
returns camera telemetry:

| Addr | Value      | What it tells you                                         |
|------|------------|-----------------------------------------------------------|
| 2304 | red_count  | Red pixels in the captured frame                          |
| 2305 | colour_hw  | Hardware colour flag, 1 = red card                        |
| 2306 | vs_count   | **The one that matters** — MIPI frame counter             |
| 2307 | pixclk     | MIPI pixel-clock activity counter                         |
| 2308 | retries    | I2C config re-runs forced by the watchdog                 |
| 2309 | cfg_step   | How far the config sequence got                           |
| 2310 | status     | [0] mipi rel, [1] cam rel, [2] audio PLL ok, [3] HDMI rdy |

2306..2310 only exist in bitstreams built after the diagnostics were added. On an
older `.sof` they read 0, indistinguishable from a dead link — hence the
`HAVE_CAM_DIAG` gate.

---

## 7. Software knobs

All at the top of `atlas_main.c`. Shipped values are the ones that score 54/54.
Change one at a time.

| Define           | Ships | What it does                                                        |
|------------------|-------|---------------------------------------------------------------------|
| INVERT           | 0     | **Leave at 0.** Training used no inversion; 1 scores ~0             |
| NORMALIZE_MINMAX | 0     | 0 = fixed /255-equivalent scale (matches training); 1 stretches      |
| COLOUR_OVERRIDE  | 1     | Software decides red/black from red_count, not the hardware flag     |
| RED_THRESH_SW    | 5000  | Software red threshold — calibrate for your deck and lighting        |
| DUMP_PGM         | 1     | Print capture as ASCII PGM, for ZOOM tuning and sim_card_cnn.py      |
| LIVE             | 1     | Loop forever, one line per inference. 0 = single verbose run for sim |
| PREVIEW_EVERY    | 0     | Half-res 24x24 preview every N frames. 10 while aiming, 0 when set   |
| HAVE_CAM_DIAG    | 1     | Trust registers 2306..2310. 0 on a pre-diagnostics bitstream         |

---

## 8. Verification loop

The standing proof the accelerator is correct: a bit-exact software model and the
board agree on the same capture. Keep it working — it is how every real bug so far
was found.

1. Set `LIVE 0` and `DUMP_PGM 1`, run once, copy the whole App Console output to a
   text file.
2. Replay it:
   ```
   python sim_card_cnn.py capture.txt
   python sim_card_cnn.py capture.txt --png    # also render the capture
   ```
   `sim_card_cnn.py` reads the same `weights/*.hex` the RTL loads and reproduces
   Q6.10 with round-to-nearest before the shift. If the paste carries the
   `zoom window:` line, the same crop is applied, so sim and board stay aligned
   when you retune `ZOOM_*`.
3. Compare. Both colour branches are printed; read the one matching the board's
   `colour_hw`. Agreement means the datapath is fine and any error is framing,
   lighting or the model. Disagreement means RTL and weights have drifted apart.

> `sim_cnn.py` models the OLD 28x28 single-head datapath and no longer matches
> anything loaded. `sim_card_cnn.py` is the current one.

---

## 9. Known issues

Live conditions of the 2026-09-06 build, not history.

### HIGH — Timing constraints are wrong, and the real margin is thin

The stock GHRD SDC declares `FPGA_CLK2_50` and `FPGA_CLK3_50` with `-period 10`
(100 MHz). Those pins are 50 MHz on a DE10-Nano. `VIDEO_PLL` is configured for a
50 MHz reference and produces 20 MHz (`MIPI_REFCLK`) and 25 MHz (`VGA_CLK`), so
every derived video clock is analysed at double its real frequency.

That is why the report looks catastrophic — setup slack **-9.680 ns**, TNS
**-271,314** on `FPGA_CLK2_50`. The Fmax table tells the truer story: **50.81 MHz
achieved against 50 MHz actual**. It passes, by 1.6% at the slow 85°C corner.
Real, but thin, and nobody signed off on that number deliberately.

**Fix:** change the two `-period 10` lines in `soc_system_timing.sdc` to
`-period 20`, re-run timing analysis, read what is actually left. Do this before
adding anything to the video path.

### HIGH — No clock groups, CDC paths timed as synchronous

The SDC has no `set_clock_groups -asynchronous` and no
`derive_clock_uncertainty`. `VGA_CLK` (25 MHz) and the 50 MHz HPS domain are
unrelated, but the analyser times paths between them anyway. That is where
`FPGA_CLK1_50`'s **-5.560 ns** / TNS **-83.8** comes from, even though its
same-domain Fmax is 54.76 MHz — comfortably above 50.

The crossings are real and rest on hand-rolled 2–3 flop synchronisers
(`trig_sync`, `win_sr`, `edge_sr`). They are probably fine. Nothing has verified
that.

**Fix:** declare the groups asynchronous, add `derive_clock_uncertainty`, audit
whatever violations survive.

### MEDIUM — M10K budget nearly full

471/553 RAM blocks, 85%. The 320x240 frame buffer was already sacrificed once to
make room. There is no second sacrifice of that size available, so any
architecture change needs a memory plan before you start.

### MEDIUM — Horizontal resolution is really 29, not 48

`ZOOM_W 29` keeps 29 of the 48 captured columns and rescales back to 48 by
nearest-neighbour to hit the 0.6 aspect. The upsample adds no information. Worth
knowing before attributing a misread to the network rather than to sampling.

### MEDIUM — Weight filenames must be renamed by hand

`export_weights_hex.py` emits `fc_shared_*` / `fc_rank_*` / `fc_suit_*` /
`fc_joker_*`; the RTL reads `fcs_*` / `fcrank_*` / `fcsuit_*` / `fcjoker_*`.
Nothing checks. A missed rename compiles cleanly into a network of zeros.

### MEDIUM — Qsys does not refresh the black-box wrappers

`soc_system/soc_system_bb.v` and `_inst.v` are not rewritten by `qsys-generate`
and are still dated 2026-08-29. Change the Qsys system and move a port, and these
go stale silently. Quartus 25.1 also dropped `--quartus-project` from
`qsys-generate`.

### LOW — Toolchain papercuts

- **Quartus refuses read-only directories.** Clear the Windows ReadOnly attribute
  on the tree; git pack files are fine to leave alone.
- **JTAG shows two TAPs** (HPS and FPGA) → Programmer says "expected 1, found 2".
  Use Auto Detect.
- **Close Quartus before Arm DS** — USB-Blaster contention.
- **LEDs are FPGA debug, not software.** `ghrd_top.v:120` drives them from
  `{dbg_ready_latched, dbg_hdmi_int, dbg_lut_index, dbg_ack}`. `led_pio` exists in
  Qsys but its input and output are looped to each other and reach no pin.

### LOW — Stale files in the tree

`atlas_main_cards.c` (top of `LAB_5/`) is the older 14-class single-head app and
does not match this bitstream. `class_labels.txt` is that model's rank-only label
list. `my_deck/` is the matching legacy dataset — 14 rank folders, not the 54-card
set the deployed model wants. `card_cnn_avalon.v` is present but not instantiated.
None are wired into the build; they are just easy to pick up by mistake.

---

## 10. Troubleshooting

| Symptom                              | Likely cause                | What to do                                                        |
|--------------------------------------|-----------------------------|-------------------------------------------------------------------|
| HDMI stays black                     | Normal pre-preloader        | `hps_fpga_reset_n` gates the camera. Launch the DS debug config    |
| Every prediction identical/nonsense  | Camera not streaming        | Check `vs` advances. Reseat the D8M ribbon                        |
| `BLANK FRAME` in the log             | Snapshot max is 0           | Same; warning comes from the capture so works on any bitstream     |
| Right rank, wrong suit               | Colour threshold            | `COLOUR_OVERRIDE 1`, tune `RED_THRESH_SW` against printed red_count|
| Accuracy near chance                 | Framing, almost always      | Corner must fill the green box, then retune `ZOOM_X`/`ZOOM_Y`      |
| Consistently wrong after retraining  | Weights not renamed         | Confirm `fcs_*`/`fcrank_*`/`fcsuit_*`/`fcjoker_*` in project root  |
| Board and simulator disagree         | RTL and weights drifted     | Rebuild the bitstream from the weight files the sim reads          |
| DS cannot connect                    | USB-Blaster contention      | Close Quartus Programmer                                           |
| Programmer: expected 1, found 2      | Two JTAG TAPs               | Auto Detect, then pick the 5CSEBA6                                 |
| Inference loop crawls                | Semihosting `putchar`       | `PREVIEW_EVERY 0` and `DUMP_PGM 0` once framed                     |
| Quartus will not open the project    | ReadOnly attribute          | Clear it recursively                                               |

---

## 11. File map

**In this bundle**

| Path                        | Role                                              |
|-----------------------------|---------------------------------------------------|
| `DE10-Nano.sof`             | FPGA image, weights baked in. Program this        |
| `.../atlas_main.axf`        | Prebuilt binary — runs without rebuilding         |
| `.../atlas_main.c`          | Host app; the file you edit to tune ZOOM          |
| `.../u-boot-spl.axf`, `.dtb`| Preloader; brings up DDR3 before the app loads    |
| `.../debug-hosted.ds`       | DS script: reset → preloader → app                |
| `.../hps_0_arm_a9_0.h`      | Qsys address map                                  |
| `.../hwlib/`, `Makefile*`   | Only needed to rebuild after editing              |

**In the full project, if you get a copy**

| Path                             | Role                                         |
|----------------------------------|----------------------------------------------|
| `card_cnn/*.v`                   | The accelerator: core, conv, pool, fc        |
| `downsample_48x48.v`             | Crop, luma cells, red detector, snapshot RAM |
| `ON_CHIP_FRAM.v`, `FRAM_BUFF.v`  | 320x240 Bayer-skipped frame buffer           |
| `camera_capture.v`               | Display window, green crop box, HDMI mux     |
| `*.hex` (project root)           | The 12 weight files the RTL `$readmemh`s     |
| `soc_system_timing.sdc`          | Constraints — see issue 1                    |
| `sim_card_cnn.py`                | Bit-exact model, the verification loop       |
| `trainings/*.py`                 | The nine-stage training pipeline             |

**Toolchain invocations that actually work**

```
QUARTUS_ROOTDIR=C:\altera_lite\25.1std\quartus
# qsys tools live in sopc_builder\bin\

qsys-generate soc_system.qsys --synthesis=VERILOG \
  --output-directory=soc_system \
  --family="Cyclone V" --part=5CSEBA6U23I7DK

# note: --quartus-project was removed in 25.1
```
