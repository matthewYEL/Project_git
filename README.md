# Card recognition on DE10-Nano — run-it-on-your-board bundle

3-head CNN (rank 13 / suit 4 / joker 1) in FPGA fabric, driven by a bare-metal
Cortex-A9 app. Weights are baked into the bitstream via `$readmemh`, so you do
**not** need Quartus or Python to try this — just program the `.sof` and run
the app.

> Full bench guide, training pipeline and known issues: **HANDBOOK.md** in this
> folder, or https://claude.ai/code/artifact/5d80e075-81ac-45fe-ae69-eff863c78e6e

## What you need

- DE10-Nano + D8M (OV8865) camera daughter card on GPIO0
- HDMI display (shows the live preview + the green target box)
- USB-Blaster (Quartus Programmer) + the Arm DS / Intel SoC EDS toolchain
- `SOCEDS_DEST_ROOT` set — start Arm DS from the Embedded Command Shell

## Contents

```
DE10-Nano.sof                          FPGA image, weights baked in
Atlas-Blinking-LED-Baremetal-GNU/      Arm DS bare-metal project
  atlas_main.c                         host app (the file you will edit)
  atlas_main.axf                       prebuilt binary — run this first
  u-boot-spl.axf/.dtb                  preloader, inits DDR3
  debug-hosted.ds                      DS script that loads the preloader
  Atlas-...-Debug.launch               DS debug configuration
  hps_0_arm_a9_0.h                     Qsys address map
  hwlib/, Makefile*, .cproject/.project  needed only if you rebuild
```

## Run it

1. **Program the FPGA.** Quartus Programmer → `DE10-Nano.sof` → Start.
   Do this after every board power cycle; the `.sof` is volatile.
2. **Import the project.** Arm DS → File → Import → Existing Projects into
   Workspace → point at `Atlas-Blinking-LED-Baremetal-GNU/`.
3. **Launch.** Run the `Atlas-Blinking-LED-Baremetal-Debug` configuration. It
   runs `debug-hosted.ds` (resets, loads `u-boot-spl.axf` to bring up DDR3),
   then loads `atlas_main.axf`.
4. **Watch the App Console.** `printf` goes there over semihosting. It builds
   with `LIVE 1`, so it loops on inference and prints a prediction per frame.
   Stop the debugger to exit.

## Framing is the whole ball game

The model was trained on the card's **rank/suit index corner**, not the whole
card — the top-left 25% x 30% of a perspective-corrected 250x350 card, squashed
to 48x48.

- Corner filling the green box → 54/54 on our deck.
- Whole card in the box → 4/54, near chance.

So aim so that corner **fills** the green box on the HDMI output: roughly
16 x 26 mm of card, about 4x closer than you would instinctively hold it.
Tolerance is tight — sliding 5% of a card width off the corner costs ~75 points
of accuracy.

`ZOOM_X/ZOOM_Y` in [atlas_main.c:149](atlas_main.c#L149) trim the captured
square down to the 0.6 aspect ratio the model expects, and **you will need to
retune them for your rig**. `DUMP_PGM 1` prints the capture as an ASCII PGM —
read the right offsets off that, edit the defines, rebuild (Project → Build),
relaunch. `ZOOM_W` is forgiving (24..32 all score 54/54); `ZOOM_X`/`ZOOM_Y` are
where the precision is needed.

## First thing to check if every prediction is wrong

The app reads MIPI diagnostics before it starts and prints
`camera: mipi_cfg=.. cam_cfg=.. vs=.. pixclk=..`.

If `vs_count` does not advance, no camera frames are arriving and every
prediction is the network's answer to a blank image — reseat the D8M ribbon,
that is the usual cause. The console spells out which stage failed.

## Not included

The training pipeline (PyTorch scripts, dataset, `export_weights_hex.py`) and
the Quartus sources. Ask if you want to retrain or recompile — different bundle.
