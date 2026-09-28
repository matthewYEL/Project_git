# Handover — camera frame buffer to DDR3 (against LAB_5_rubric_satisfied)

This version is edited against the actual project you sent
(`LAB_5_rubric_satisfied.zip`), not a generic guide. `camera_capture.v` and
`ghrd_top.v` here are your real files with the DDR3 change applied — diff
them against your originals to see exactly what moved.

**Confirmed while editing:** `card_cnn_core` is wired directly into
`ghrd_top.v`, fed by `img_wr_addr`/`img_wr_data` PIOs from software —
completely separate from `camera_capture`'s HDMI/snapshot path. This DDR3
change touches `camera_capture.v` only. Nothing about `card_cnn_core`, the
`img_wr_*` PIOs, or the `downsample_28x28`/snapshot buffer path changes.

**Nothing here has been simulated or run on hardware.** Read "What is not
proven" at the bottom before starting.

## quartus_project/

| file | what changed |
|---|---|
| `ddr3_fram.v` | NEW. Drop-in replacement for `ON_CHIP_FRAM.v` |
| `camera_capture.v` | `ON_CHIP_FRAM fra(...)` → `ddr3_fram fra(...)`; added `avm_*`/`dbg_*` ports to the module's own port list so they reach the top level |
| `ghrd_top.v` | added `avm_*`/`dbg_*` wires; `u_camera` now connects them; `soc_inst` now connects `f2h_sdram0_data_*` |

## Bandwidth budget — for the datasheet and the viva

Port is 32 bits at 50 MHz once enabled (`AVL_DATA_WIDTH_PORT=32` in your
`soc_system.qsys`; you set the frequency in step 1 below).

| | |
|---|---|
| Peak | 32 bits x 50 MHz = 200 MB/s |
| Write, 2 px packed per 32-bit word | 38,400 words/frame x 60 fps = 2.30 M words/s |
| Display read-back | 2.30 M words/s |
| Total | 4.61 M words/s = 18.4 MB/s, ~9% of peak |

Why bursting matters: a non-pipelined master pays full latency per word
(~2.5-5 M words/s ceiling, uncomfortably close to 4.61 M needed). `ddr3_fram`
bursts 16 words at a time, dropping the requirement to ~288K bursts/s. Same
memory, same clock — the difference is transaction granularity.

---

## Order of operations

### 0. Make a rollback point FIRST

```powershell
git commit -am "working build before DDR3 frame buffer"
Copy-Item output_files\*.sof ..\sof_backup\
```

The current build works. Do not start this without a way back to it.

### 1. Reserve the DDR3 region in Linux — on the board, before any FPGA work

```bash
cat /proc/iomem        # see what the kernel already owns
```

Pick a free region near the top of DRAM, reserve it (`memmap=` kernel
argument or a `reserved-memory` device-tree node), reboot, confirm with
`/proc/iomem` that the kernel no longer lists it. Skipping this produces
corruption that looks exactly like an RTL bug — worth doing first precisely
so it's ruled out later.

One frame is 38,400 words = 153,600 bytes.

### 2. Set `BASE_WORD` in `ddr3_fram.v` to match

Default is `30'h0F00_0000` — a **word index**, not a byte address
(`f2h_sdram0_data` is word-addressed). Set it to the region from step 1.

### 3. Enable and export f2h_sdram0_data in Platform Designer

**Confirmed against your actual `soc_system.qsys`: this port is not
currently exported.** `hps_0`'s `F2H_SDRAM0_CLOCK_FREQ` sits at its disabled
placeholder value (`100`), same as the unused ports 1-5. Nothing else in
your qsys uses F2H SDRAM port 0, so there's no conflict — but this step is
real and required, not a formality:

1. Open `soc_system.qsys`.
2. Double-click `hps_0` → FPGA Interfaces tab → F2H SDRAM Interface →
   enable **Port 0**, width 32 bits.
3. Connect `hps_0.f2h_sdram0_clock` from `clk_0.clk` — the same source
   already exported as `clk` and wired to `fpga_clk_50` at your top level.
4. Export `hps_0.f2h_sdram0_data` as `f2h_sdram0_data`, type Avalon,
   direction End.
5. **Generate HDL.**
6. Confirm the generated port names before trusting what's in `ghrd_top.v`
   here — same check this project already does for every other conduit:

```powershell
Select-String -Path soc_system\soc_system.v -Pattern "f2h_sdram0"
```

### 4. Copy in the three files from `quartus_project/`

Overwrite your existing `camera_capture.v` and `ghrd_top.v`; add
`ddr3_fram.v` new. `avm_clk`/`avm_rst` in `ghrd_top.v` are wired to
`fpga_clk_50`/`~hps_fpga_reset_n` — the same net as `clk_0.clk` from step 3.
If that connection is ever changed to a different clock, this must change
with it.

### 5. Remove the old buffer from the project file list

```powershell
Remove-Item ON_CHIP_FRAM.v, FRAM_BUFF.v
```

Deleting the instantiation is not enough — unused modules still in the file
list still get compiled and still consume blocks.

### 6. Wire the debug counters somewhere readable

`dbg_rd_underrun`/`dbg_wr_overflow` from `ghrd_top.v` are currently
**unconnected to anything visible** — your 8 LEDs are already fully spoken
for (`dbg_ready_latched`, `dbg_hdmi_int`, `dbg_lut_index`, `dbg_ack`). Route
them out through a spare PIO, or watch them with SignalTap, rather than
leaving them silently unmonitored. They should read zero in steady state; if
`rd_underrun` climbs, the display is starving; if `wr_overflow` climbs,
camera pixels are being dropped.

### 7. Compile, check the fit report

Confirm M10K usage actually dropped. If it didn't, step 5 wasn't done
properly.

### 8. Program after Linux has booted

The SD card's `.rbf` is loaded by U-Boot at every boot and overwrites
anything programmed earlier. `scp` from the laptop, not the board.

---

## Bring-up order

```
./bridge_test 0x10040     # LEDs walk = bitstream is live
./probe_cnn 0x20000       # +0x18 must return 0xCA5D0001
```

Then, one variable at a time:

1. **Anything on HDMI at all?** Blank = read path not delivering — check
   `dbg_rd_underrun` first.
2. **Stable, or torn/rolling?** Points at the frame-start resync or read
   FIFO underrun — check counters before changing logic.
3. **Shifted horizontally, or crop box misaligned with what the CNN sees?**
   The latency trap (below) — check this before touching crop coordinates.
4. **Colours wrong / noise-like?** The 2x2 Bayer skip gates both the
   position counter and write enable to keep the colour phase valid; if that
   broke, the mosaic is invalid.
5. Only then run the snapshot path (`downsample_28x28`) and `card_cnn` —
   this whole change doesn't touch that path, so it should behave exactly
   as before.

## The latency trap

`R_DATA` is pipelined through two registers to match `FRAM_BUFF`'s
altsyncram (2 cycles) — what keeps `camera_capture.v`'s `win_sr[3]` correct.
If the image shifts or tears diagonally, verify in simulation that `R_DATA`
appears exactly 2 `R_CLK` edges after `R_DE`, before touching anything else.

## What is not proven

Not simulated, not run. In order of suspicion:

1. Three clock-domain crossings — `W_CLK`→`avm_clk`, `avm_clk`→`R_CLK`, the
   frame-start resync.
2. The read prefetch — depends on the display staying strictly sequential
   and VS-restarted.
3. The read/write arbiter under real HPS contention on the same bridge.

Worth a golden-vector testbench before hardware: drive a known Bayer pattern
in, read it back, compare bit-exact.
