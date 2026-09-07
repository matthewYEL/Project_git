# Running the card CNN under Linux, on both Cortex-A9 cores

The same FPGA bitstream as the bare-metal flow, but the HPS boots Linux from
an SD card and the work is split across the two Cortex-A9 cores as two
CPU-pinned pthreads.

```
                 FPGA fabric                    HPS, Linux SMP
  D8M ──▶ frame buf ──▶ crop/96x96 ──▶ 3-head CNN
                                          │
                             LW bridge 0xFF200000 (/dev/mem)
                                          │
                        ┌─────────────────┴──────────────────┐
                        │                                    │
             CPU0  vision thread                   CPU1  game thread
             trigger, read 9216 cells,             blackjack state machine,
             scale to Q6.10, upload,               Monte-Carlo odds engine
             start CNN, poll, decode               (200k trials per card)
                        └────────── ring buffer ─────────────┘
```

Why this split: the vision thread is I/O bound — about 64,500 uncached bridge
accesses per frame — while the odds engine is pure compute. Putting them on
separate cores means the camera keeps running at full rate while the odds are
being computed, instead of the two taking turns.

---

## 1. What you need

- DE10-Nano + D8M camera on GPIO0, HDMI display
- microSD card, 8 GB or larger, and a card reader
- The Terasic **DE10-Nano LXDE or console Linux SD image**
  (Terasic DE10-Nano CD-ROM → "Linux LXDE Desktop" / "Linux Console").
  Any DE10-Nano image with an `a2` boot partition works; the Intel GSRD
  and the Angstrom images are equivalent for this purpose.
- `soc_system.rbf` — already built and sitting next to this README, converted
  from the 2026-09-06 bitstream with:

  ```
  quartus_cpf -c output_files/DE10-Nano.sof linux/soc_system.rbf
  ```

  Regenerate it whenever you recompile the FPGA, or Linux will configure the
  fabric with a stale accelerator and the app will read nonsense.

---

## 2. Prepare the SD card

1. **Write the stock image** with Win32DiskImager, balenaEtcher or `dd`. Boot it
   once as-is and confirm you get a login prompt on the UART (115200 8N1, the
   mini-USB at J13 enumerates as a serial port) or on HDMI. Do not skip this —
   it separates "my image is bad" from "my bitstream is bad" later.

2. **Replace the FPGA image.** The first partition is FAT32 and mounts on any
   PC. Copy `soc_system.rbf` from this folder over the one already there,
   keeping the existing name:

   ```
   copy soc_system.rbf  <FAT partition>\soc_system.rbf
   ```

   Some images name it `de10_nano.rbf` or put it under `extlinux/`. Match
   whatever name is already present rather than adding a second file — u-boot
   loads the name baked into its boot script, and an unused extra file looks
   exactly like a working one until you wonder why nothing changed.

3. **Copy the application sources** onto the ext partition (or `scp` them after
   boot — see below).

The partition layout you should see, for reference:

| Partition | Type | Holds |
|-----------|------|-------|
| 1 | FAT32 | `zImage`, `soc_system.rbf`, `*.dtb`, `u-boot.scr` / `extlinux/` |
| 2 | ext3/ext4 | root filesystem |
| 3 | raw, type `a2` | preloader + u-boot |

The `a2` partition is not a filesystem; the boot ROM reads it directly. Leave
it alone — this project reuses the stock preloader and u-boot rather than
building its own.

> **Why the stock preloader is fine.** The preloader configures DDR3 and the
> HPS pin mux, both of which are board properties, not project properties. The
> FPGA side of this design talks to the HPS only through the lightweight bridge
> and needs nothing special from the preloader beyond releasing
> `hps_fpga_reset_n`, which every DE10-Nano preloader does. If you do hit DDR3
> or pin-mux trouble, regenerate the preloader from this project's
> `hps_isw_handoff/` with `bsp-editor` and rewrite the `a2` partition.

---

## 3. Boot and check the fabric is live

Set the MSEL DIP switches for SD-card boot (the DE10-Nano default,
`00000` on SW10) and power on. Log in over UART or HDMI.

```sh
# 1. Both cores must be up. This is a hard requirement of the demo.
nproc                      # expect 2
grep -c ^processor /proc/cpuinfo
dmesg | grep -i smp        # "SMP: Total of 2 processors activated"

# 2. The FPGA must be configured and the bridges open.
cat /sys/class/fpga/fpga0/status 2>/dev/null    # if the fpga-mgr driver exists
ls /sys/class/fpga-bridge/                      # lwhps2fpga, hps2fpga, fpga2hps
for b in /sys/class/fpga-bridge/*/ ; do
    echo 1 > "$b/enable" 2>/dev/null
done
```

On most DE10-Nano images u-boot already programmed the FPGA and enabled the
bridges before handing over, and the loop above is a no-op. If
`/sys/class/fpga-bridge/` does not exist at all, the bridges are enabled by
u-boot only, which is fine — the check that matters is whether the app can read
sensible telemetry (next section).

---

## 4. Build and run

Build **natively on the board**. The Intel SoC EDS that ships with this project
provides only `arm-eabi` (bare metal), so there is no Linux cross-toolchain on
the development PC; every DE10-Nano Linux image has gcc.

```sh
scp -r linux/ root@<board-ip>:/root/card_cnn
ssh root@<board-ip>
cd /root/card_cnn
make
./card_cnn --stats
```

`/dev/mem` is needed, so run as root (or via `sudo`).

Expected first lines:

```
DE10-Nano card CNN -- Linux, 2 CPUs online
mapped LW bridge 0xFF200000 (4096 bytes)
camera: mipi_cfg=1 cam_cfg=1 vs=417 pixclk=63488 retries=0 (audpll=1 hdmi=1)

task allocation:
  CPU0  vision  camera trigger, 96x96 snapshot read, Q6.10 scale, ...
  CPU1  game    blackjack state machine, Monte-Carlo odds ...

[vision] running on CPU0 -- camera + CNN accelerator
[game]   running on CPU1 -- blackjack + Monte-Carlo odds
```

**`vs=` must advance between runs.** If it is stuck, no MIPI frames are
arriving and every prediction is the network's answer to a blank image —
reseat the D8M ribbon. The app says so explicitly rather than making you guess.

### Options

| Flag | Effect |
|------|--------|
| `--stats` | per-core utilisation from `/proc/stat` every 5 s, plus per-thread CPU time |
| `--rt` | run both worker threads `SCHED_FIFO` instead of `SCHED_OTHER` |
| `--quiet` | only print committed cards, not every frame |
| `--fake` | deal a fixed card sequence, never touch the FPGA |

`--fake` needs no board and no root. Use it to rehearse the demo, to check the
two-core split works on a given image, and to show the game logic if the camera
is misbehaving.

### Playing

Hold a card so its **index corner fills the green box** on HDMI — roughly
16 x 26 mm of card, about 4x closer than instinct says. The model was trained
on the corner, never on a whole card; whole-card framing drops to near chance
and looks like a broken model rather than a framing mistake. A card must read
the same for 3 consecutive frames before it counts, so hold it steady.

Show a **Joker** to reset the round.

---

## 5. Showing the marker that both cores are working

The rubric asks for clear task allocation, scheduling, and evidence that both
cores contribute meaningfully. Each of those has something concrete to point at:

**Task allocation** — printed at startup, and visible per line: every vision
line is tagged `cpu0`, every game line `cpu1`. Those tags are live
`sched_getcpu()` results, not constants.

**Both cores really running** — in a second terminal:

```sh
top -1                      # press 1 for per-core rows
# or
ps -eLo pid,tid,comm,psr,pcpu | grep -E 'card_cnn|vision|game|PSR'
```

`psr` is the core each thread is on: the `vision` thread sits on 0, `game` on 1.
The threads are named, so they are identifiable in `top`/`htop` directly.

**Meaningful contribution** — `--stats` prints both cores' utilisation and each
thread's own accumulated CPU time:

```
[stats]  cpu0  71.2%  cpu1  48.9% | vision 214 frames (18.4s cpu) | game 9 cards, 7 odds runs (12.1s cpu)
```

Two separate accounting sources agreeing (kernel `/proc/stat` per core, and
`CLOCK_THREAD_CPUTIME_ID` per thread) is much harder to wave away than a
printed thread ID.

**Scheduling** — run with `--rt` and show the policy changed:

```sh
chrt -p $(pgrep card_cnn)
ps -eLo tid,comm,cls,rtprio | grep -E 'vision|game'
```

**Affinity is enforced, not hoped for** — `pin_to_cpu()` calls
`pthread_setaffinity_np`, then calls `sched_getcpu()` and complains loudly if it
did not land where it asked. If you see no `!!` lines, the pinning took.

---

## 6. If something is wrong

| Symptom | Cause | Fix |
|---------|-------|-----|
| `open /dev/mem: Permission denied` | not root | `sudo ./card_cnn` |
| `mmap: Operation not permitted` | kernel has `CONFIG_STRICT_DEVMEM` | boot with `iomem=relaxed` on the kernel command line |
| All registers read `0xFFFFFFFF` or `0` | bridges disabled, or FPGA unconfigured | enable `/sys/class/fpga-bridge/*/enable`; confirm the `.rbf` actually loaded |
| `vs=` never advances | no MIPI frames | reseat the D8M ribbon; the app prints which stage failed |
| `only 1 CPU online` | kernel booted UP, or `maxcpus=1` | check `dmesg | grep -i smp` and the kernel command line |
| Every prediction wrong | framing, almost always | corner must **fill** the green box; then retune `ZOOM_X`/`ZOOM_Y` in `card_cnn_hw.h` |
| Predictions differ from bare metal | `ZOOM_*` drifted between the two apps | `make check` — it fails loudly if the front end no longer matches the model |
| Right rank, wrong suit | colour threshold | tune `RED_THRESH_SW` in `card_cnn_hw.h` against the printed `red` count |
| HDMI black | FPGA not configured | the `.rbf` did not load; check the u-boot log over UART |

---

## 7. Keeping the three implementations honest

There are now three implementations of the same pipeline: the RTL, the
bare-metal `atlas_main.c`, and this app. They agree only as long as someone
keeps them agreeing.

```sh
make check
```

builds `preprocess_check` and replays a **real board capture** (`capture.txt`,
via `testdata/`) through this app's fixed-point front end, requiring all 9216
Q6.10 pixels to match what `sim_card_cnn.py` computes. It fails loudly if
`ZOOM_*` or the scaling is changed on one side only — which is exactly the
class of bug that produces a confidently wrong card.

Run it after any change to `card_cnn_hw.h`. If you retune `ZOOM_*` on the
bench, regenerate `testdata/` for the new window and re-run.

---

## 8. Files

| File | Role |
|------|------|
| `card_cnn_linux.c` | the application: mmap, two pinned threads, game, stats |
| `card_cnn_hw.h` | register map, geometry, `ZOOM_*`, and `ccp_preprocess()` |
| `preprocess_check.c` | offline cross-check against `sim_card_cnn.py` |
| `testdata/` | a real board capture and the model's expected Q6.10 output |
| `soc_system.rbf` | FPGA image for Linux boot, from the 2026-09-06 `.sof` |
| `Makefile` | native build on the board; `make check` for the self-check |
