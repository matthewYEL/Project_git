# FreeRTOS on CPU0, bare-metal worker on CPU1

The Arm DS build of the card demo, using both Cortex-A9 cores with an RTOS on
one of them. This is the bare-metal counterpart of `../../linux/`, which does
the same job under Linux with pinned pthreads.

Both exist on purpose: the Linux build is the safer thing to demo, the
bare-metal build is the one that shows the mechanism — there is no kernel
hiding the second core's release.

---

## 1. What runs where

```
CPU0 -- FreeRTOS 3 tasks                    CPU1 -- no OS at all
┌────────────────────────────────┐          ┌──────────────────────────┐
│ Capture (prio 3)               │          │ worker loop              │
│   camera trigger               │  slot    │   poll req != done       │
│   read 2304 raw cells  ────────┼─────────▶│   preprocess() to Q6.10  │
│                                │          │   set done, raise SGI 0  │
│ Infer   (prio 2)               │◀─────────┤                          │
│   collect q from CPU1          │  done    └──────────────────────────┘
│   upload 2304 px, start CNN    │           shared block @ 0x03000000
│   poll result                  │           stack @ 0x03018000
│                                │
│ Report  (prio 1)               │
│   printf over semihosting      │
└────────────────────────────────┘
```

Two frame slots circulate between Capture and Infer through FreeRTOS queues, so
CPU0 reads frame N+1 out of the FPGA while CPU1 preprocesses frame N. Priorities
follow deadlines: the camera is the only stage that can lose data by being late,
so it is highest; `printf` over semihosting is slow and nothing waits on it, so
it is lowest.

CPU1 never touches a FreeRTOS object. The kernel is not SMP-safe, and the whole
handshake is deliberately just two sequence numbers in shared DDR3.

---

## 2. How CPU1 is actually started

The part that is easy to get wrong:

1. CPU1 is held in reset by `rstmgr.mpumodrst` bit 1 (`0xFFD05010`) after the
   preloader runs.
2. When that bit clears, **CPU1 fetches from physical address 0.** It does not
   start in the boot ROM and does not read any start-address register by itself.
3. So `ampStartCore1()` writes a five-instruction trampoline to address 0 first.
   It loads `sysmgr.romcodegrp.cpu1startaddr` (`0xFFD080C4`), reads the entry
   point out of it, and branches. This is the same two-step Linux uses in
   `arch/arm/mach-socfpga/headsmp.S`.

Overwriting address 0 is safe here: it holds u-boot SPL's vector table, SPL has
finished by the time the app runs, and the app's own vectors are reached through
VBAR rather than through address 0.

`ampCore1Entry` sets CPU1's stack, enables the FPU exactly as hwlib's
`lowlevel_init` does for CPU0 — that code runs on CPU0 only, and a
compiler-emitted NEON instruction on a core with the FPU off would fault into a
vector table CPU1 does not have — and drops into the worker loop.

**If CPU1 never checks in, the demo still runs.** `ampStartCore1()` waits 200 ms
for a magic value in shared memory and reports honestly; `Infer` then does the
preprocessing itself and the report line says `pre ... on CPU0`. Releasing a
second core is the part most likely to fail on a given board, and it is not
allowed to take the rest down with it.

---

## 3. What the lab template did not have

`alt_base.c` enables NEON and jumps to `_mainCRTStartup`. That is the whole of
the lab's startup. Everything an RTOS needs is added in `a9_startup.c`:

| Piece | Why |
|-------|-----|
| 8-entry vector table + VBAR | there was none; SVC and IRQ route to the FreeRTOS port |
| IRQ / abort / undefined mode stacks | `FreeRTOS_IRQ_Handler` runs on the IRQ stack before switching to system mode |
| GIC distributor + CPU interface init | nothing had configured either |
| A9 private timer, IRQ 29 | the 1 kHz tick |
| A9 global timer | free-running timebase for run-time stats and the AMP timeouts |
| Data-abort reporting | prints DFAR/DFSR — on this platform a data abort is almost always a bad PIO address |

The tick divisor comes from the real `mpu_periph_clk`, read through hwlib's
`alt_clk_freq_get()`, not from a guessed constant. A wrong guess gives a
scheduler running at the wrong speed that never looks broken.

---

## 4. Build and run

```
# from the Intel Embedded Command Shell, so SOCEDS_DEST_ROOT is set
make
```

Then program the FPGA with `DE10-Nano.sof`, close Quartus (USB-Blaster
contention), and launch the `Atlas-Blinking-LED-Baremetal-Debug` configuration
in Arm DS as usual. `debug-hosted.ds` resets, loads `u-boot-spl.axf` to bring up
DDR3, then loads `atlas_main.axf`.

### Switches — `app_config.h`

| Define | Ships | Meaning |
|--------|-------|---------|
| `RTOS_MODE` | 1 | 1 = FreeRTOS + CPU1. **0 = the original single-threaded polling app, unchanged** |
| `USE_CORE1` | 1 | 0 keeps FreeRTOS but does the preprocessing on CPU0 |
| `CORE1_FRAME_TIMEOUT_US` | 100000 | how long Infer waits for CPU1 before doing the work itself |
| `STATS_EVERY` | 10 | print FreeRTOS run-time stats every N frames; 0 = off |

`RTOS_MODE 0` is the escape hatch. It rebuilds the exact application that was
tuned on the bench and reads the deck 54/54, so if anything about the scheduler
or the second core misbehaves during a demo, one define gets the known-good
build back.

---

## 5. Expected output

```
=== DE10-Nano card CNN -- FreeRTOS on CPU0, worker on CPU1 ===
  mpu_periph_clk 200000000 Hz
camera: mipi_cfg=1 cam_cfg=1  vs=417 pixclk=63488 retries=0   (audpll=1 hdmi=1)
  releasing CPU1 (trampoline at 0x0, entry via sysmgr.cpu1startaddr)...
  CPU1 is up: preprocessing runs there, shared block at 0x03000000
  tasks: Capture(p3) -> Infer(p2) -> Report(p1), 2 frame slots
  tick: private timer, mpu_periph_clk 200000000 Hz, load 199999 -> 1000 Hz
  starting scheduler

[    0] 3 of Hearts     logit   418  red  9210->red    min  310 max 5980  | cap 8123 us  pre 2110 us on CPU1  inf 41230 us
...

  FreeRTOS run-time stats (CPU0)
  Task            Abs time     %
  Capture         120414          18%
  Infer           418922          63%
  Report          71204           10%
  IDLE            52001           7%
  Tmr Svc         12              <1%
  CPU1 worker: 10 frames, 21102 us busy (2110 us/frame)
  preprocessing: 10 frames on CPU1, 0 on CPU0
```

---

## 6. Showing it to a marker

| Claim | Where it is proved |
|-------|--------------------|
| An RTOS is running | the tick line, and the run-time stats table — per-task CPU time can only come from a scheduler |
| Task allocation is deliberate | three named tasks with stated priorities, printed at startup, tied to deadlines in §1 |
| Scheduling is real | `vTaskGetRunTimeStats()` percentages shift with what the camera is doing; `Infer` blocks rather than spins |
| Both cores contribute | `pre NNNN us on CPU1` on every frame, plus `preprocessing: N frames on CPU1, M on CPU0`. Set `USE_CORE1` to 0, rebuild, and the same counter moves to CPU0 — the difference is the second core |
| CPU1 is genuinely a second core | it is released from reset by this code (§2); in Arm DS, the Debug view shows two Cortex-A9 contexts and CPU1 stopped inside `ampCore1Main` |

The last one is the strongest demonstration available: pause the debugger and
show CPU1's call stack sitting in `ampCore1Main`, in a loop that CPU0 never
enters.

---

## 7. Files

| File | Role |
|------|------|
| `app_config.h` | which application this build produces |
| `a9_startup.c` | vectors, VBAR, mode stacks, GIC, tick, global timer, fault reporting |
| `FreeRTOSConfig.h` | kernel configuration for the Cyclone V GIC and timers |
| `amp.h` / `amp.c` | CPU1 release, shared block, cache maintenance, req/done handshake |
| `app_rtos.c` | the three tasks, the queues, the report and stats output |
| `card_pipeline.h` | pipeline stages, geometry and result bits, shared with `amp.c` |
| `atlas_main.c` | unchanged pipeline stages; `main()` dispatches on `RTOS_MODE` |
| `FreeRTOS-Kernel/` | vendored kernel, MIT licensed, ARM_CA9 GCC port |

`atlas_main.c.prertos-bak` is the file as it was before this work, for diffing.

---

## 8. Known limitations

- **Cache coherency is handled by hand.** CPU0 and CPU1 exchange data through
  explicit clean/invalidate to the point of coherency (`amp.c`), not through the
  SCU. This is correct whether or not the MMU is enabled, but it does mean the
  shared block layout matters: the control fields sit in the first 32 bytes of
  each structure so the handshake only has to push one cache line around. A
  compile-time assertion checks the block cannot reach CPU1's stack, but nothing
  checks that a field added to the middle stays inside the maintained region.
- **The CPU1 doorbell SGI is an optimisation, not a mechanism.** `ampWait()`
  also polls with a timeout, so a lost SGI costs latency, not correctness. The
  SGI path has not been separately verified on hardware.
- **CPU1 has no exception vectors.** If it faults it will restart through the
  trampoline at address 0 rather than reporting. The `core1_frames` counter
  stalling is the symptom.
- **Not verified on hardware.** Everything here builds clean and the vector
  table, the CPU1 entry stub and the shared-block layout have been checked in
  the linked binary — but this code has not been run on the board.
