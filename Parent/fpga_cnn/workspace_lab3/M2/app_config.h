/*
 * Which application this build produces.
 *
 *   RTOS_MODE 1   FreeRTOS on CPU0 (Input / Vision / Report tasks) with the
 *                 vision job -- capture, sample, CNN, decode -- on CPU1
 *                 (app_rtos.c, amp.c). The Milestone 1 "both cores + RTOS"
 *                 build. Default.
 *   RTOS_MODE 0   the single-core polling loop in atlas_main.c -- FreeRTOS is
 *                 linked but never started. The fallback if the RTOS image
 *                 misbehaves on a board (atlas_main_single.axf is a prebuilt
 *                 copy of it).
 *
 * Both builds run the same pipeline code (vision_run in atlas_main.c) and print
 * the same result line, so switching cannot change what the network is shown.
 *
 * History: the first RTOS build overlapped host-side pixel preprocessing on
 * CPU1 with capture on CPU0 -- a pipeline for LIVE inference -- and was retired
 * when the 384x384 snapshot build made the fabric write the CNN input itself.
 * CPU1's job is now the whole vision job instead.
 */
#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#define RTOS_MODE   1

/* Release the second Cortex-A9 and give it the vision job.
 *
 * With this at 0 -- or if CPU1 fails to check in at run time -- the jobs run on
 * CPU0 and the app still works. Bringing a second core out of reset is the part
 * of this design most likely to fail on a given board, and it is not allowed to
 * take the demo down with it. */
#define USE_CORE1   1

/* How long CPU0 waits for CPU1 to finish one job before declaring it wedged and
 * running jobs itself. A job is at most the capture timeouts (a few s) plus the
 * 5 s inference timeout; 12 s also stays under the ~21 s wrap of the 32-bit
 * global-timer counter the wait is timed on. */
#define CORE1_JOB_TIMEOUT_US    12000000UL

/* Print the FreeRTOS run-time task statistics every N results. This is the
 * scheduling evidence: per-task CPU0 time straight out of the kernel, plus
 * CPU1's job count and busy time. 0 = never. (SW2 used to print them on
 * demand; since 29 Sep it shows the 3x3 guide grid on HDMI.) */
#define STATS_EVERY 5

/* SW3 up: start a job automatically this often (measured from the previous
 * job's start, and only when none is in flight), so no KEY press is needed.
 * With SW0 also up, each job prints the ASCII preview -- a slow viewfinder for
 * aiming the card corner when there is no HDMI picture. */
#define AUTO_PERIOD_MS  3000u

/* KEY1 held this long clears the HDMI tracker; a shorter press undoes the
 * newest card only. */
#define KEY1_CLEAR_MS   2000u

/* Whole-grid scan (M2 demo, 29 Sep): UART m toggles whether KEY0 scans the
 * whole grid or reads one card at the green box, UART s scans whatever the
 * mode. KEY0 scans the whole grid from power-up (since 30 Sep), so the demo
 * needs no UART cable or PuTTY at all. SCAN_ORIENT_DEFAULT indexes
 * app_rtos.c's ORIENTS[]: 0 upright camera, 1 turned clockwise, 2
 * anticlockwise, 3 upside down -- set it from the rig (UART o cycles it at run
 * time). */
#define SCAN_AUTO_DEFAULT       1
#define SCAN_ORIENT_DEFAULT     0

/* SW1: pictures kept in DDR for Arm DS to save (pics.h), the newest this many;
 * past it the oldest is overwritten. 512x384 x 2 bytes each, ~9.4 MB for 24. */
#define PIC_SLOTS               24

/* Reads per card in a grid scan (1, 3 or 5): each through a slightly different
 * window (turned, shifted, zoomed), majority vote. No time limit in the demo,
 * so 5: ~1.7 s each, ~77 s for a 3x3. On the 28 Sep photos, 243 card reads:
 * 1 read 237 right, 5 reads 241. app_rtos.c VARIANTS. */
#define SCAN_READS              5

int rtos_main( void );

#endif /* APP_CONFIG_H */
