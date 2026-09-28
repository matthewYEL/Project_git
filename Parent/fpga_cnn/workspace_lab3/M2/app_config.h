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

/* Print the FreeRTOS run-time task statistics every N results (and on any
 * press with SW2 up). This is the scheduling evidence: per-task CPU0 time
 * straight out of the kernel, plus CPU1's job count and busy time. 0 = only
 * on SW2. */
#define STATS_EVERY 5

/* SW3 up: start a job automatically this often (measured from the previous
 * job's start, and only when none is in flight), so no KEY press is needed.
 * With SW0 also up, each job prints the ASCII preview -- a slow viewfinder for
 * aiming the card corner when there is no HDMI picture. */
#define AUTO_PERIOD_MS  3000u

/* KEY1 held this long clears the HDMI tracker; a shorter press undoes the
 * newest card only. */
#define KEY1_CLEAR_MS   2000u

int rtos_main( void );

#endif /* APP_CONFIG_H */
