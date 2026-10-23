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

/* The individual practical (12 Oct, "individual demo.jpeg"): each KEY0 reads
 * the one card in the green box, and its value -- joker 0, ace 1, 2..10, J 11,
 * Q 12, K 13 (grid_value) -- fills an N x N matrix row by row on HDMI; a card
 * above 9 is not a digit and is refused. KEY1 tap undoes the last one, held
 * starts a new matrix. 0 = the M2 app as it was (KEY0 scans the whole grid). */
#define MATRIX_N                0

/* Whole-grid scan (M2 demo, 29 Sep): UART m toggles whether KEY0 scans the
 * whole grid or reads one card at the green box, UART s scans whatever the
 * mode. KEY0 scans the whole grid from power-up (since 30 Sep; not in the
 * matrix practical), so the demo needs no UART cable or PuTTY at all.
 * SCAN_ORIENT_DEFAULT indexes app_rtos.c's ORIENTS[]: 0 upright camera, 1
 * turned clockwise, 2 anticlockwise, 3 upside down -- set it from the rig
 * (UART o cycles it at run time). */
#define SCAN_AUTO_DEFAULT       ( MATRIX_N == 0 )
#define SCAN_ORIENT_DEFAULT     0

/* SW1: pictures kept in DDR for Arm DS to save (pics.h), the newest this many;
 * past it the oldest is overwritten. 512x384 x 2 bytes each, ~9.4 MB for 24. */
#define PIC_SLOTS               24

/* Reads per card in a grid scan (1, 3 or 5): each through a slightly different
 * window (turned, shifted, zoomed), majority vote. No time limit in the demo,
 * so 5: ~1.7 s each, ~77 s for a 3x3. On the 28 Sep photos, 243 card reads:
 * 1 read 237 right, 5 reads 241. app_rtos.c VARIANTS. */
#define SCAN_READS              5

/* M3, "suit sort" (21 Oct): the camera watches the board all the time -- no
 * KEY per card. A card shown in the SHOW box (right of the board) is read and
 * HDMI says where it goes: one column per suit (spades, hearts, clubs,
 * diamonds, left to right), highest rank at the top; only the order counts,
 * gaps are fine. An opponent's joker dropped anywhere is found and pointed at
 * until it is taken off. Rules: sortgame.c. 0 = the M2 app as it was.
 * RTOS build only (the game lives in the Vision task). */
#ifndef SORT_GAME                       /* gui_host_test.c builds the M2 screens with -DSORT_GAME=0 */
#define SORT_GAME               1
#endif

/* Board rows per suit column. 4 fits the M2 4x4 camera height (~60 px cards,
 * ~1 cm gaps); 3 would let the camera come down for bigger cards. */
#define SORT_ROWS               4

/* Identical card finds in a row before the game acts on what it sees: a hand
 * over the board changes the boxes, so it waits until the hand is gone. One
 * find is ~0.7 s (capture + locate). */
#define SORT_STILL              2

/* A card's look (sortgame.c sort_sig, 16 cells scaled to its brightest) may
 * differ from the one stored when it was placed by this much (mean per cell,
 * 0..255) before the game re-reads it -- how a joker laid exactly over one of
 * our cards is caught. lab 5/sig_check.py, 22 Oct, deck at ~62 px: the same
 * card in two frames 0-15 (p95 7), a joker in its place 19-76 (p5 24). If the
 * board re-reads still cards every find (UART: reads with nothing new), the
 * real frames are noisier: raise it. */
#define SORT_SIG_DIFF           18

/* M3 fast read (app_rtos.c prvReadPolicy): the first two windows must agree,
 * each with at least this rank logit, before the vote stops early.
 * lab 5/read_policy_check.py, 22 Oct (photos + deck at ~62 px, 378 cards):
 * full vote 378/378 at 6.0 reads a card; fast 378/378 at 2.6 with 0, 3.3 /
 * 3.8 / 4.4 reads with 1000 / 1500 / 2000 and no card gained. */
#define READ_LOGIT_MIN          0

int rtos_main( void );

#endif /* APP_CONFIG_H */
