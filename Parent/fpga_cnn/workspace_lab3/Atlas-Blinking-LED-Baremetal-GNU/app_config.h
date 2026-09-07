/*
 * Which application this build produces.
 *
 * The single-threaded polling app in atlas_main.c is the one that was tuned on
 * the bench and reads the deck 54/54, so it stays the default and stays
 * reachable. The RTOS/two-core build is layered on top of it rather than
 * replacing it: it calls the same pipeline stages, so a regression in the
 * scheduler or in the CPU1 handshake cannot quietly change what the network is
 * being shown.
 *
 *   RTOS_MODE 0   atlas_main.c exactly as before -- FreeRTOS is not even linked
 *   RTOS_MODE 1   FreeRTOS on CPU0, three tasks, preprocessing on CPU1
 */
#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#define RTOS_MODE   1

/* Release the second Cortex-A9 and hand it the preprocessing stage.
 *
 * With this at 0 -- or if CPU1 fails to check in at run time -- the same work
 * is done on CPU0 and the app still runs, just without the overlap. Bringing a
 * second core out of reset is the part of this design most likely to fail on a
 * given board, and it is not allowed to take the demo down with it. */
#define USE_CORE1   1

/* How long CPU0 waits for CPU1 to finish a frame before giving up on it and
 * doing the work itself. Preprocessing measures in single-digit milliseconds,
 * so this is generous by two orders of magnitude. */
#define CORE1_FRAME_TIMEOUT_US  100000UL

/* Print the FreeRTOS run-time task statistics every N inferences. This is the
 * scheduling evidence: per-task CPU time straight out of the kernel. 0 = off. */
#define STATS_EVERY 10

int rtos_main( void );

#endif /* APP_CONFIG_H */
