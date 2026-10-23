/*
 * Card finder for the M2 whole-grid scan -- a line-by-line port of
 * lab 5/locate.py, the reference, which explains every step and was tuned on
 * the 28 Sep photos of real layouts. It reads the frame through snap_read()
 * (on CPU1, during a VJOB_LOCATE) and uses no printf and no RTOS, so a host
 * test can link it against an array. locate_host_test.c checks it against
 * locate.py card for card.
 */
#ifndef LOCATE_H
#define LOCATE_H

#include "card_pipeline.h"

/* Find the face-up cards in the fw x fh frame (normally FRAME_W x FRAME_H) and
 * number them by the professor's (row, col) under `orient` (ORIENT_* bits).
 * Also decides each card's colour from its index corners. Returns out->n. */
unsigned locate_cards( struct scan_result * out, unsigned orient, int fw, int fh );

/* conv1's window for one card: a square 1.15 x its longer side, centred, with
 * the clip on the card plus 4 % so neighbours read black, and turned upright
 * if it lies sideways. */
void window_for_card( const struct scan_card * c, unsigned orient, int fw, int fh,
                      struct win_regs * w );

#endif /* LOCATE_H */
