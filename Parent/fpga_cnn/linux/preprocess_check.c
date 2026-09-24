/*
 * Self-check for the fixed-point front end in card_cnn_hw.h.
 *
 * The standing correctness argument for this project is that the board and
 * sim_card_cnn.py agree on the same capture. The Linux app is a third
 * implementation of that pipeline, so it has to join the agreement rather than
 * be assumed to.
 *
 * testdata/ was produced from capture.txt -- a real Arm DS console dump from
 * the board -- by sim_card_cnn.py's own parser and crop:
 *   capture_raw.txt  the 48x48 raw cell sums the board read
 *                    (the PGM carries raw*255/6375 and 6375/255 == 25 exactly,
 *                    so raw == pgm*25 is a lossless reconstruction)
 *   capture_q.txt    the Q6.10 values the simulator uploads for that capture,
 *                    zoom window (9,0,29,48)
 *
 * This runs ccp_preprocess() over capture_raw.txt and requires every one of
 * the 2304 outputs to match capture_q.txt exactly. A mismatch means the C and
 * the model have drifted -- most likely someone changed ZOOM_* on one side
 * only, which is exactly the failure this is here to catch.
 *
 *   make check
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "card_cnn_hw.h"

static int read_ints(const char *path, long *out, int n)
{
    FILE *f = fopen(path, "r");
    int i;

    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return -1;
    }
    for (i = 0; i < n; i++) {
        if (fscanf(f, "%ld", &out[i]) != 1) {
            fprintf(stderr, "%s: only %d of %d values\n", path, i, n);
            fclose(f);
            return -1;
        }
    }
    fclose(f);
    return 0;
}

int main(void)
{
    static long     raw_in[IMG_PIXELS], q_ref[IMG_PIXELS];
    static uint16_t raw[IMG_PIXELS], q[IMG_PIXELS];
    int bad = 0, i;

    if (read_ints("testdata/capture_raw.txt", raw_in, IMG_PIXELS) != 0 ||
        read_ints("testdata/capture_q.txt",   q_ref,  IMG_PIXELS) != 0)
        return 1;

    for (i = 0; i < IMG_PIXELS; i++) {
        if (raw_in[i] < 0 || raw_in[i] > (long)SUM_MAX) {
            fprintf(stderr, "capture_raw.txt[%d] = %ld out of 0..%u\n",
                    i, raw_in[i], SUM_MAX);
            return 1;
        }
        raw[i] = (uint16_t)raw_in[i];
    }

    ccp_preprocess(raw, q);

    for (i = 0; i < IMG_PIXELS; i++) {
        if ((long)q[i] != q_ref[i]) {
            if (bad < 8)
                fprintf(stderr, "  mismatch at %4d (row %2d col %2d): "
                        "got %4u, model says %4ld\n",
                        i, i / IMG_DIM, i % IMG_DIM, q[i], q_ref[i]);
            bad++;
        }
    }

    printf("zoom window x=%d y=%d w=%d h=%d\n", ZOOM_X, ZOOM_Y, ZOOM_W, ZOOM_H);

    if (bad) {
        printf("FAIL: %d of %d pixels differ from sim_card_cnn.py\n",
               bad, IMG_PIXELS);
        printf("      If ZOOM_* was retuned, regenerate testdata/ with the\n"
               "      same window and re-run.\n");
        return 1;
    }

    printf("PASS: all %d Q6.10 pixels match sim_card_cnn.py "
           "on the board capture\n", IMG_PIXELS);
    return 0;
}
