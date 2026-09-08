/* ---------------------------------------------------------------------------
 * snapshot_to_accel.c  --  glue between downsample_96x96.v and the accelerator
 *
 * The hardware and the model do not speak the same units. This bridges them.
 *
 *   downsample_96x96.v gives                 the accelerator wants
 *   ----------------------------------       -------------------------------
 *   9216 cell sums (96 x 96)                 2304 pixels (48 x 48)
 *   each cell = 2 source pixels summed       Q6.10, white = 1024
 *   as (r+g+b), so 0..510 per cell           signed 16-bit
 *   + red pixel count at address 9216
 *   + colour flag at address 9217
 *
 * WHY 96x96 IS AVERAGED DOWN RATHER THAN USED DIRECTLY
 *   A 96x96 model does not fit. The flatten becomes 9,216, so fc_shared needs
 *   589,888 weights = 1,153 M10K blocks against the device's 553. Getting
 *   there means a third conv+pool stage (96 -> 48 -> 24 -> 12) to bring the
 *   flatten back to 2,304, which also takes inference from ~50 ms to ~258 ms.
 *
 *   Averaging 2x2 in software costs nothing and is still a real improvement:
 *   the previous path kept 29 genuine columns and nearest-neighbour stretched
 *   them to 48, so 40% of the network's horizontal samples carried no new
 *   information. Now every column is a real measurement.
 *
 * THE COLOUR FLAG IS NOT INTERCHANGEABLE
 *   colour_hw counts pixels whose redness exceeds RED_THRESH. The software
 *   path takes the 98th percentile of a white-balanced chromaticity ratio and
 *   compares against 99. These are different statistics -- the threshold does
 *   NOT carry across. Calibrate RED_THRESH separately: print red_count for a
 *   handful of known red and known black cards and pick a value in the gap.
 * ------------------------------------------------------------------------- */

#include <stdint.h>

#define HW_DIM      96
#define MODEL_DIM   48
#define CELL_MAX    510      /* (r+g+b) summed over the 2 source pixels in a cell */
#define GROUP_MAX   (CELL_MAX * 4)   /* 2x2 cells = 4 cells = 8 source pixels */
#define Q_ONE       1024     /* 1.0 in Q6.10 */

#define ADDR_RED_COUNT   9216
#define ADDR_COLOUR_HW   9217

/* Supplied by the caller: reads one word from the snapshot buffer, whatever
 * the PIO or bridge arrangement happens to be. */
typedef uint16_t (*snapshot_read_fn)(uint32_t addr);

/* Reads the 96x96 cell sums, averages each 2x2 block, scales to Q6.10, and
 * writes 2304 pixels into `out` in the accelerator's row-major order.
 *
 * Returns the hardware colour flag so the caller can pass it through
 * CONTROL bit 2. */
int snapshot_to_accel_image(snapshot_read_fn rd, int16_t *out /* [2304] */)
{
    for (int y = 0; y < MODEL_DIM; y++) {
        for (int x = 0; x < MODEL_DIM; x++) {
            /* four neighbouring cells in the 96x96 grid */
            uint32_t a = rd((2*y    ) * HW_DIM + (2*x    ));
            uint32_t b = rd((2*y    ) * HW_DIM + (2*x + 1));
            uint32_t c = rd((2*y + 1) * HW_DIM + (2*x    ));
            uint32_t d = rd((2*y + 1) * HW_DIM + (2*x + 1));
            uint32_t sum = a + b + c + d;          /* 0 .. 2040 */

            /* Scale to Q6.10. Integer maths only -- the HPS has an FPU but
             * this runs 2304 times per frame and there is no reason to use it.
             *
             * NOT INVERTED. The lab's downsample_28x28 computes
             * (255*3) - (r+g+b) because MNIST is a white digit on black. This
             * model was trained on photographs of cards: white card, dark ink.
             * Inverting here feeds the network the negative of everything it
             * learned, and it fails completely while looking like a wiring
             * fault. */
            uint32_t q = (sum * Q_ONE) / GROUP_MAX;
            if (q > Q_ONE) q = Q_ONE;              /* clamp, should not trigger */

            out[y * MODEL_DIM + x] = (int16_t)q;
        }
    }

    return rd(ADDR_COLOUR_HW) & 1;
}

/* Prints red_count so RED_THRESH can be calibrated. Run it over several known
 * red cards and several known black ones, then set RED_THRESH in
 * downsample_96x96.v to a value in the gap -- the MIDPOINT, not the edge, so
 * there is margin on both sides. */
uint16_t snapshot_red_count(snapshot_read_fn rd)
{
    return rd(ADDR_RED_COUNT);
}
