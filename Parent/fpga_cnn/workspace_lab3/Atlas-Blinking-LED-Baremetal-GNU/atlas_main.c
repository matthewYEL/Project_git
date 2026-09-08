/*
 * Card recognition demo: friend's 3-head CNN (rank / suit / joker) on the FPGA.
 *
 * Flow: trigger the D8M snapshot -> wait for snapshot_done -> read the 96x96
 * luminance box sums plus the red pixel count -> ASCII preview (and optional
 * PGM dump) -> scale to Q6.10 -> upload to the CNN -> start -> poll done ->
 * decode rank/suit/joker.
 *
 * FRAMING IS THE WHOLE BALL GAME. The model was trained on the card's rank/suit
 * INDEX CORNER, not the whole card -- specifically the top-left 25% x 35.7% of a
 * perspective-corrected 250x350 card (friend's build_warped_dataset.py
 * corner_crops()), squashed to 96x96. Feed it the corner and it reads my_deck
 * 54/54; feed it the whole card and it drops to 4/54, near chance.
 *
 * So aim that corner to FILL the green box on the HDMI output -- roughly
 * 16 x 26 mm of card, about 4x closer than framing the whole card. The box is
 * now a tall rectangle carrying the model's aspect in hardware, so ZOOM_* is
 * identity and there is nothing per-rig left to tune. Tolerance is still tight:
 * sliding 5% of a card width off the corner costs about 75 points of accuracy.
 *
 * PIO base addresses are the lab's; only widths and bit layouts changed
 * (see ghrd_top.v). printf goes to the Arm DS App Console (semihosting).
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "socal.h"
#include "hps_0_arm_a9_0.h"

#include "app_config.h"
#include "card_pipeline.h"

int __auto_semihosting;

#define IMG_WR_CTRL_PIO_BASE      (0xFF200000)   /* [0] wr_en, [2] colour override en, [3] colour override val */
#define IMG_WR_DATA_PIO_BASE      (0xFF200010)   /* Q6.10 pixel */
#define IMG_WR_ADDR_PIO_BASE      (0xFF200020)   /* 0..2303 = y*48 + x (model grid, not the capture grid) */
#define CAMERA_TRIGGER_PIO_BASE   (0xFF200030)
#define SNAPSHOT_DATA_PIO_BASE    (0xFF200040)   /* raw 1x2 luminance sum; addr 9216 = red_count, 9217 = colour_hw */

/* Snapshot telemetry addresses and status bits moved to card_pipeline.h,
 * so app_rtos.c can read the same diagnostics. */
#define SNAPSHOT_ADDR_PIO_BASE    (0xFF200050)
#define CNN_START_PIO_BASE        (0xFF200060)
#define CNN_RESULT_PIO_BASE       (0xFF200070)

/* cnn_result bit layout, image geometry and the pipeline stage
 * prototypes now live in card_pipeline.h, so amp.c and app_rtos.c
 * share this one definition of them. */

/* Experiment switches - no FPGA recompile needed.
 * Defaults match the training pipeline: PIL convert("L") -> ToTensor() (/255), no inversion. */
#define INVERT            0   /* 1: ink bright / background dark */
#define NORMALIZE_MINMAX  0   /* 0: fixed /255-equivalent scale; 1: stretch min..max to 0..1024 */
#define COLOUR_OVERRIDE   1   /* 1: software decides red/black from red_count instead of the hardware flag */
#define RED_THRESH_SW     1600u
#define DUMP_PGM          1   /* 1: also print the capture as an ASCII PGM (P2) for recognize_card.py */

/* LIVE 1 loops forever printing one line per inference, so you can watch what
 * the board sees without HDMI. LIVE 0 is the original single verbose run and is
 * what feeds sim_card_cnn.py -- keep it working, sim/board agreement is the
 * standing proof the accelerator is correct.
 *
 * PREVIEW_EVERY exists because semihosting putchar is slow: a full 96x96 dump is
 * ~9200 characters and would dominate the loop even at 261 ms per inference.
 * Set it to 10 or so while aiming, back to 0 once framed. The preview is
 * quarter resolution (24x24) for the same reason -- a sixteenth of the
 * characters, and the same size the 48x48 build previewed at. */
#define LIVE              1
#define PREVIEW_EVERY     0

/* The 9218..9223 diagnostic registers only exist once camera_capture.v has
 * been recompiled into the .sof. On an older bitstream they read back as 0,
 * which is indistinguishable from a genuinely dead link -- so gate the
 * report rather than print a confident wrong verdict. Set to 1 after the
 * Quartus compile. The BLANK FRAME warning works either way, since it is
 * derived from the snapshot itself. */
#define HAVE_CAM_DIAG     1

/* The hardware flag trips at red_count > RED_THRESH (192 in downsample_96x96.v),
 * which is ~1% of the 96x192 crop. A face card's illustration is red and gold,
 * and the red flag locks the suit argmax to {H,D} (card_cnn_core.v) before the
 * net gets a say, so the threshold matters as much as the network.
 *
 * RECALIBRATE. The old numbers -- 2048 for a Queen of Spades, 10458 for a
 * genuine red card, 5000 between them -- were measured on the 240x240 crop with
 * whole cards in frame. The crop is now 18,432 pixels, 3.1x smaller, so the
 * equivalent scaling is ~1600, which is what RED_THRESH_SW ships as. Measure
 * your own: the log prints red_count on every line.
 *
 * PROJECT_REPORT.md section 10 gives the hardware equivalent of the Python
 * colour test directly: count pixels over the threshold and compare against 2%
 * of the region -- ~370 of the 18,432 pixels in the crop, against the 192
 * (1.04%) the fabric uses.
 *
 * Their detector went through seven versions (report section 6). Two of its
 * conclusions the fabric version already satisfies: a brightness floor to stop
 * near-black pixels reading as saturated red (RED_MIN 64 here, theirs 30), and
 * cropping to the index corner so face-card artwork is out of frame -- that
 * single change took them 93% -> 99%.
 *
 * The one it does NOT satisfy is per-image white balance. The D8M's greenish
 * cast makes `R > 1.5G` harder to satisfy, so once the artwork is out of frame
 * the failure flips direction: expect red cards to read LOW, not black cards
 * high. If that happens it is the cast, not the threshold -- measure with
 * friends files/trainings/calibrate_color_threshold.py. */

/* Software digital zoom: upload a sub-rectangle of the capture, nearest-
 * neighbour scaled back to IMG_DIM. Units are capture cells.
 *
 * IDENTITY NOW, AND IT SHOULD STAY THAT WAY. downsample_96x96.v crops
 * 96 x 192 buffer pixels into 96x96 cells of 1x2, so the hardware already
 * delivers the 0.5 aspect the retrained model wants and every column is a real
 * sample. The 48x48 build could not: its crop was square, so this trimmed it
 * to ZOOM_W 29 of 48 and stretched 29 columns back to 48, throwing away 40% of
 * the horizontal samples.
 *
 * AIM THE CORNER TO FILL THE BOX -- the green box is now a tall rectangle that
 * matches the crop, so what you see is what the network gets. Do not use these
 * to magnify a distant card: re-magnifying resamples detail the camera pipeline
 * already threw away.
 *
 * Kept as an escape hatch, and so the "zoom window:" line the board prints
 * still parses in sim_card_cnn.py. Verify with: python sim_card_cnn.py <paste> */
#define ZOOM_X   0
#define ZOOM_Y   0
#define ZOOM_W  96
#define ZOOM_H  96

#if (ZOOM_X + ZOOM_W > IMG_DIM) || (ZOOM_Y + ZOOM_H > IMG_DIM) || (ZOOM_W < 1) || (ZOOM_H < 1)
#error "zoom window falls outside the capture"
#endif

static const char *const RANK_NAMES[13] = {"2","3","4","5","6","7","8","9","10","J","Q","K","A"};
static const char *const SUIT_NAMES[4]  = {"Spades","Clubs","Hearts","Diamonds"};

static void delay(volatile int n) { while (n--) ; }

unsigned snap_read(unsigned addr)
{
    alt_write_word(SNAPSHOT_ADDR_PIO_BASE, addr);
    (void)alt_read_word(SNAPSHOT_DATA_PIO_BASE);          /* registered RAM read: one dummy access */
    return alt_read_word(SNAPSHOT_DATA_PIO_BASE) & 0xFFFFu;
}

/* ---- pipeline stages, shared by the live loop and the one-shot path ---- */

int capture_once(void)
{
    int i;
    unsigned r = 0;

    alt_write_word(CAMERA_TRIGGER_PIO_BASE, 1);
    delay(200);
    alt_write_word(CAMERA_TRIGGER_PIO_BASE, 0);

    /* snapshot_done is a level left high by the previous capture: wait for it to
     * drop (trigger accepted), then rise again (next frame captured). */
    for (i = 0; i < 100000 && RES_SNAP_DONE(alt_read_word(CNN_RESULT_PIO_BASE)); i++) ;
    for (i = 0; i < 4000000; i++) {
        r = alt_read_word(CNN_RESULT_PIO_BASE);
        if (RES_SNAP_DONE(r)) return 1;
    }
    return 0;
}

void read_snapshot(uint16_t *raw, unsigned *minv, unsigned *maxv)
{
    int i;
    unsigned lo = 0xFFFFu, hi = 0;
    for (i = 0; i < IMG_PIXELS; i++) {
        raw[i] = (uint16_t)snap_read((unsigned)i);
        if (raw[i] < lo) lo = raw[i];
        if (raw[i] > hi) hi = raw[i];
    }
    *minv = lo;
    *maxv = hi;
}

/* Crop to the zoom window and rescale to Q6.10. See the ZOOM_* notes above: the
 * window's job is to trim the square capture to the 0.6 aspect the model was
 * trained on, not to magnify. */
void preprocess(const uint16_t *raw, uint16_t *q)
{
    int i, j;

    /* min/max for the stretch must come from the window that is actually
     * uploaded, not the whole frame -- otherwise zooming past the darkest cell
     * silently changes the scale. */
    unsigned zmin = 0xFFFFu, zmax = 0;
    for (i = ZOOM_Y; i < ZOOM_Y + ZOOM_H; i++) {
        for (j = ZOOM_X; j < ZOOM_X + ZOOM_W; j++) {
            unsigned v = raw[i * IMG_DIM + j];
            if (v < zmin) zmin = v;
            if (v > zmax) zmax = v;
        }
    }

    for (i = 0; i < IMG_DIM; i++) {
        unsigned sy = ZOOM_Y + (unsigned)i * ZOOM_H / IMG_DIM;
        for (j = 0; j < IMG_DIM; j++) {
            unsigned sx = ZOOM_X + (unsigned)j * ZOOM_W / IMG_DIM;
            unsigned src = raw[sy * IMG_DIM + sx];
            unsigned v;
#if NORMALIZE_MINMAX
            unsigned range = (zmax > zmin) ? (zmax - zmin) : 1u;
            v = ((uint32_t)(src - zmin) * Q_ONE) / range;
#else
            v = ((uint32_t)src * Q_ONE) / SUM_MAX;         /* = luminance/255 * 1024 */
#endif
            if (v > Q_ONE) v = Q_ONE;
#if INVERT
            v = Q_ONE - v;
#endif
            q[i * IMG_DIM + j] = (uint16_t)v;
        }
    }
#if !NORMALIZE_MINMAX
    (void)zmin; (void)zmax;
#endif
}

/* Returns the raw result word, or 0 on timeout. Bit 7 (done) is always set in a
 * real result, so 0 is unambiguous. */
unsigned upload_and_infer(const uint16_t *q, unsigned red_count)
{
    int i;
    unsigned r = 0;
    unsigned ctrl_base = 0;
#if COLOUR_OVERRIDE
    ctrl_base = 0x4u | ((red_count > RED_THRESH_SW) ? 0x8u : 0x0u);
#else
    (void)red_count;
#endif

    /* 96x96 capture -> 48x48 model input, averaging each 2x2 block.
     *
     * The accelerator is the handover_96_v2 network, which takes 48x48. The
     * hardware still captures 96x96 at the 1:2 cell aspect, so a plain 2x2 mean
     * lands exactly on the "1:2 crop squashed square" the model was trained on
     * -- no resampling, every output pixel backed by four real samples. This is
     * what board/snapshot_to_accel.c does in the handover bundle; done here on
     * the Q6.10 values instead of the raw sums, which is equivalent (the scale
     * is linear) and keeps INVERT / NORMALIZE_MINMAX applied first.
     *
     * The rounding term matters: truncating four times per pixel biases the
     * whole image dark by up to 1.5 LSB, and the network is sensitive to it. */
    for (i = 0; i < MODEL_PIXELS; i++) {
        unsigned oy = (unsigned)i / MODEL_DIM;
        unsigned ox = (unsigned)i % MODEL_DIM;
        const uint16_t *p = &q[(oy * 2u) * IMG_DIM + (ox * 2u)];
        unsigned avg = ((unsigned)p[0] + p[1] + p[IMG_DIM] + p[IMG_DIM + 1] + 2u) >> 2;

        alt_write_word(IMG_WR_ADDR_PIO_BASE, (unsigned)i);
        alt_write_word(IMG_WR_DATA_PIO_BASE, avg);
        alt_write_word(IMG_WR_CTRL_PIO_BASE, ctrl_base | 0x1u);   /* wr_en high for >= 1 clock */
        alt_write_word(IMG_WR_CTRL_PIO_BASE, ctrl_base);
    }

    alt_write_word(CNN_START_PIO_BASE, 1);   /* hardware edge-detects start, clears sticky done */
    delay(50);
    alt_write_word(CNN_START_PIO_BASE, 0);
    for (i = 0; i < 20000000; i++) {
        r = alt_read_word(CNN_RESULT_PIO_BASE);
        if (RES_DONE(r)) return r;
    }
    return 0;
}

/* step 1 = full 96x96, step 4 = 24x24. Same character ramp either way, so a
 * preview can be compared straight against a reference corner crop. */
static void print_preview(const uint16_t *raw, int step)
{
    int i, j;
    for (i = 0; i < IMG_DIM; i += step) {
        for (j = 0; j < IMG_DIM; j += step) {
            unsigned v = raw[i * IMG_DIM + j];
            putchar((v < SUM_MAX / 5)     ? '@' :
                    (v < 2 * SUM_MAX / 5) ? '#' :
                    (v < 3 * SUM_MAX / 5) ? '+' :
                    (v < 4 * SUM_MAX / 5) ? '-' : ' ');
        }
        putchar('\n');
    }
}

void print_result_name(unsigned r)
{
    unsigned rank = RES_RANK(r), suit = RES_SUIT(r);
    if (RES_JOKER(r))    printf("%-14s", "JOKER");
    else if (rank < 13)  printf("%-3s of %-9s", RANK_NAMES[rank], SUIT_NAMES[suit]);
    else                 printf("bad rank %-5u", rank);
}

#if HAVE_CAM_DIAG
static void print_camera_diag(void)
{
    unsigned st = snap_read(SNAP_STATUS);
    /* SNAP_CFG_STEP is not printed: MIPI_BRIDGE_CAMERA_Config declares STEP
     * but its .STEP connection is commented out inside that wrapper, so the
     * port is undriven and always reads 0. Use WCNT instead if it is ever
     * brought out. */
    printf("camera: mipi_cfg=%u cam_cfg=%u  vs=%u pixclk=%u retries=%u"
           "   (audpll=%u hdmi=%u)\n",
           ST_MIPI_REL(st), ST_CAM_REL(st),
           snap_read(SNAP_VS_COUNT), snap_read(SNAP_PIXCLK),
           snap_read(SNAP_RETRIES),
           ST_AUDPLL_OK(st), ST_HDMI_RDY(st));
}
#endif

/* The camera is the thing that has actually been failing, so say so plainly
 * rather than letting a blank frame come back as a confident card. */
void check_camera_alive(void)
{
#if !HAVE_CAM_DIAG
    printf("camera diagnostics absent from this bitstream -- recompile\n"
           "camera_capture.v and set HAVE_CAM_DIAG to read vs/pixclk.\n");
#else
    unsigned vs0, vs1, pix0, pix1;
    volatile int i;

    print_camera_diag();

    vs0  = snap_read(SNAP_VS_COUNT);
    pix0 = snap_read(SNAP_PIXCLK);
    for (i = 0; i < 2000000; i++) ;              /* short settle, tens of ms */
    vs1  = snap_read(SNAP_VS_COUNT);
    pix1 = snap_read(SNAP_PIXCLK);

    if (vs1 == vs0) {
        printf("\n*** WARNING: no MIPI frames (vs_count stuck at %u) ***\n", vs1);
        /* The bridge only emits MIPI_PIXEL_CLK once the sensor is streaming,
         * and the camera LUT ends with "wake up, streaming" -- so pixclk=0
         * with cam_cfg=0 is a CONSEQUENCE of the config never finishing, not
         * independent evidence of a dead link. Only call the link dead when
         * the config actually completed. */
        if (pix1 == pix0 && !ST_CAM_REL(snap_read(SNAP_STATUS)))
            printf("    camera I2C config never completes, so the sensor never\n"
                   "    streams and the bridge has nothing to clock out.\n"
                   "    Note the bridge bus IS fine (mipi_cfg=1): this is the\n"
                   "    separate CAMERA_I2C bus or the sensor itself.\n");
        else if (pix1 == pix0)
            printf("    config completed but no pixel clock -> link down: reseat\n"
                   "    the D8M; the module or cable is at fault.\n");
        else
            printf("    pixel clock runs but no frames -> bridge clocks without\n"
                   "    framing; wrong lane count or format.\n");
        printf("    Every prediction below is the network's answer to a blank image.\n\n");
    }
#endif
}

#if RTOS_MODE

/* FreeRTOS build: the pipeline is driven by the tasks in app_rtos.c, which
 * call the same stages above, and the preprocessing stage runs on CPU1.
 * Set RTOS_MODE to 0 in app_config.h to get the original single-threaded
 * polling app back. */
int main(void)
{
    return rtos_main();
}

#else

int main(void)
{
    static uint16_t raw[IMG_PIXELS];
    static uint16_t q[IMG_PIXELS];
    unsigned minv, maxv, red_count, colour_hw, r;

    check_camera_alive();

#if LIVE
    {
        unsigned frame = 0;
        unsigned prev_vs = snap_read(SNAP_VS_COUNT);
        (void)prev_vs;

        printf("live inference -- zoom x=%u y=%u w=%u h=%u  (stop the debugger to exit)\n\n",
               (unsigned)ZOOM_X, (unsigned)ZOOM_Y, (unsigned)ZOOM_W, (unsigned)ZOOM_H);

        for (;;) {
            unsigned vs;

            if (!capture_once()) {
                printf("[%5u] snapshot timeout - camera not streaming\n", frame++);
                continue;
            }
            read_snapshot(raw, &minv, &maxv);
            red_count = snap_read(SNAP_RED_COUNT);
            colour_hw = snap_read(SNAP_COLOUR_HW) & 1u;
            vs        = snap_read(SNAP_VS_COUNT);
            (void)vs;
            (void)colour_hw;

            preprocess(raw, q);
            r = upload_and_infer(q, red_count);

            printf("[%5u] ", frame);
            if (!r) {
                printf("inference timeout\n");
            } else {
                print_result_name(r);
                printf("  logit %5d  red %5u->%-5s  min %4u max %4u",
                       RES_SCORE(r), red_count, RES_COLOUR(r) ? "red" : "black",
                       minv, maxv);
#if HAVE_CAM_DIAG
                printf("  vs+%u", (unsigned)(vs - prev_vs));
#endif
                if (maxv == 0) printf("   <- BLANK FRAME");
                putchar('\n');
            }
            prev_vs = vs;

#if PREVIEW_EVERY
            if ((frame % PREVIEW_EVERY) == 0) {
                print_preview(raw, 4);
                putchar('\n');
            }
#endif
            frame++;
        }
    }
#else
    {
        int i;

        printf("Triggering camera capture...\n");
        if (!capture_once()) {
            printf("ERROR: snapshot_done timeout (camera not streaming?)\n");
            return 1;
        }

        printf("Reading 96x96 snapshot...\n");
        read_snapshot(raw, &minv, &maxv);
        red_count = snap_read(SNAP_RED_COUNT);
        colour_hw = snap_read(SNAP_COLOUR_HW) & 1u;

        printf("\nCaptured 96x96 (raw luminance sums)  min %u  max %u  red_count %u  colour_hw %s\n",
               minv, maxv, red_count, colour_hw ? "RED" : "black");
        print_preview(raw, 1);
        putchar('\n');

#if DUMP_PGM
        /* Paste from "P2" to the last number into a file and run it through
         * sim_card_cnn.py: if the sim agrees with the board, the accelerator is
         * faithful and any error is upstream of it. */
        printf("--- PGM begin ---\nP2\n%d %d\n255\n", IMG_DIM, IMG_DIM);
        for (i = 0; i < IMG_PIXELS; i++) {
            unsigned v8 = ((uint32_t)raw[i] * 255u) / SUM_MAX;
            if (v8 > 255u) v8 = 255u;
            printf("%u%c", v8, ((i % IMG_DIM) == IMG_DIM - 1) ? '\n' : ' ');
        }
        printf("--- PGM end ---\n\n");
#endif

        printf("zoom window: x=%u y=%u w=%u h=%u\n",
               (unsigned)ZOOM_X, (unsigned)ZOOM_Y, (unsigned)ZOOM_W, (unsigned)ZOOM_H);
        preprocess(raw, q);

        printf("Loading image into CNN...\nRunning inference...\n");
        r = upload_and_infer(q, red_count);
        if (!r) { printf("ERROR: inference timeout\n"); return 1; }

        printf("\n=== Predicted: ");
        print_result_name(r);
        printf(" ===\n");
        printf("    rank logit %d/1024 (Q6.10)  colour used: %s  red_count %u  raw result 0x%08x\n",
               RES_SCORE(r), RES_COLOUR(r) ? "red" : "black", red_count, r);
        return 0;
    }
#endif
}

#endif /* RTOS_MODE */
