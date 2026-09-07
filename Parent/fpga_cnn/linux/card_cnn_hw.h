/*
 * DE10-Nano card-CNN accelerator: register map and the host-side arithmetic
 * that has to agree with it.
 *
 * Everything here is a transcription of the values proven on the board by
 * atlas_main.c (bare metal) and sim_card_cnn.py (offline model). Those three
 * must stay in step: the standing correctness argument for the accelerator is
 * that the board and the simulator produce the same answer for the same
 * capture, and this header is what lets the Linux app join that agreement.
 *
 * If you retune ZOOM_* on the bench, change it here too, or the Linux app and
 * the bare-metal app will disagree about what they are looking at.
 */
#ifndef CARD_CNN_HW_H
#define CARD_CNN_HW_H

#include <stdint.h>

/* ---- lightweight HPS-to-FPGA bridge ------------------------------------ */
#define LWBRIDGE_BASE        0xFF200000UL
#define LWBRIDGE_SPAN        0x1000UL         /* the PIOs all live in the first 4K */

#define REG_IMG_WR_CTRL      0x00   /* [0] wr_en, [2] colour override en, [3] colour value */
#define REG_IMG_WR_DATA      0x10   /* Q6.10 pixel */
#define REG_IMG_WR_ADDR      0x20   /* 0..9215 = y*96 + x */
#define REG_CAMERA_TRIGGER   0x30
#define REG_SNAPSHOT_DATA    0x40   /* raw 5x5 luminance sum at snapshot_addr */
#define REG_SNAPSHOT_ADDR    0x50
#define REG_CNN_START        0x60
#define REG_CNN_RESULT       0x70

/* Reading past the image cells returns camera telemetry instead. Expressed as
 * offsets from IMG_PIXELS rather than literals: these addresses moved when the
 * capture went 48x48 -> 96x96, and a literal left behind in any one of the
 * three host copies reads image data as telemetry, which looks exactly like a
 * dead camera link. */
#define SNAP_RED_COUNT       ((unsigned)IMG_PIXELS + 0u)
#define SNAP_COLOUR_HW       ((unsigned)IMG_PIXELS + 1u)
#define SNAP_VS_COUNT        ((unsigned)IMG_PIXELS + 2u)   /* MIPI frame counter -- the one that matters */
#define SNAP_PIXCLK          ((unsigned)IMG_PIXELS + 3u)
#define SNAP_RETRIES         ((unsigned)IMG_PIXELS + 4u)
#define SNAP_CFG_STEP        ((unsigned)IMG_PIXELS + 5u)
#define SNAP_STATUS          ((unsigned)IMG_PIXELS + 6u)

#define ST_MIPI_REL(s)       ((s) & 1u)
#define ST_CAM_REL(s)        (((s) >> 1) & 1u)
#define ST_AUDPLL_OK(s)      (((s) >> 2) & 1u)
#define ST_HDMI_RDY(s)       (((s) >> 3) & 1u)

/* ---- cnn_result_pio bit layout ----------------------------------------- */
#define RES_RANK(r)          ((r) & 0xFu)
#define RES_SUIT(r)          (((r) >> 4) & 0x3u)
#define RES_JOKER(r)         (((r) >> 6) & 0x1u)
#define RES_DONE(r)          (((r) >> 7) & 0x1u)
#define RES_COLOUR(r)        (((r) >> 8) & 0x1u)
#define RES_SNAP_DONE(r)     (((r) >> 9) & 0x1u)
#define RES_SCORE(r)         ((int16_t)((r) >> 16))

/* ---- image geometry and fixed point ------------------------------------ */
#define IMG_DIM              96
#define IMG_PIXELS           (IMG_DIM * IMG_DIM)   /* 9216 */
#define SUM_MAX              510u    /* 2 px * 255: ITU-R 601 luminance of white */
#define Q_ONE                1024u   /* 1.0 in Q6.10 */

/* ---- software digital zoom ---------------------------------------------
 * IDENTITY, and it should stay that way. downsample_96x96.v crops 96 x 192
 * buffer pixels into 96x96 cells of 1x2, so the hardware already delivers the
 * model's aspect and every column is a real sample. The 48x48 build cropped a
 * square and trimmed it here (ZOOM_W 29 of 48), which threw away 40% of the
 * horizontal samples by stretching 29 columns back to 48.
 *
 * The defines survive so the board's "zoom window:" line still parses in
 * sim_card_cnn.py, and as the escape hatch if a rig ever needs a sub-crop. */
#define ZOOM_X               0
#define ZOOM_Y               0
#define ZOOM_W               96
#define ZOOM_H               96

/* Software colour decision. The fabric flag trips at ~1% red pixels, which a
 * red-and-gold face card can reach on its own; this threshold is applied to
 * the reported red_count instead.
 *
 * RECALIBRATE. 5000 was measured on the 240x240 crop; the crop is now
 * 96 x 192 = 18,432 pixels, 3.1x smaller, so the same card yields ~1600.
 * Read the printed red_count for known red and black cards at your framing and
 * put this between the two populations -- a wrong colour flag locks the suit
 * argmax to the wrong pair before the network gets a say. */
#define RED_THRESH_SW        1600u

#define RANK_COUNT           13
#define SUIT_COUNT           4

static const char *const RANK_NAMES[RANK_COUNT] =
    { "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A" };
static const char *const SUIT_NAMES[SUIT_COUNT] =
    { "Spades", "Clubs", "Hearts", "Diamonds" };

/*
 * Crop to the zoom window and rescale the raw cell sums to Q6.10.
 *
 * Byte-for-byte the same arithmetic as atlas_main.c preprocess() with
 * NORMALIZE_MINMAX 0 and INVERT 0, and as sim_card_cnn.py apply_zoom() followed
 * by `q = min(p * 1024 // 255, 1024)`. Integer division truncates in all three,
 * which is why this is written out rather than "cleaned up" into floats.
 */
static inline void ccp_preprocess(const uint16_t *raw, uint16_t *q)
{
    int i, j;

    for (i = 0; i < IMG_DIM; i++) {
        unsigned sy = ZOOM_Y + (unsigned)i * ZOOM_H / IMG_DIM;
        for (j = 0; j < IMG_DIM; j++) {
            unsigned sx = ZOOM_X + (unsigned)j * ZOOM_W / IMG_DIM;
            unsigned v  = ((uint32_t)raw[sy * IMG_DIM + sx] * Q_ONE) / SUM_MAX;
            if (v > Q_ONE)
                v = Q_ONE;
            q[i * IMG_DIM + j] = (uint16_t)v;
        }
    }
}

#endif /* CARD_CNN_HW_H */
