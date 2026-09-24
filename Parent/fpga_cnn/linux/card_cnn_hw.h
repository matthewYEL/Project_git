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
#define REG_IMG_WR_ADDR      0x20   /* 0..2303 = y*48 + x */
#define REG_CAMERA_TRIGGER   0x30
#define REG_SNAPSHOT_DATA    0x40   /* raw 5x5 luminance sum at snapshot_addr */
#define REG_SNAPSHOT_ADDR    0x50
#define REG_CNN_START        0x60
#define REG_CNN_RESULT       0x70

/* Reading past the 2304 image cells returns camera telemetry instead. */
#define SNAP_RED_COUNT       2304u
#define SNAP_COLOUR_HW       2305u
#define SNAP_VS_COUNT        2306u   /* MIPI frame counter -- the one that matters */
#define SNAP_PIXCLK          2307u
#define SNAP_RETRIES         2308u
#define SNAP_CFG_STEP        2309u
#define SNAP_STATUS          2310u

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
#define IMG_DIM              48
#define IMG_PIXELS           (IMG_DIM * IMG_DIM)   /* 2304 */
#define SUM_MAX              6375u   /* 25 px * 255: ITU-R 601 luminance of white */
#define Q_ONE                1024u   /* 1.0 in Q6.10 */

/* ---- software digital zoom ---------------------------------------------
 * Trims the square hardware crop to the 0.595 aspect the model was trained on
 * (corner_crops(frac_w=0.25, frac_h=0.30) of a 250x350 card). Its job is
 * ASPECT, not magnification -- aim so the card's index corner fills the green
 * box on HDMI, then use these only to fix the ratio.
 *
 * ZOOM_W is forgiving (24..32 all score 54/54 in simulation); ZOOM_X/ZOOM_Y
 * are where the precision is, and are rig-specific. */
#define ZOOM_X               9
#define ZOOM_Y               0
#define ZOOM_W               29      /* 29/48 = 0.60 */
#define ZOOM_H               48

/* Software colour decision. The fabric flag trips at ~1% red pixels, which a
 * red-and-gold face card can reach on its own; this threshold is applied to
 * the reported red_count instead. Recalibrate for your deck and lighting. */
#define RED_THRESH_SW        5000u

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
