/*
 * The card recognition pipeline, as stages other translation units can call.
 *
 * The stages themselves still live in atlas_main.c -- this only exposes them,
 * so that the FreeRTOS tasks (app_rtos.c) and the CPU1 worker (amp.c) reuse
 * exactly the code the single-threaded path was tuned and verified with,
 * rather than acquiring a second copy that can drift.
 *
 * Geometry and the result bit layout live here for the same reason: amp.h
 * needs IMG_PIXELS to size the shared frame slots.
 */
#ifndef CARD_PIPELINE_H
#define CARD_PIPELINE_H

#include <stdint.h>

/* ---- image geometry and fixed point ------------------------------------ */
#define IMG_DIM      96
#define IMG_PIXELS   ( IMG_DIM * IMG_DIM )    /* 9216 */
#define SUM_MAX      510u    /* 2 px * 255 (ITU-R 601 luminance of white) */
#define Q_ONE        1024u   /* 1.0 in Q6.10 */

/* ---- cnn_result_pio bit layout ----------------------------------------- */
#define RES_RANK( r )       ( ( r ) & 0xFu )
#define RES_SUIT( r )       ( ( ( r ) >> 4 ) & 0x3u )
#define RES_JOKER( r )      ( ( ( r ) >> 6 ) & 0x1u )
#define RES_DONE( r )       ( ( ( r ) >> 7 ) & 0x1u )
#define RES_COLOUR( r )     ( ( ( r ) >> 8 ) & 0x1u )
#define RES_SNAP_DONE( r )  ( ( ( r ) >> 9 ) & 0x1u )
#define RES_SCORE( r )      ( ( int16_t ) ( ( r ) >> 16 ) )

/* Camera-link diagnostics, added to camera_capture.v alongside the snapshot RAM.
 * vs_count is the one that matters: if it does not advance, no MIPI frames are
 * arriving and every prediction is the network's response to a blank image. */
#define SNAP_RED_COUNT    ( ( unsigned ) IMG_PIXELS + 0u )
#define SNAP_COLOUR_HW    ( ( unsigned ) IMG_PIXELS + 1u )
#define SNAP_VS_COUNT     ( ( unsigned ) IMG_PIXELS + 2u )   /* MIPI frame counter */
#define SNAP_PIXCLK       ( ( unsigned ) IMG_PIXELS + 3u )   /* MIPI pixel-clock activity counter */
#define SNAP_RETRIES      ( ( unsigned ) IMG_PIXELS + 4u )   /* I2C config re-runs forced by the watchdog */
#define SNAP_CFG_STEP     ( ( unsigned ) IMG_PIXELS + 5u )   /* where the config sequence got to */
#define SNAP_STATUS       ( ( unsigned ) IMG_PIXELS + 6u )
#define ST_MIPI_REL(s)    ((s) & 1u)
#define ST_CAM_REL(s)     (((s) >> 1) & 1u)
#define ST_AUDPLL_OK(s)   (((s) >> 2) & 1u)   /* AUDIO pll (feeds HDMI), NOT the MIPI clock */
#define ST_HDMI_RDY(s)    (((s) >> 3) & 1u)
/* wde_ticks, added after this build -- the earlier sticky saw_wde bit was
 * folded to a constant by synthesis and removed, so it is not read here. */
#define SNAP_WDE_TICKS    ( ( unsigned ) IMG_PIXELS + 7u )

/* ---- pipeline stages, defined in atlas_main.c -------------------------- */

/* Read one cell of the snapshot RAM, or one of the diagnostic registers past
 * the image cells. */
unsigned snap_read( unsigned addr );

/* Trigger a capture and wait for snapshot_done. 0 on timeout. */
int capture_once( void );

/* Read all IMG_PIXELS raw luminance sums, and their min/max. */
void read_snapshot( uint16_t * raw, unsigned * minv, unsigned * maxv );

/* Crop to the zoom window and rescale to Q6.10. Pure function on arrays: no
 * FPGA access, which is what makes it safe to run on CPU1. */
void preprocess( const uint16_t * raw, uint16_t * q );

/* Upload the IMG_PIXELS pixels, start the CNN, poll for done. Returns the raw result
 * word, or 0 on timeout -- bit 7 is always set in a real result. */
unsigned upload_and_infer( const uint16_t * q, unsigned red_count );

/* "3 of Hearts" / "JOKER", printed with a fixed field width. */
void print_result_name( unsigned r );

/* Report the MIPI link state, loudly if no frames are arriving. */
void check_camera_alive( void );

#endif /* CARD_PIPELINE_H */
