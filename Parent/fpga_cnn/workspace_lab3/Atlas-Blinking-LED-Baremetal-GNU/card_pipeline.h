/*
 * The card recognition pipeline, as stages other translation units can call.
 *
 * The stages themselves live in atlas_main.c -- this only exposes them, so the
 * single-core loop (atlas_main.c), the FreeRTOS tasks on CPU0 (app_rtos.c) and
 * the vision worker on CPU1 (amp.c) all run the SAME pipeline code.
 *
 * Which functions may run where: vision_run() and everything it calls touch
 * only the FPGA PIOs and memory -- no printf -- so CPU1 can run them. The
 * print_* / dump_pgm functions use semihosted printf and are CPU0-only.
 */
#ifndef CARD_PIPELINE_H
#define CARD_PIPELINE_H

#include <stdint.h>

/* ---- image geometry and fixed point ------------------------------------ */
/* 384x384, sampled 1:1 from the camera's 640x480 frame by capture_384.v. Must
 * match ghrd_top.v's IMG_PIXELS: it is where the diagnostics below start. */
#define IMG_DIM      384
#define IMG_PIXELS   ( IMG_DIM * IMG_DIM )    /* 147456 */
/* snapshot_data returns each pixel as the Q6.10 value conv1 actually read,
 * which the fabric makes as gray*4 (full scale 1020, not 1024). So the 8-bit
 * gray is exact to recover, whatever IMG_DW the bitstream was built with. */
#define PIX_GRAY( q )       ( ( unsigned ) ( q ) >> 2 )

/* ---- cnn_result_pio bit layout ----------------------------------------- */
#define RES_RANK( r )       ( ( r ) & 0xFu )
#define RES_SUIT( r )       ( ( ( r ) >> 4 ) & 0x3u )
#define RES_JOKER( r )      ( ( ( r ) >> 6 ) & 0x1u )
#define RES_DONE( r )       ( ( ( r ) >> 7 ) & 0x1u )
#define RES_COLOUR( r )     ( ( ( r ) >> 8 ) & 0x1u )
#define RES_SNAP_DONE( r )  ( ( ( r ) >> 9 ) & 0x1u )
/* The fabric's DDR3 weight fetch timed out this inference: the result is not a
 * real classification (see ghrd_top.v cnn_result, fc_layer.v ddr_err). */
#define RES_DDR_ERR( r )    ( ( ( r ) >> 10 ) & 0x1u )

/* ---- fc_shared's weights in HPS DDR3 ------------------------------------- */
/* Byte address; one Q6.10 weight per 32-bit word, sign-extended. The fabric
 * reads them through the f2h_sdram0 port, which is WORD-addressed, so
 * card_cnn_core.v's FCS_W_WORD_BASE must be this >> 2 (0x04000000). 256 MB up
 * is clear of the app (linked at 0x00100040) and everything else in DDR3. */
#define FCS_W_DDR_BASE      0x10000000u

/* sdr.ctrlgrp.fpgaportrst: per-port reset release for the FPGA-to-SDRAM ports.
 * The preloader sets it from its own compiled-in port configuration; writing
 * all ones here releases the port even if that configuration had none. */
#define SDR_FPGAPORTRST     0xFFC25080u
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

/* Read one pixel of the captured image (Q6.10), or one of the diagnostic
 * registers at IMG_PIXELS + n. Image reads are only valid while no inference
 * is running -- the fabric hands the image RAM's read port to conv1 then. */
unsigned snap_read( unsigned addr );

/* Trigger a capture and wait for snapshot_done. 1 once the fabric has written
 * the whole 384x384 image into the CNN's image RAM. 0 on any failure: a
 * timeout, or the trigger never accepted (camera pixel clock stopped), in which
 * case the image RAM still holds the previous capture and must not be
 * classified. */
int capture_once( void );

/* Start the CNN on the captured image and poll for done. There is no upload:
 * capture_384.v already wrote the pixels. red_count feeds the software colour
 * override. Returns the raw result word, or 0 on timeout -- bit 7 is always
 * set in a real result. */
unsigned infer( unsigned red_count );

/* "3 of Hearts" / "JOKER", printed with a fixed field width. */
void print_result_name( unsigned r );

/* Report the MIPI link state, loudly if no frames are arriving. */
void check_camera_alive( void );

/* One line: mipi_cfg / cam_cfg / vs / pixclk / retries. CPU0 only. */
void print_camera_diag( void );

/* ---- one vision job ------------------------------------------------------
 * capture -> red count -> 48x48 sample (min/max + ASCII preview) -> CNN. */

/* dipsw_pio bits, read at the press */
#define SW_PREVIEW      0x1u    /* SW0: print the 48x48 ASCII preview        */
#define SW_PGM          0x2u    /* SW1: dump the full capture as a PGM       */
#define SW_STATS        0x4u    /* SW2: print scheduling stats (RTOS build)  */
#define SW_AUTO         0x8u    /* SW3: capture automatically, no KEY needed
                                 * (RTOS build; every AUTO_PERIOD_MS)       */

#define PREVIEW_DIM     48      /* IMG_DIM / 8: every 8th pixel each way */

enum vision_status
{
    VIS_OK = 0,             /* result holds a real result word (may carry RES_DDR_ERR) */
    VIS_NO_TRIGGER,         /* trigger never accepted: camera pixel clock stopped      */
    VIS_CAP_TIMEOUT,        /* trigger accepted but no frame completed                 */
    VIS_INFER_TIMEOUT       /* the CNN never raised done                               */
};

struct vision_result
{
    uint32_t status;        /* enum vision_status */
    uint32_t result;        /* raw cnn_result word, VIS_OK only */
    uint32_t red_count;
    uint32_t minv, maxv;    /* over the 48x48 sample grid; equal = flat/blank frame */
    uint32_t cap_us, inf_us;
    char     art[ PREVIEW_DIM * PREVIEW_DIM ];   /* valid when the capture succeeded */
};

/* Run one job. FPGA PIOs and memory only -- safe on CPU1. */
void vision_run( struct vision_result * v );

/* CPU0 only. print_vision_result prints one "[shot] ..." line (plus camera
 * diagnostics on a camera failure); `where` is appended as "  [where]" when
 * non-NULL. */
void print_preview( const char * art );
void print_vision_result( unsigned shot, const struct vision_result * v, const char * where );
void dump_pgm( void );

/* KEY0/KEY1 presses latched by button_pio's edge-capture register since the
 * last call (bit per key; the call clears them), and the DIP switches. */
unsigned key_presses( void );
unsigned read_switches( void );

#endif /* CARD_PIPELINE_H */
