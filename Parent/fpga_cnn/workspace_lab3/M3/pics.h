/*
 * Pictures kept in board memory: the SW1 frame dump without the UART.
 *
 * After a scan (or a guided read) with SW1 up, pic_store() copies the stored
 * 512x384 frame -- the exact words the card finder and the CNN read, {gray, 0,
 * red}, the same values dump_pgm() prints -- together with the board's own
 * reads into g_pics, a ring of PIC_SLOTS pictures in DDR. Arm DS saves the
 * whole ring to a file over the USB-Blaster cable -- Interrupt, then in the
 * Commands view (Arm DS command reference 101471, "dump"), then Continue:
 *
 *     dump binary value "C:/.../dumps/pics1.bin" &g_pics
 *     dump binary memory "C:/.../dumps/pics1.bin" START +SIZE   (the same, by number)
 *
 * START and SIZE from `arm-eabi-nm -S atlas_main.axf | grep g_pics`; they move
 * when the app is rebuilt, the symbol form does not. Add -r after the file
 * name to overwrite. (Arm DS's END is inclusive, hence +SIZE.) Then
 * lab 5/sim_card_cnn.py reads the file like a PuTTY log (--frames, --scan).
 * Added 30 Sep, when the UART cable died: a 1-2 s copy instead of a 70 s dump,
 * and no cable needed.
 *
 * This struct IS the file format: little-endian, every field sized so the
 * compiler adds no padding (static_assert in pics.c), and mirrored by
 * sim_card_cnn.py's reader. Change both together.
 */
#ifndef PICS_H
#define PICS_H

#include <stdint.h>

#include "app_config.h"
#include "card_pipeline.h"

#define PIC_MAGIC   "D8MPICS1"      /* 8 bytes, stored without a terminator */

struct pic_meta
{
    uint32_t seq;                   /* picture number from 1; 0 = empty slot */
    uint32_t n;                     /* cards read; 0 for a guided read */
    uint32_t words[ SCAN_MAX ];     /* each card's final result word (RES_*) */
    uint8_t  row[ SCAN_MAX ];       /* ... and its (row, col) */
    uint8_t  col[ SCAN_MAX ];
    uint8_t  pad[ 2 ];
};

struct pic_store
{
    char            magic[ 8 ];
    uint32_t        width, height;  /* FRAME_W, FRAME_H */
    uint32_t        slots;          /* PIC_SLOTS */
    uint32_t        count;          /* pictures taken since launch; slot = (seq - 1) % slots */
    uint32_t        reserved[ 2 ];
    struct pic_meta meta[ PIC_SLOTS ];
    uint16_t        frame[ PIC_SLOTS ][ FRAME_H * FRAME_W ];
};

extern struct pic_store g_pics;

/* Keep the stored frame and the reads (s and words may be NULL: a guided read)
 * in the next slot; past PIC_SLOTS the oldest is overwritten. Returns the
 * picture number. CPU0, and only while the core is idle -- during a job the
 * frame's read port is conv1's. */
unsigned pic_store( const struct scan_result * s, const uint32_t * words );

#endif /* PICS_H */
