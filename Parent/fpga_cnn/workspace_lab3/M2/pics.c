/* Pictures kept in board memory -- see pics.h. */
#include <string.h>

#include "pics.h"

/* the file format has no padding: Python reads it field by field */
typedef char pic_meta_packed[ sizeof( struct pic_meta ) == 8 + 4 * SCAN_MAX + 2 * SCAN_MAX + 2 ? 1 : -1 ];
typedef char pic_store_packed[ sizeof( struct pic_store ) ==
                               32 + PIC_SLOTS * ( sizeof( struct pic_meta ) + 2u * FRAME_H * FRAME_W ) ? 1 : -1 ];

/* ~9.4 MB of .bss in DDR, well below the AMP block (0x03000000) and fc_shared's
 * weights (0x10000000). The caches are off, so Arm DS reads it as written. */
struct pic_store g_pics;

unsigned pic_store( const struct scan_result * s, const uint32_t * words )
{
    unsigned          seq  = g_pics.count + 1u, i, y, x;
    struct pic_meta * m    = &g_pics.meta[ ( seq - 1u ) % PIC_SLOTS ];
    uint16_t *        f    = g_pics.frame[ ( seq - 1u ) % PIC_SLOTS ];

    memcpy( g_pics.magic, PIC_MAGIC, sizeof g_pics.magic );
    g_pics.width  = FRAME_W;
    g_pics.height = FRAME_H;
    g_pics.slots  = PIC_SLOTS;

    m->seq = 0;                             /* an empty slot until the copy is whole */
    for( y = 0; y < FRAME_H; y++ )
        for( x = 0; x < FRAME_W; x++ )
            *f++ = ( uint16_t ) ( snap_read( FRAME_ADDR( y, x ) ) & 0x3FFu );

    memset( m, 0, sizeof *m );
    if( s && words ) {
        m->n = s->n < SCAN_MAX ? s->n : SCAN_MAX;
        for( i = 0; i < m->n; i++ ) {
            m->words[ i ] = words[ i ];
            m->row[ i ]   = s->card[ i ].row;
            m->col[ i ]   = s->card[ i ].col;
        }
    }
    m->seq       = seq;
    g_pics.count = seq;
    return seq;
}
