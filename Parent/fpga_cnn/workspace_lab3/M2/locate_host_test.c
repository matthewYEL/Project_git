/*
 * Host test for locate.c: reads one frame file, runs locate_cards() and
 * window_for_card(), prints one line per card. lab 5/locate_check.py builds
 * the frames from the 28 Sep photos and diffs these lines against locate.py.
 *
 *   gcc -O2 -I. -o locate_host_test locate_host_test.c locate.c
 *   locate_host_test frame.bin ORIENT
 *
 * frame.bin: uint16 width, uint16 height, then width*height uint16 words as the
 * board's snapshot_data returns them ({gray, 0, red} = gray << 2 | red).
 */
#include <stdio.h>
#include <stdlib.h>

#include "locate.h"

static uint16_t s_frame[ FRAME_W * FRAME_H ];
static int      s_w, s_h;

/* the board's snap_read, over the loaded frame; FRAME_ADDR is row*FRAME_W+col */
unsigned snap_read( unsigned addr )
{
    unsigned row = addr / FRAME_W, col = addr % FRAME_W;

    if( row >= ( unsigned ) s_h || col >= ( unsigned ) s_w ) return 0;
    return s_frame[ row * FRAME_W + col ];
}

int main( int argc, char ** argv )
{
    static struct scan_result r;
    uint16_t hdr[ 2 ];
    unsigned orient, i;
    int      y;
    FILE *   f;

    if( argc != 3 || !( f = fopen( argv[ 1 ], "rb" ) ) ) {
        fprintf( stderr, "usage: locate_host_test frame.bin ORIENT\n" );
        return 2;
    }
    orient = ( unsigned ) atoi( argv[ 2 ] );
    if( fread( hdr, 2, 2, f ) != 2 || hdr[ 0 ] > FRAME_W || hdr[ 1 ] > FRAME_H ) return 2;
    s_w = hdr[ 0 ];
    s_h = hdr[ 1 ];
    for( y = 0; y < s_h; y++ )
        if( fread( &s_frame[ y * FRAME_W ], 2, ( size_t ) s_w, f ) != ( size_t ) s_w ) return 2;
    fclose( f );

    locate_cards( &r, orient, s_w, s_h );
    printf( "thr %u n %u rows %u cols %u\n", r.threshold, r.n, r.nrows, r.ncols );
    for( i = 0; i < r.n; i++ ) {
        const struct scan_card * c = &r.card[ i ];
        struct win_regs w;

        window_for_card( c, orient, s_w, s_h, &w );
        printf( "(%u,%u) %u %u %u %u red %u up %u win %d %d %u %u clip %u %u %u %u\n",
                c->row, c->col, c->x0, c->y0, c->x1, c->y1, c->red, c->up,
                w.x0, w.y0, w.step, w.flags, w.clip[ 0 ], w.clip[ 1 ], w.clip[ 2 ], w.clip[ 3 ] );
    }
    return 0;
}
