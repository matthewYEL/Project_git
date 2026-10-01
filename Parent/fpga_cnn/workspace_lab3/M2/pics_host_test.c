/*
 * Host test for pics.c: stores 26 pictures through a fake snap_read (so the
 * 24-slot ring wraps), then writes g_pics to a file -- the bytes Arm DS saves
 * from the board. lab 5/sim_card_cnn.py's _pic_file() must read back every
 * frame and read exactly (see the check in the 30 Sep notes).
 *
 *   gcc -O2 -Wall -Wextra -I. -o pics_host_test pics_host_test.c pics.c
 *   pics_host_test out.bin
 *
 * Picture k: frame word at addr = (addr * 2654435761 + k * 40503) >> 22 (mod
 * 2^32, 10 bits); k % 10 cards at (i / 3 + 1, i % 3 + 1) with result word
 * k << 16 | done | rank i; every 7th picture a guided read (no cards).
 */
#include <stdio.h>

#include "pics.h"

static unsigned s_pic;          /* the picture the fake frame belongs to */

unsigned snap_read( unsigned addr )
{
    return ( addr * 2654435761u + s_pic * 40503u ) >> 22;
}

int main( int argc, char ** argv )
{
    static struct scan_result s;
    uint32_t words[ SCAN_MAX ];
    unsigned i, k;
    FILE *   f;

    if( argc != 2 || !( f = fopen( argv[ 1 ], "wb" ) ) ) {
        fprintf( stderr, "usage: pics_host_test out.bin\n" );
        return 2;
    }
    for( k = 1; k <= 26; k++ ) {
        s_pic = k;
        s.n   = k % 10;
        for( i = 0; i < s.n; i++ ) {
            s.card[ i ].row = ( uint8_t ) ( i / 3 + 1 );
            s.card[ i ].col = ( uint8_t ) ( i % 3 + 1 );
            words[ i ]      = ( k << 16 ) | ( 1u << 7 ) | i;
        }
        if( pic_store( k % 7 ? &s : NULL, k % 7 ? words : NULL ) != k ) return 3;
    }
    printf( "sizeof g_pics %u, count %u\n", ( unsigned ) sizeof g_pics, ( unsigned ) g_pics.count );
    if( fwrite( &g_pics, sizeof g_pics, 1, f ) != 1 ) return 4;
    fclose( f );
    return 0;
}
