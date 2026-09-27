/*
 * Off-board check of hdmi_gui.c -- NOT part of the board build (not in the
 * Makefile's APP_SRC). The PIO writes land in an emulated text RAM with the
 * fabric's semantics (img_wr_addr[13] = level write enable), a scripted run of
 * presses is played through the GUI, and the screen is checked. The final RAM
 * goes to a file that text_layer.py renders the way the HDMI mux would:
 *
 *   gcc -DGUI_HOST -Wall -Wextra -I. gui_host_test.c hdmi_gui.c -o gui_test
 *   ./gui_test ram.hex
 *   python ../../text_layer.py render ram.hex gui.png --font ../../font8x16.hex
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "card_pipeline.h"
#include "hdmi_gui.h"

const char * const RANK_NAMES[ 13 ] = { "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A" };
const char * const SUIT_NAMES[ 4 ]  = { "Spades", "Clubs", "Hearts", "Diamonds" };

/* ---- the fabric: img_wr_data / img_wr_addr -> text RAM ------------------ */

#define CELLS 2400
static uint16_t ram[ CELLS ];
static uint32_t pio_data, pio_addr;
static unsigned n_writes;

void alt_write_word( uint32_t addr, uint32_t value )
{
    if( addr == 0xFF200010u ) pio_data = value & 0xFFFFu;
    else if( addr == 0xFF200020u ) pio_addr = value & 0x3FFFu;
    else assert( !"the GUI wrote a register other than img_wr_data / img_wr_addr" );
    n_writes++;
    /* the RAM writes on every clk50 while the enable is up */
    if( pio_addr & 0x2000u ) {
        assert( ( pio_addr & 0xFFFu ) < CELLS );
        ram[ pio_addr & 0xFFFu ] = ( uint16_t ) pio_data;
    }
}

/* ---- screen probes ------------------------------------------------------- */

static const char * row_text( int row )
{
    static char s[ 81 ];
    int c;

    for( c = 0; c < 80; c++ ) {
        unsigned ch = ram[ row * 80 + c ] & 0xFFu;
        s[ c ] = ( ch >= 0x20 && ch < 0x7F ) ? ( char ) ch : '.';
    }
    s[ 80 ] = '\0';
    return s;
}

static int on_row( int row, const char * want )
{
    return strstr( row_text( row ), want ) != NULL;
}

#define EXPECT( row, want ) \
    do { if( !on_row( row, want ) ) { printf( "row %2d: \"%s\"\n  missing \"%s\"\n", row, row_text( row ), want ); assert( 0 ); } } while( 0 )

static unsigned bg_at( int row, int col ) { return ram[ row * 80 + col ] >> 12; }
static unsigned ch_at( int row, int col ) { return ram[ row * 80 + col ] & 0xFFu; }

static unsigned count_ch( int row, int col, int h, int w, unsigned ch )
{
    unsigned n = 0;
    int r, c;

    for( r = row; r < row + h; r++ )
        for( c = col; c < col + w; c++ ) n += ch_at( r, c ) == ch;
    return n;
}

/* The fabric draws the camera in cells 1..40 x 2..16 wherever a glyph pixel is
 * background, so any non-blank cell there would print over the preview. */
#define PREVIEW_BLANK() assert( count_ch( 2, 1, 15, 40, ' ' ) == 15u * 40u )

/* the LAST CARD face, 13x9 at row 2, column 44 */
#define BIG_PIPS( ch ) count_ch( 2, 44, 9, 13, ( ch ) )

/* ---- a job result, as vision_run would leave it ------------------------- */

static struct vision_result card( unsigned rank, unsigned suit, int joker, int score )
{
    struct vision_result v;

    memset( &v, 0, sizeof v );
    v.status    = VIS_OK;
    v.result    = ( rank & 0xFu ) | ( ( suit & 3u ) << 4 ) | ( ( unsigned ) !!joker << 6 ) | ( 1u << 7 ) |
                  ( ( suit >= 2 ) ? 1u << 8 : 0u ) | ( 1u << 9 ) | ( ( uint32_t ) ( uint16_t ) score << 16 );
    v.red_count = ( suit >= 2 ) ? 20000u : 900u;
    v.minv      = 12;
    v.maxv      = 240;
    v.inf_us    = 1712000u;
    return v;
}

enum { R7 = 5, R10 = 8, RQ = 10, RK = 11, RA = 12 };
enum { SPADES, CLUBS, HEARTS, DIAMONDS };

int main( int argc, char ** argv )
{
    struct vision_result v;
    FILE * f;
    int i;

    gui_init();
    EXPECT( 0, "CARD TABLE" );
    EXPECT( 0, " READY " );
    assert( bg_at( 0, 78 ) == GUI_LGREEN && bg_at( 0, 79 ) == GUI_BROWN );  /* pill ends in col 78 */
    EXPECT( 1, "LIVE CAMERA" );
    EXPECT( 1, "LAST CARD" );
    EXPECT( 3, "press KEY0" );
    EXPECT( 12, "TABLE" );
    EXPECT( 13, " 0/52" );
    EXPECT( 19, "THE BOARD" );
    EXPECT( 18, "inside the green box" );
    EXPECT( 29, "KEY1 undo" );
    assert( bg_at( 20, 0 ) == GUI_BROWN && bg_at( 20, 79 ) == GUI_BROWN );  /* the rail */
    assert( bg_at( 18, 60 ) == GUI_GREEN );                  /* bare felt */
    assert( ch_at( 21, 19 ) == 0xB1u );                      /* nothing dealt: card backs */
    PREVIEW_BLANK();

    v = card( RQ, HEARTS, 0, 1234 );    gui_job_done( 0, &v, 1, "CPU1" );
    v = card( R7, SPADES, 0, 987 );     gui_job_done( 1, &v, 1, "CPU1" );
    v = card( RA, DIAMONDS, 0, 1500 );  gui_job_done( 2, &v, 1, "CPU1" );
    EXPECT( 13, " 3/52" );
    EXPECT( 3, "A of Diamonds" );
    assert( BIG_PIPS( 0x04 ) == 1 + 2 );                     /* an ace: one pip + two indices */

    gui_undo();                                              /* drops #2, A of Diamonds */
    EXPECT( 0, "UNDID #2" );
    EXPECT( 13, " 2/52" );
    EXPECT( 3, "7 of Spades" );                              /* LAST CARD falls back */
    assert( BIG_PIPS( 0x06 ) == 7 + 2 );

    v = card( R10, CLUBS, 0, 800 );     gui_job_done( 3, &v, 1, "CPU1" );
    assert( BIG_PIPS( 0x05 ) == 10 + 2 );
    v = card( R7, SPADES, 0, 990 );     gui_job_done( 4, &v, 1, "CPU1" );   /* a repeat */
    v = card( 0, 0, 1, 700 );           gui_job_done( 5, &v, 1, "CPU0" );   /* joker */
    EXPECT( 13, " 3/52" );
    EXPECT( 15, "REPEATS  1" );
    EXPECT( 16, "JOKERS   1" );
    EXPECT( 17, "READS    5" );
    /* deck chips: row 25 + suit, column 12 + 4 + 4 * rank */
    assert( bg_at( 25, 16 + 4 * R7 ) == GUI_YELLOW );        /* 7 of Spades seen twice */
    assert( bg_at( 27, 16 + 4 * RQ ) == GUI_WHITE );         /* Q of Hearts once */
    assert( bg_at( 26, 16 + 4 * R10 ) == GUI_WHITE );        /* 10 of Clubs once */
    assert( bg_at( 28, 16 + 4 * RA ) == GUI_GREEN );         /* A of Diamonds undone: unseen */
    /* the board, oldest on the left: QH 7S 10C 7S JOKER, cards 9 apart from column 18 */
    assert( ch_at( 21, 19 ) == 'Q' );
    EXPECT( 22, "JOKER" );
    assert( ch_at( 19, 18 + 4 * 9 + 3 ) == 0x1Fu );          /* the newest marker, over the joker */

    v = card( RK, DIAMONDS, 0, 1100 );  gui_job_done( 6, &v, 0, "CPU1" );   /* SW3 auto: shown only */
    EXPECT( 17, "READS    5" );
    EXPECT( 3, "K of Diamonds" );
    EXPECT( 8, "auto: not tracked" );
    EXPECT( 0, "READY (auto)" );
    assert( ch_at( 6, 44 + 6 ) == 'K' );                     /* a court card: framed letter */

    v = card( RK, SPADES, 0, 1100 );    v.result |= 1u << 10;               /* DDR3 error */
    gui_job_done( 7, &v, 1, "CPU1" );
    EXPECT( 0, "DDR3 PORT ERROR" );
    v = card( RK, SPADES, 0, 1100 );    v.maxv = 0;                         /* blank frame */
    gui_job_done( 8, &v, 1, "CPU1" );
    EXPECT( 0, "BLANK FRAME" );
    v.status = VIS_NO_TRIGGER;          gui_job_done( 9, &v, 1, "CPU1" );
    EXPECT( 0, "CAMERA STOPPED" );
    EXPECT( 17, "READS    5" );                              /* none of those tracked */
    gui_status( "READING #12345 (CPU1)", GUI_YELLOW );       /* the longest app_rtos.c sends */
    EXPECT( 0, "READING #12345 (CPU1) " );
    EXPECT( 0, "CARD TABLE" );

    v = card( RQ, CLUBS, 0, 1320 );     gui_job_done( 10, &v, 1, "CPU1" );
    assert( ch_at( 21, 19 ) == '7' && ch_at( 21, 18 + 4 * 9 + 1 ) == 'Q' );  /* the board slid left */
    PREVIEW_BLANK();
    printf( "screen after the scripted run (%u PIO writes so far):\n", n_writes );
    for( i = 0; i < 30; i++ ) printf( "  %2d |%s|\n", i, row_text( i ) );

    if( argc > 1 ) {
        f = fopen( argv[ 1 ], "w" );
        assert( f );
        for( i = 0; i < CELLS; i++ ) fprintf( f, "%04x\n", ram[ i ] );
        fclose( f );
        printf( "text RAM -> %s\n", argv[ 1 ] );
    }

    gui_clear();
    EXPECT( 0, "TRACKER CLEARED" );
    EXPECT( 13, " 0/52" );
    EXPECT( 17, "READS    0" );
    EXPECT( 3, "press KEY0" );
    assert( ch_at( 21, 19 ) == 0xB1u );
    gui_undo();
    EXPECT( 0, "NOTHING TO UNDO" );

    printf( "gui_host_test: all checks passed\n" );
    return 0;
}
