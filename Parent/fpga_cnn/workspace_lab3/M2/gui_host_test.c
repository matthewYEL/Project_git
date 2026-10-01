/*
 * Off-board check of grid.c and hdmi_gui.c -- NOT part of the board build (not
 * in the Makefile's APP_SRC). The PIO writes land in an emulated text RAM with
 * the fabric's semantics (img_wr_addr[13] = level write enable), scripted
 * rounds are played through the GUI, and the screen is checked. Up to three
 * RAM snapshots go to files that text_layer.py renders the way the HDMI mux
 * would (a finished 3x3 scan, a 5x5 scan part way, a 4x4 game):
 *
 *   gcc -DGUI_HOST -Wall -Wextra -I. gui_host_test.c hdmi_gui.c grid.c -o gui_test
 *   ./gui_test scan3.hex scan5.hex game4.hex
 *   python ../../text_layer.py render scan3.hex scan3.png --font ../../font8x16.hex
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "card_pipeline.h"
#include "grid.h"
#include "hdmi_gui.h"

const char * const RANK_NAMES[ 13 ] = { "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A" };
const char * const SUIT_NAMES[ 4 ]  = { "Spades", "Clubs", "Hearts", "Diamonds" };

/* ---- the fabric: img_wr_data / img_wr_addr -> text RAM ------------------ */

#define CELLS 2400
static uint16_t ram[ CELLS ];
static uint16_t win_reg[ 8 ];           /* ghrd_top.v's window registers, 0xE00 + n */
static uint32_t pio_data, pio_addr;
static unsigned n_writes;

void alt_write_word( uint32_t addr, uint32_t value )
{
    if( addr == 0xFF200010u ) pio_data = value & 0xFFFFu;
    else if( addr == 0xFF200020u ) pio_addr = value & 0x3FFFu;
    else assert( !"the GUI wrote a register other than img_wr_data / img_wr_addr" );
    n_writes++;
    /* the RAM writes on every clk50 while the enable is up; 0xE00 up is the
     * window registers, and the text RAM only takes cells below 2400 */
    if( pio_addr & 0x2000u ) {
        unsigned a = pio_addr & 0xFFFu;

        if( ( a & 0xF00u ) == 0xE00u ) win_reg[ a & 7u ] = ( uint16_t ) pio_data;
        else {
            assert( a < CELLS );
            ram[ a ] = ( uint16_t ) pio_data;
        }
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
static unsigned fg_at( int row, int col ) { return ( ram[ row * 80 + col ] >> 8 ) & 0xFu; }
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

static void dump( const char * path )
{
    FILE * f;
    int i;

    if( !path ) return;
    f = fopen( path, "w" );
    assert( f );
    for( i = 0; i < CELLS; i++ ) fprintf( f, "%04x\n", ram[ i ] );
    fclose( f );
    printf( "text RAM -> %s\n", path );
}

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

enum { R3 = 1, R7 = 5, R10 = 8, RQ = 10, RK = 11, RA = 12 };
enum { SPADES, CLUBS, HEARTS, DIAMONDS };

/* a face-up 3x3 deal in cell order A1 B1 C1 / A2 B2 C2 / A3 B3 C3: pairs of 7s,
 * Qs and 3s, a third 3 left over, a king and a joker */
static const struct { unsigned rank, suit; int joker; } DEAL[ 9 ] = {
    { R7, SPADES, 0 }, { R7, HEARTS, 0 }, { RQ, CLUBS, 0 },
    { RQ, DIAMONDS, 0 }, { R3, SPADES, 0 }, { R3, HEARTS, 0 },
    { R3, CLUBS, 0 }, { RK, SPADES, 0 }, { 0, 0, 1 },
};

int main( int argc, char ** argv )
{
    struct vision_result v;
    struct grid t;
    int i;

    /* ---- grid.c on its own ----------------------------------------------- */

    grid_new( &t, 3, GRID_SCAN );
    assert( t.target == 0 && t.n_read == 0 && t.pairs == 0 );
    for( i = 0; i < 9; i++ ) {
        v = card( DEAL[ i ].rank, DEAL[ i ].suit, DEAL[ i ].joker, 1000 );
        assert( grid_read( &t, v.result, ( unsigned ) i ) == ( i == 1 || i == 3 || i == 5 ? GRID_PAIRED : GRID_PLACED ) );
    }
    assert( t.n_read == 9 && t.target == -1 && t.pairs == 3 );
    assert( t.cell[ 0 ].pair == 1 && t.cell[ 1 ].pair == 1 );          /* 7s: A1 + B1 */
    assert( t.cell[ 2 ].pair == 2 && t.cell[ 3 ].pair == 2 );          /* Qs: C1 + A2 */
    assert( t.cell[ 4 ].pair == 3 && t.cell[ 5 ].pair == 3 );          /* 3s: B2 + C2 */
    assert( t.cell[ 6 ].pair == 0 && t.cell[ 8 ].pair == 0 );          /* the third 3, the joker */
    v = card( RK, CLUBS, 0, 1 );
    assert( grid_read( &t, v.result, 9 ) == GRID_NO_CELL );             /* full */
    assert( grid_parse_cell( &t, "b3" ) == 7 && grid_parse_cell( &t, "C1" ) == 2 );
    assert( grid_parse_cell( &t, "d1" ) == -1 && grid_parse_cell( &t, "a4" ) == -1 && grid_parse_cell( &t, "a" ) == -1 );
    /* a correction: B2 read again as a king -- the 3s at C2 and A3 now pair,
     * and the new king pairs with B3's */
    assert( grid_target( &t, 4 ) );
    v = card( RK, HEARTS, 0, 1 );
    grid_read( &t, v.result, 10 );
    assert( t.pairs == 4 && t.n_read == 9 && t.target == -1 );
    assert( t.cell[ 5 ].pair == 3 && t.cell[ 6 ].pair == 3 && t.cell[ 4 ].pair == 4 && t.cell[ 7 ].pair == 4 );

    grid_new( &t, 4, GRID_GAME );
    assert( t.target == -1 && grid_left( &t ) == 16 && t.team == 0 && t.turn == 1 );
    v = card( R7, SPADES, 0, 1 );
    assert( grid_read( &t, v.result, 0 ) == GRID_NO_CELL );             /* no cell chosen yet */
    assert( grid_target( &t, 0 ) && grid_read( &t, v.result, 0 ) == GRID_OPENED );
    assert( !grid_target( &t, 0 ) );                                    /* it is already up */
    assert( grid_target( &t, 5 ) );
    v = card( R7, HEARTS, 0, 1 );
    assert( grid_read( &t, v.result, 1 ) == GRID_JUDGED && t.outcome == OUT_MATCH );
    assert( t.score[ 0 ] == 1 && t.team == 1 && t.turn == 2 && grid_left( &t ) == 14 );
    assert( !grid_target( &t, 0 ) );                                    /* matched: taken */
    assert( grid_target( &t, 1 ) );
    v = card( RQ, CLUBS, 0, 1 );
    grid_read( &t, v.result, 2 );
    assert( grid_target( &t, 2 ) );
    v = card( RK, CLUBS, 0, 1 );
    assert( grid_read( &t, v.result, 3 ) == GRID_JUDGED && t.outcome == OUT_NO_MATCH );
    assert( t.score[ 1 ] == 0 && t.team == 0 && t.turn == 3 );
    assert( t.cell[ 1 ].state == CELL_OPEN );                           /* up until the next turn */
    assert( grid_target( &t, 3 ) && t.cell[ 1 ].state == CELL_DOWN && t.cell[ 2 ].state == CELL_DOWN );
    printf( "grid.c: scan and game rules pass\n" );

    /* ---- the dashboard: a face-up 3x3 scan (round 1) --------------------- */

    gui_init();
    EXPECT( 0, "GAME OF POKER" );
    EXPECT( 0, " READY " );
    assert( bg_at( 0, 78 ) == GUI_LGREEN && bg_at( 0, 79 ) == GUI_BROWN );  /* pill ends in col 78 */
    EXPECT( 1, "LIVE CAMERA" );
    EXPECT( 1, "FACE-UP 3x3" );
    EXPECT( 18, "whole card in the green box" );
    EXPECT( 19, "NEXT: A1" );
    EXPECT( 21, "READ 0/9" );
    EXPECT( 29, "KEY1 undo/hold:new" );
    EXPECT( 29, "UART h" );
    EXPECT( 29, "SW0 turn" );
    assert( bg_at( 20, 0 ) == GUI_BROWN && bg_at( 20, 79 ) == GUI_BROWN );  /* the rail */
    assert( bg_at( 18, 60 ) == GUI_GREEN );                  /* bare felt */
    /* 3x3 cards are 9x7 at columns 47/58/69, rows 5/13/21; A1 is next (brass) */
    assert( ch_at( 5, 47 ) == 0xDA && fg_at( 5, 47 ) == GUI_YELLOW );
    assert( ch_at( 5, 58 ) == 0xDA && fg_at( 5, 58 ) == GUI_LGREEN );
    assert( ch_at( 4, 51 ) == 'A' && ch_at( 4, 73 ) == 'C' && ch_at( 8, 45 ) == '1' && ch_at( 24, 45 ) == '3' );
    PREVIEW_BLANK();

    for( i = 0; i < 9; i++ ) {
        v = card( DEAL[ i ].rank, DEAL[ i ].suit, DEAL[ i ].joker, 1000 + i );
        gui_job_done( ( unsigned ) i, &v, 1, "CPU1" );
        if( i == 1 ) EXPECT( 0, "PAIR: 7 + B1" );
    }
    EXPECT( 19, "SCAN DONE - 3 PAIRS FOUND" );
    EXPECT( 21, "READ 9/9    PAIRS 3" );
    EXPECT( 22, " 1 7  A1+B1" );
    EXPECT( 23, " 2 Q  C1+A2" );
    EXPECT( 24, " 3 3  B2+C2" );
    EXPECT( 28, "last: JOKER -> C3" );
    assert( bg_at( 6, 48 ) == GUI_YELLOW && ch_at( 6, 48 ) == '7' );  /* A1: paired, brass */
    assert( ch_at( 10, 48 ) == '#' && ch_at( 10, 49 ) == '1' && fg_at( 10, 49 ) == GUI_BLUE );  /* its pair tag */
    assert( bg_at( 22, 48 ) == GUI_WHITE && ch_at( 22, 48 ) == '3' ); /* A3: the odd 3, white */
    assert( bg_at( 22, 59 ) == GUI_WHITE && ch_at( 22, 59 ) == 'K' ); /* B3: unpaired king */
    assert( count_ch( 21, 69, 7, 9, 0x01 ) == 1 );                    /* C3: joker face */

    gui_undo();                                              /* takes the joker off C3 */
    EXPECT( 0, "UNDID C3" );
    EXPECT( 19, "NEXT: C3" );
    EXPECT( 21, "READ 8/9" );
    assert( ch_at( 21, 69 ) == 0xDA && fg_at( 21, 69 ) == GUI_YELLOW );  /* an empty slot again, next */
    v = card( 0, 0, 1, 700 );
    gui_job_done( 9, &v, 1, "CPU1" );
    EXPECT( 21, "READ 9/9" );
    v = card( RA, CLUBS, 0, 700 );
    gui_job_done( 10, &v, 1, "CPU1" );                       /* full: refused, nothing moves */
    EXPECT( 0, "GRID FULL" );
    EXPECT( 21, "READ 9/9    PAIRS 3" );

    assert( gui_command( "b2" ) );                           /* re-read a cell to correct it */
    EXPECT( 0, "NEXT: B2" );
    EXPECT( 19, "NEXT: B2" );
    v = card( RK, HEARTS, 0, 900 );
    gui_job_done( 11, &v, 1, "CPU1" );
    EXPECT( 21, "PAIRS 4" );
    EXPECT( 25, " 4 K  B2+B3" );

    v = card( RA, SPADES, 0, 800 );                          /* SW3 auto: shown, never placed */
    gui_job_done( 12, &v, 0, "CPU1" );
    EXPECT( 0, "READY (auto)" );
    EXPECT( 28, "(not placed)" );
    EXPECT( 21, "READ 9/9    PAIRS 4" );

    v = card( RK, SPADES, 0, 1100 );    v.result |= 1u << 10;               /* DDR3 error */
    gui_job_done( 13, &v, 1, "CPU1" );
    EXPECT( 0, "DDR3 PORT ERROR" );
    v = card( RK, SPADES, 0, 1100 );    v.maxv = 0;                         /* blank frame */
    gui_job_done( 14, &v, 1, "CPU1" );
    EXPECT( 0, "BLANK FRAME" );
    v.status = VIS_NO_TRIGGER;          gui_job_done( 15, &v, 1, "CPU1" );
    EXPECT( 0, "CAMERA STOPPED" );
    EXPECT( 21, "READ 9/9    PAIRS 4" );                     /* none of those placed */
    gui_status( "READING #12345 (CPU1)", GUI_YELLOW );       /* the longest app_rtos.c sends */
    EXPECT( 0, "READING #12345 (CPU1) " );
    EXPECT( 0, "GAME OF POKER" );
    gui_status( "READY", GUI_LGREEN );
    PREVIEW_BLANK();
    dump( argc > 1 ? argv[ 1 ] : NULL );

    /* ---- 5x5 (round 2): cards 5x4 at columns 48+6k, rows 4+5k ------------ */

    assert( gui_command( "5" ) );
    EXPECT( 0, "NEW 5x5 SCAN" );
    EXPECT( 1, "FACE-UP 5x5" );
    EXPECT( 21, "READ 0/25" );
    for( i = 0; i < 12; i++ ) {
        v = card( ( unsigned ) ( i % 7 ) + 2u, ( unsigned ) i % 4u, 0, 900 );
        gui_job_done( ( unsigned ) ( 20 + i ), &v, 1, "CPU1" );
    }
    EXPECT( 19, "NEXT: C3" );                                /* cell 12 */
    EXPECT( 21, "READ 12/25" );
    assert( ch_at( 14, 60 ) == 0xDA && fg_at( 14, 60 ) == GUI_YELLOW );   /* C3: next, brass slot */
    assert( ch_at( 3, 50 ) == 'A' && ch_at( 3, 74 ) == 'E' && ch_at( 5, 46 ) == '1' && ch_at( 25, 46 ) == '5' );
    PREVIEW_BLANK();
    dump( argc > 2 ? argv[ 2 ] : NULL );

    /* ---- a face-down 4x4 game (round 3): cards 7x5 at 47+8k, rows 5+6k --- */

    assert( gui_command( "g4" ) );
    EXPECT( 1, "FACE-DOWN GAME 4x4" );
    EXPECT( 19, "TEAM A: type the 1st card's cell" );
    assert( count_ch( 6, 48, 3, 5, 0xB1 ) == 15 );           /* A1 face down */
    v = card( R7, SPADES, 0, 1 );
    gui_job_done( 40, &v, 1, "CPU1" );
    EXPECT( 0, "TYPE THE CELL FIRST" );
    assert( gui_command( "a1" ) );
    EXPECT( 19, "TEAM A: turn up A1" );
    assert( bg_at( 5, 48 ) == GUI_YELLOW );                  /* the chosen back gets a brass edge */
    gui_job_done( 41, &v, 1, "CPU1" );
    EXPECT( 19, "TEAM A: type the 2nd card's cell" );
    assert( gui_command( "a1" ) );
    EXPECT( 0, "A1 IS NOT FACE DOWN" );
    assert( gui_command( "b2" ) );
    v = card( R7, HEARTS, 0, 1 );
    gui_job_done( 42, &v, 1, "CPU1" );
    EXPECT( 0, "MATCH!" );
    EXPECT( 21, "TURN 2    TEAM B TO PLAY" );
    EXPECT( 22, "SCORE   A 1 : B 0" );
    EXPECT( 23, "LEFT    14 of 16 cards" );
    EXPECT( 25, "MATCH  A1+B2  (team A)" );
    assert( gui_command( "c1" ) );
    v = card( RQ, CLUBS, 0, 1 );
    gui_job_done( 43, &v, 1, "CPU1" );
    assert( gui_command( "d1" ) );
    v = card( RK, DIAMONDS, 0, 1 );
    gui_job_done( 44, &v, 1, "CPU1" );
    EXPECT( 0, "NO MATCH" );
    EXPECT( 21, "TURN 3    TEAM A TO PLAY" );
    EXPECT( 25, "NO MATCH  C1+D1  (team B)" );
    assert( ch_at( 6, 64 ) == 'Q' );                         /* C1 still up for the players to see */
    PREVIEW_BLANK();
    dump( argc > 3 ? argv[ 3 ] : NULL );

    assert( gui_command( "a2" ) );                           /* the next turn: C1, D1 face down again */
    assert( count_ch( 6, 64, 3, 5, 0xB1 ) == 15 && count_ch( 6, 72, 3, 5, 0xB1 ) == 15 );
    gui_undo();                                              /* back to before D1 was read */
    EXPECT( 0, "UNDID D1" );
    EXPECT( 21, "TURN 2    TEAM B TO PLAY" );
    gui_clear();                                             /* KEY1 held: the same board again */
    EXPECT( 0, "NEW 4x4 GAME" );
    assert( count_ch( 28, 2, 1, 39, ' ' ) == 39 );         /* no stale "last:" line */
    EXPECT( 22, "SCORE   A 0 : B 0" );
    gui_undo();
    EXPECT( 0, "NOTHING TO UNDO" );
    assert( !gui_command( "zz" ) && !gui_command( "e5" ) && !gui_command( "9" ) );
    PREVIEW_BLANK();

    /* ---- whole-grid scan: the 28 Sep photo layout, then a 4th row added ---- */
    {
        static const struct { unsigned rank, suit; } LAY[ 12 ] = {
            { R7, HEARTS },   { RQ, CLUBS },    { 0, SPADES },       /* 7H QC 2S */
            { RA, DIAMONDS }, { R7, DIAMONDS }, { R3, CLUBS },       /* AD 7D 3C */
            { RK, HEARTS },   { 9, SPADES },    { RA, HEARTS },      /* KH JS AH */
            { R3, DIAMONDS }, { 7, SPADES },    { 0, HEARTS },       /* 3D 9S 2H: the added row */
        };
        struct scan_result s;
        struct win_regs    w = { -37, 5, 94, WIN_T | WIN_FY, { 10, 20, 300, 200 } };
        unsigned           k;

        win_write( &w );                                     /* the registers, not the text RAM */
        assert( win_reg[ 0 ] == ( ( unsigned ) -37 & 0x7FFu ) && win_reg[ 1 ] == 5 && win_reg[ 2 ] == 94 &&
                win_reg[ 3 ] == ( WIN_T | WIN_FY ) && win_reg[ 4 ] == 10 && win_reg[ 5 ] == 20 &&
                win_reg[ 6 ] == 300 && win_reg[ 7 ] == 200 );

        memset( &s, 0, sizeof s );
        s.n = 9;
        s.nrows = s.ncols = 3;
        for( k = 0; k < 12; k++ ) {
            s.card[ k ].row = ( uint8_t ) ( k / 3 + 1 );
            s.card[ k ].col = ( uint8_t ) ( k % 3 + 1 );
        }
        gui_scan_begin( &s );
        EXPECT( 0, "SCANNING 9 CARDS" );
        EXPECT( 1, "GRID SCAN 3 x 3" );
        for( k = 0; k < 9; k++ ) {
            v = card( LAY[ k ].rank, LAY[ k ].suit, 0, 900 );
            gui_scan_card( s.card[ k ].row, s.card[ k ].col, v.result, 100 + k );
        }
        gui_scan_done();
        EXPECT( 0, "SCAN DONE - 2 PAIRS" );
        EXPECT( 19, "GRID SCAN - 9 CARDS, 2 PAIRS" );
        EXPECT( 22, " 1 7  (1,1)+(2,2)" );
        EXPECT( 23, " 2 A  (2,1)+(3,3)" );
        EXPECT( 28, "last: A of Hearts -> (3,3)" );
        PREVIEW_BLANK();
        dump( argc > 4 ? argv[ 4 ] : NULL );

        s.n = 12;                                            /* the professor adds a row */
        s.nrows = 4;
        gui_scan_begin( &s );
        for( k = 0; k < 12; k++ ) {
            v = card( LAY[ k ].rank, LAY[ k ].suit, 0, 900 );
            gui_scan_card( s.card[ k ].row, s.card[ k ].col, v.result, 200 + k );
        }
        gui_scan_done();
        EXPECT( 0, "SCAN DONE - 4 PAIRS" );
        EXPECT( 1, "GRID SCAN 4 x 3" );
        EXPECT( 24, " 3 3  (2,3)+(4,1)" );
        EXPECT( 25, " 4 2  (1,3)+(4,3)" );
        gui_scan_card( 9, 9, v.result, 300 );                 /* off the board: ignored */
        gui_scan_card( 2, 2, 0, 301 );                       /* no result: status only */
        EXPECT( 0, "NO READ AT (2,2)" );
        PREVIEW_BLANK();
        dump( argc > 5 ? argv[ 5 ] : NULL );

        gui_undo();                                          /* KEY1 tap: the 3x3 result again */
        EXPECT( 0, "UNDID LAST SCAN" );
        EXPECT( 1, "GRID SCAN 3 x 3" );
        EXPECT( 21, "READ 9/9    PAIRS 2" );
        EXPECT( 22, " 1 7  (1,1)+(2,2)" );
        gui_undo();                                          /* one level only */
        EXPECT( 0, "NOTHING TO UNDO" );

        /* the 4x4 round: 16 cards, card k and k + 8 share a rank -> eight pairs */
        s.n = 16;
        s.nrows = s.ncols = 4;
        for( k = 0; k < 16; k++ ) {
            s.card[ k ].row = ( uint8_t ) ( k / 4 + 1 );
            s.card[ k ].col = ( uint8_t ) ( k % 4 + 1 );
        }
        gui_scan_begin( &s );
        EXPECT( 0, "SCANNING 16 CARDS" );
        EXPECT( 1, "GRID SCAN 4 x 4" );
        for( k = 0; k < 16; k++ ) {
            v = card( k % 8, k < 8 ? SPADES : HEARTS, 0, 900 );
            gui_scan_card( s.card[ k ].row, s.card[ k ].col, v.result, 400 + k );
        }
        gui_scan_done();
        EXPECT( 0, "SCAN DONE - 8 PAIRS" );
        EXPECT( 21, "READ 16/16    PAIRS 8" );
        EXPECT( 22, " 1 2  (1,1)+(3,1)" );
        EXPECT( 22, " 7 8  (2,3)+(4,3)" );                   /* the second column of pairs */
        EXPECT( 23, " 8 9  (2,4)+(4,4)" );
        PREVIEW_BLANK();
        dump( argc > 7 ? argv[ 7 ] : NULL );
        gui_clear();                                         /* KEY1 held: a clean board ... */
        EXPECT( 21, "READ 0/16" );
        gui_undo();                                          /* ... and a tap brings the 4x4 back */
        EXPECT( 0, "UNDID LAST SCAN" );
        EXPECT( 1, "GRID SCAN 4 x 4" );
        EXPECT( 21, "READ 16/16    PAIRS 8" );

        assert( gui_command( "3" ) );                        /* a guided board: letters again */
        EXPECT( 1, "FACE-UP 3x3" );

        gui_mode( 1, "cam CW" );                             /* AUTO: frame guides over the preview */
        EXPECT( 18, "AUTO: grid inside the lines (cam CW)" );
        assert( count_ch( 4, 5, 11, 1, 0xB3 ) == 11 && count_ch( 4, 36, 11, 1, 0xB3 ) == 11 );
        assert( count_ch( 2, 1, 15, 40, ' ' ) == 15u * 40u - 22u );
        gui_mode( 0, "cam CW" );                             /* guided again: the preview clear */
        EXPECT( 18, "fit the whole card in the green box" );
        PREVIEW_BLANK();

    }

    /* ---- SW2: the camera's face-down game, one player ----------------------- */
    {
        uint32_t h7 = card( R7, HEARTS, 0, 900 ).result, d7 = card( R7, DIAMONDS, 0, 900 ).result,
                 sk = card( RK, SPADES, 0, 900 ).result;
        static const struct { unsigned r1, c1, r2, c2, rank; } ALL[ 4 ] = {
            { 1, 1, 1, 2, RA }, { 1, 3, 2, 1, 0 }, { 2, 2, 2, 3, R3 }, { 3, 1, 3, 2, 2 },
        };
        unsigned k;

        gui_game_begin( 3 );                                 /* the camera saw 3x3, all face down */
        EXPECT( 0, "GAME 3x3 READY" );
        EXPECT( 1, "FACE-DOWN GAME 3 x 3" );
        EXPECT( 19, "TURN 1: turn 2 cards up, KEY0" );
        EXPECT( 21, "TURN 1    PAIRS 0" );
        EXPECT( 22, "LEFT    9 of 9 cards" );
        EXPECT( 29, "SW2 game" );
        assert( gui_game_n() == 3 && gui_game_matched() == 0 );

        assert( gui_game_turn( 1, 1, h7, 2, 2, sk, 500 ) == OUT_NO_MATCH );
        EXPECT( 0, "NO PAIR 7H(1,1) KS(2,2)" );
        EXPECT( 19, "TURN 2: turn 2 cards up, KEY0" );
        EXPECT( 24, "NO PAIR  (1,1)+(2,2) - turn them back" );
        assert( gui_game_matched() == 0 );

        assert( gui_game_turn( 1, 1, h7, 3, 2, d7, 501 ) == OUT_MATCH );   /* the 7 again, and its partner */
        EXPECT( 0, "PAIR! 7H(1,1) 7D(3,2)" );
        EXPECT( 21, "TURN 3    PAIRS 1" );
        EXPECT( 22, "LEFT    7 of 9 cards" );
        EXPECT( 24, "PAIR  (1,1)+(3,2)" );
        assert( gui_game_matched() == ( ( 1u << 0 ) | ( 1u << ( 2 * GRID_MAX + 1 ) ) ) );
        PREVIEW_BLANK();
        dump( argc > 6 ? argv[ 6 ] : NULL );

        assert( gui_game_turn( 1, 1, h7, 2, 2, sk, 502 ) == OUT_NONE );    /* a found pair's cell: refused */
        EXPECT( 0, "NOT FACE-DOWN CELLS" );
        EXPECT( 21, "TURN 3    PAIRS 1" );                                  /* nothing changed */
        gui_undo();                                                          /* KEY1 tap: back before the pair */
        EXPECT( 0, "UNDID LAST TURN" );
        EXPECT( 21, "TURN 2    PAIRS 0" );
        assert( gui_game_matched() == 0 );
        gui_undo();
        EXPECT( 0, "NOTHING TO UNDO" );

        gui_clear();                                                         /* KEY1 held: all face down again */
        EXPECT( 19, "TURN 1: turn 2 cards up, KEY0" );
        EXPECT( 22, "LEFT    9 of 9 cards" );
        assert( gui_game_n() == 3 );
        for( k = 0; k < 4; k++ ) {                                           /* a whole game: four pairs */
            uint32_t w = card( ALL[ k ].rank, SPADES, 0, 900 ).result, x = card( ALL[ k ].rank, CLUBS, 0, 900 ).result;
            assert( gui_game_turn( ALL[ k ].r1, ALL[ k ].c1, w, ALL[ k ].r2, ALL[ k ].c2, x, 510 + k ) == OUT_MATCH );
        }
        EXPECT( 19, "GAME OVER - 4 PAIRS IN 4 TURNS" );
        EXPECT( 22, "LEFT    1 of 9 cards" );

        assert( gui_command( "g3" ) );                       /* the guided two-team game is unchanged */
        EXPECT( 1, "FACE-DOWN GAME 3x3" );
        EXPECT( 21, "TURN 1    TEAM A TO PLAY" );
        assert( gui_game_n() == 0 );
        PREVIEW_BLANK();
    }

    printf( "screen at the end (%u PIO writes):\n", n_writes );
    for( i = 0; i < 30; i++ ) printf( "  %2d |%s|\n", i, row_text( i ) );
    printf( "gui_host_test: all checks passed\n" );
    return 0;
}
