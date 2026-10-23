/*
 * Off-board check of sortgame.c (+ the M3 HDMI screen) -- NOT part of the
 * board build (not in the Makefile's APP_SRC). A fake camera stands in for
 * the board: cards are rectangles with a pattern of their own drawn into a
 * 512x384 grey frame, overlapping ones fuse into one box as locate's do, and
 * a "CNN read" returns the card on top. KEY0's read of a held card is
 * sort_held() with what app_rtos.c would have read. Whole games are scripted
 * through it -- slides, mistakes, a misread held card, a card held up over
 * the board, cards hidden by a hand, jokers in a gap, exactly over a card,
 * half over one, a board off to one side of the camera -- and the rules'
 * answers are checked. Screens go to files text_layer.py renders:
 *
 *   gcc -DGUI_HOST -Wall -Wextra -I. sort_host_test.c sortgame.c hdmi_gui.c grid.c -o sort_test
 *   ./sort_test placed.hex joker.hex place.hex air.hex memory.hex shifted.hex ignored.hex
 *   python ../../text_layer.py render place.hex place.png --font ../../font8x16.hex
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "card_pipeline.h"
#include "hdmi_gui.h"
#include "sortgame.h"

const char * const RANK_NAMES[ 13 ] = { "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A" };
const char * const SUIT_NAMES[ 4 ]  = { "Spades", "Clubs", "Hearts", "Diamonds" };

/* ---- the fabric: img_wr_data / img_wr_addr -> text RAM (gui_host_test.c) - */

#define CELLS 2400
static uint16_t ram[ CELLS ];
static uint32_t pio_data, pio_addr;

void alt_write_word( uint32_t addr, uint32_t value )
{
    if( addr == 0xFF200010u ) pio_data = value & 0xFFFFu;
    else if( addr == 0xFF200020u ) pio_addr = value & 0x3FFFu;
    else assert( !"the GUI wrote a register other than img_wr_data / img_wr_addr" );
    if( pio_addr & 0x2000u ) {
        unsigned a = pio_addr & 0xFFFu;

        if( ( a & 0xF00u ) != 0xE00u ) {
            assert( a < CELLS );
            ram[ a ] = ( uint16_t ) pio_data;
        }
    }
}

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

/* ---- the fake camera ------------------------------------------------------- */

enum { S_SPADES, S_CLUBS, S_HEARTS, S_DIAMONDS };
enum { R2, R3, R4, R5, R6, R7, R8, R9, R10, RJ, RQ, RK, RA };

static uint32_t W( unsigned rank, unsigned suit )         /* a cnn_result word, as the core returns it */
{
    return ( rank & 0xFu ) | ( ( suit & 3u ) << 4 ) | ( 1u << 7 ) | ( ( suit >= 2 ) ? 1u << 8 : 0u ) | ( 2000u << 16 );
}
static const uint32_t JOKER = ( 1u << 6 ) | ( 1u << 7 ) | ( 1500u << 16 );

struct obj { uint32_t word; int x, y, w, h; };              /* centre and size, frame px; drawn in array order */
#define CW  60                                              /* a card lying on the board */
#define CH  84
#define HW  170                                             /* ... held up over it */
#define HH  238
static struct obj scene[ 32 ];
static unsigned   n_obj;
static uint8_t    frame_px[ FRAME_H ][ FRAME_W ];

static int slot_x( int col ) { return SORT_X0 + col * SORT_BAND_W + SORT_BAND_W / 2; }
static int slot_y( int row ) { return row * SORT_BAND_H + SORT_BAND_H / 2; }

static void put_obj( uint32_t word, int x, int y )
{
    struct obj o = { word, x, y, CW, CH };

    scene[ n_obj++ ] = o;
}
static void put_at( uint32_t word, int col, int row ) { put_obj( word, slot_x( col ), slot_y( row ) ); }
static void put_held( uint32_t word, int x, int y )
{
    struct obj o = { word, x, y, HW, HH };

    scene[ n_obj++ ] = o;
}

static int find_obj( uint32_t word )
{
    unsigned i;

    for( i = 0; i < n_obj; i++ ) if( scene[ i ].word == word ) return ( int ) i;
    return -1;
}

static void take( uint32_t word )
{
    int i = find_obj( word );

    assert( i >= 0 );
    memmove( &scene[ i ], &scene[ i + 1 ], ( n_obj - ( unsigned ) i - 1 ) * sizeof scene[ 0 ] );
    n_obj--;
}

static void move_to( uint32_t word, int col, int row )
{
    int i = find_obj( word );

    assert( i >= 0 );
    scene[ i ].x = slot_x( col );
    scene[ i ].y = slot_y( row );
}

static struct scan_card box_of( const struct obj * o )
{
    struct scan_card c;

    memset( &c, 0, sizeof c );
    c.x0 = ( uint16_t ) ( o->x - o->w / 2 ); c.x1 = ( uint16_t ) ( o->x + o->w / 2 - 1 );
    c.y0 = ( uint16_t ) ( o->y - o->h / 2 ); c.y1 = ( uint16_t ) ( o->y + o->h / 2 - 1 );
    return c;
}

/* white card, then dark marks that are its own: a 6 x 8 bit pattern from its word */
static void render( void )
{
    unsigned i;
    int x, y;

    memset( frame_px, 40, sizeof frame_px );
    for( i = 0; i < n_obj; i++ ) {
        const struct obj * o = &scene[ i ];
        uint32_t h = ( o->word & 0x7Fu ) * 2654435761u;
        int x0 = o->x - o->w / 2, y0 = o->y - o->h / 2, m = o->w / 10;

        for( y = 0; y < o->h; y++ )
            for( x = 0; x < o->w; x++ ) {
                int cx = ( x - m ) * 6 / ( o->w - 2 * m ), cy = ( y - m ) * 8 / ( o->h - 2 * m ), bit;

                bit = x >= m && x < o->w - m && y >= m && y < o->h - m && ( ( h >> ( ( cy * 6 + cx ) % 32 ) ) & 1u );
                if( RES_JOKER( o->word ) ) bit = x > o->w / 6 && x < o->w - o->w / 6 && y > o->h / 7 && y < o->h * 2 / 3;
                if( y0 + y >= 0 && y0 + y < FRAME_H && x0 + x >= 0 && x0 + x < FRAME_W )
                    frame_px[ y0 + y ][ x0 + x ] = ( uint8_t ) ( bit ? 30 : 225 );
            }
    }
}

static unsigned gray( int x, int y )
{
    assert( x >= 0 && x < FRAME_W && y >= 0 && y < FRAME_H );
    return frame_px[ y ][ x ];
}

static int overlap( const struct scan_card * a, const struct scan_card * b )
{
    return a->x0 <= b->x1 && b->x0 <= a->x1 && a->y0 <= b->y1 && b->y0 <= a->y1;
}

/* locate's answer: one box per card, overlapping cards fused into one */
static void find_cards( struct scan_result * s )
{
    unsigned i, j;
    int merged = 1;

    memset( s, 0, sizeof *s );
    for( i = 0; i < n_obj; i++ ) {
        struct scan_card * c = &s->card[ s->n++ ];

        *c     = box_of( &scene[ i ] );
        c->red = RES_SUIT( scene[ i ].word ) >= 2 && !RES_JOKER( scene[ i ].word );
        c->up  = 1;
    }
    while( merged ) {
        merged = 0;
        for( i = 0; i < s->n && !merged; i++ )
            for( j = i + 1; j < s->n && !merged; j++ )
                if( overlap( &s->card[ i ], &s->card[ j ] ) ) {
                    struct scan_card * a = &s->card[ i ], * b = &s->card[ j ];

                    a->x0 = a->x0 < b->x0 ? a->x0 : b->x0; a->y0 = a->y0 < b->y0 ? a->y0 : b->y0;
                    a->x1 = a->x1 > b->x1 ? a->x1 : b->x1; a->y1 = a->y1 > b->y1 ? a->y1 : b->y1;
                    s->card[ j ] = s->card[ --s->n ];
                    merged = 1;
                }
    }
}

/* the CNN: the topmost card under the window, the policy's read count */
static unsigned n_reads, n_calls;
static uint32_t misread_next;                           /* the next VERIFY read says this instead */

static uint32_t read_card( const struct scan_card * c, int how, unsigned * reads, unsigned * agree )
{
    int      i;
    uint32_t w = 0;

    assert( !sort_in_air( c ) );                        /* a card held up is never read on the board */
    for( i = ( int ) n_obj - 1; i >= 0; i-- ) {
        struct scan_card o = box_of( &scene[ i ] );

        if( overlap( &o, c ) ) { w = scene[ i ].word; break; }
    }
    if( how == READ_VERIFY && misread_next ) { w = misread_next; misread_next = 0; }
    *reads = how == READ_VERIFY ? 1u : how == READ_FULL ? 7u : RES_JOKER( w ) || !( RES_SUIT( w ) >= 2 ) ? 3u : 2u;
    *agree = w ? *reads : 1u;
    n_reads += *reads;
    n_calls++;
    return w;
}

/* ---- the game, a still scene at a time ------------------------------------ */

static struct sort_game g;
static struct scan_result s;
static unsigned clk;                                    /* ms: a card find every 700 */

/* KEY0: app_rtos.c's read of the card held up (it never sees the board) */
static void held( uint32_t word )
{
    unsigned r = RES_JOKER( word ) || RES_SUIT( word ) < 2 ? 3u : 2u;

    n_reads += r;
    sort_held( &g, word, r, word ? r : 1u );
}

static int tick( void )
{
    clk += 700;
    return sort_tick( &g, &s, clk, read_card, gray );
}

/* the scene stands still for SORT_STILL finds; returns the busiest tick's
 * answer (TICK_EVENT if any of them acted) */
static int settle( void )
{
    int r = TICK_MOVING, k, t;

    render();
    find_cards( &s );
    for( k = 0; k < SORT_STILL; k++ ) {
        t = tick();
        if( t > r ) r = t;
    }
    return r;
}

/* the same scene for another `ms` */
static void wait_ms( unsigned ms )
{
    unsigned until = clk + ms;

    while( clk < until ) tick();
}

#define EXPECT_STR( field, want ) \
    do { if( !strstr( g.field, want ) ) { printf( "%s: \"%s\"\n  missing \"%s\"\n", #field, g.field, want ); assert( 0 ); } } while( 0 )

static void show_state( const char * what )
{
    printf( "%-34s phase %u  n %u  reads %2u | %s | %s | %s\n", what, g.phase, g.n, g.reads, g.prompt, g.hint, g.note );
}

static void draw_screen( const char * path )
{
    gui_sort_draw( &g, 1 );
    dump( path );
}

int main( int argc, char ** argv )
{
    unsigned before, k;
    int      r;

    setvbuf( stdout, NULL, _IONBF, 0 );
    gui_init();
    sort_new( &g );

    /* an empty board: nothing to read */
    r = settle();
    assert( r == TICK_EVENT && g.reads == 0 && !g.fresh && g.phase == SP_SHOW );
    EXPECT_STR( prompt, "HOLD A CARD UP" );
    show_state( "empty board" );
    assert( settle() == TICK_SAME );                    /* nothing new: no work */

    /* 9S held up, KEY0: anywhere in the spade column -- its rank's own row, row 2 */
    held( W( R9, S_SPADES ) );
    show_state( "9S held up" );
    assert( g.phase == SP_PLACE && g.tcol == 0 && g.trow == 1 );
    assert( g.n == 1 && !g.card[ 0 ].seen );           /* in the memory at once, not on the board yet */
    EXPECT_STR( prompt, "PLACE 9" );
    EXPECT_STR( hint, "ANY ROW" );
    put_at( W( R9, S_SPADES ), 0, 1 );
    before = n_reads;
    settle();
    show_state( "9S laid down" );
    assert( g.phase == SP_SHOW && g.n == 1 && g.turn == 1 && n_reads == before + 1 );  /* one read where it lies */
    draw_screen( argc > 1 ? argv[ 1 ] : NULL );
    assert( strstr( row_text( 19 ), "GOOD!" ) && strstr( row_text( 1 ), "THE BOARD" ) );
    assert( strstr( row_text( 20 ), "SEES 1 CARD, 84 PX TALL" ) );

    /* JS: above the 9 (the user's example) */
    held( W( RJ, S_SPADES ) );
    EXPECT_STR( hint, "ABOVE 9" );
    assert( g.tcol == 0 && g.trow == 0 );
    put_at( W( RJ, S_SPADES ), 0, 0 );
    settle();
    assert( g.n == 2 && g.phase == SP_SHOW );

    /* QS: no room above the J -- slide J and 9 down one first, Q in row 1 */
    held( W( RQ, S_SPADES ) );
    show_state( "QS held up" );
    assert( g.phase == SP_PLACE && g.trow == 0 );
    EXPECT_STR( hint, "SLIDE J" );
    EXPECT_STR( hint, "DOWN" );
    draw_screen( argc > 3 ? argv[ 3 ] : NULL );
    assert( strstr( row_text( 19 ), "PLACE Q" ) && strstr( row_text( 18 ), "SLIDE J" ) );
    assert( ( ram[ 4 * 80 + 8 ] & 0xFFu ) == 'Q' );      /* its name in the target slot, over the preview */
    /* the memory already slid J and 9 down one and holds the Q above them --
     * on the board panel too, with arrows by the cards to slide */
    assert( g.n == 3 && g.card[ 2 ].row == 0 && g.card[ 1 ].row == 1 && g.card[ 0 ].row == 2 && !g.card[ 2 ].seen );
    assert( ( ram[ 5 * 80 + 48 ] & 0xFFu ) == 'Q' && ( ram[ 11 * 80 + 48 ] & 0xFFu ) == 'J' &&
            ( ram[ 17 * 80 + 48 ] & 0xFFu ) == '9' );
    assert( ( ram[ 12 * 80 + 54 ] & 0xFFu ) == 0x19 && ( ram[ 18 * 80 + 54 ] & 0xFFu ) == 0x19 );
    move_to( W( R9, S_SPADES ), 0, 2 );                 /* the slide ... */
    move_to( W( RJ, S_SPADES ), 0, 1 );
    settle();
    show_state( "  slid" );
    assert( g.phase == SP_PLACE && g.n == 3 && !g.card[ 2 ].seen && g.card[ 0 ].seen && g.card[ 1 ].seen );
    put_at( W( RQ, S_SPADES ), 0, 0 );                  /* ... then the Q */
    settle();
    show_state( "  QS placed" );
    assert( g.phase == SP_SHOW && g.n == 3 && g.mistakes == 0 );

    /* 5H laid in the spade column: wrong column, then moved -- followed by its look */
    held( W( R5, S_HEARTS ) );
    assert( g.tcol == 1 );
    put_at( W( R5, S_HEARTS ), 0, 3 );
    settle();
    show_state( "5H wrong column" );
    assert( g.phase == SP_PLACE && g.mistakes == 1 );
    EXPECT_STR( prompt, "MUST GO RIGHT OF" );
    assert( settle() == TICK_SAME && g.mistakes == 1 ); /* counted once */
    move_to( W( R5, S_HEARTS ), 1, 2 );
    before = n_reads;
    settle();
    assert( g.phase == SP_SHOW && g.n == 4 && n_reads == before );

    /* KH below the 5H: wrong order, then fixed */
    held( W( RK, S_HEARTS ) );
    put_at( W( RK, S_HEARTS ), 1, 3 );
    settle();
    show_state( "KH below 5H" );
    assert( g.phase == SP_PLACE && g.mistakes == 2 );
    EXPECT_STR( prompt, "MUST GO ABOVE 5" );
    move_to( W( RK, S_HEARTS ), 1, 0 );
    settle();
    assert( g.phase == SP_SHOW && g.n == 5 );

    /* the slide and the placement in one go: 10S between J and 9 -- the 9
     * alone slides down (the fewest moves), the 10 takes its row */
    held( W( R10, S_SPADES ) );
    show_state( "10S held up" );
    assert( g.trow == 2 );
    EXPECT_STR( hint, "SLIDE 9" );
    move_to( W( R9, S_SPADES ), 0, 3 );
    put_at( W( R10, S_SPADES ), 0, 2 );
    settle();
    show_state( "  slid + placed" );
    assert( g.phase == SP_SHOW && g.n == 6 && g.mistakes == 2 );

    /* the spade column is full now: a fifth spade is set aside */
    held( W( R2, S_SPADES ) );
    show_state( "2S, column full" );
    assert( g.phase == SP_SHOW );
    EXPECT_STR( prompt, "FULL" );

    /* a held card read wrong (8C taken for 8H): it goes down in the hearts
     * column as told, the board reads 8C there -- told to move it, then kept */
    held( W( R8, S_HEARTS ) );
    assert( g.tcol == 1 && g.trow == 1 );
    put_at( W( R8, S_CLUBS ), 1, 1 );
    settle();
    show_state( "8C held up as 8H, laid" );
    assert( g.phase == SP_PLACE && g.mistakes == 3 && RES_SUIT( g.card[ g.last ].word ) == S_CLUBS );
    assert( !sort_on_board( &g, W( R8, S_HEARTS ) ) );   /* the misread 8H is out of the memory */
    EXPECT_STR( note, "THE BOARD READ 8" );
    EXPECT_STR( prompt, "MUST GO RIGHT OF" );
    move_to( W( R8, S_CLUBS ), 2, 1 );
    settle();
    assert( g.phase == SP_SHOW && g.n == 7 );

    /* a card held up over the board, bigger than a slot: not read, not judged;
     * the cards it hides are not reported either */
    before = n_reads;
    put_held( W( R7, S_DIAMONDS ), 260, 190 );
    settle();
    show_state( "7D held up over the board" );
    assert( n_reads == before && g.phase == SP_SHOW && g.mistakes == 3 && g.hiding );
    gui_sort_draw( &g, 0 );
    dump( argc > 4 ? argv[ 4 ] : NULL );
    assert( strstr( row_text( 20 ), "CARD HELD UP: READY" ) );
    wait_ms( 3000 );
    assert( g.phase == SP_SHOW && n_reads == before );  /* 3 s: still nothing */
    take( W( R7, S_DIAMONDS ) );
    settle();
    assert( g.phase == SP_SHOW && !g.hiding && n_reads == before );

    /* KEY0 with the opponent's joker held up */
    held( JOKER );
    show_state( "joker held up" );
    assert( g.phase == SP_SHOW );
    EXPECT_STR( prompt, "THAT'S A JOKER" );

    /* the opponent's joker in an empty slot */
    put_at( JOKER, 2, 2 );
    settle();
    show_state( "joker in a gap" );
    assert( g.phase == SP_ALERT && g.alert == AL_JOKER && g.acov < 0 );
    EXPECT_STR( prompt, "JOKER ON" );
    EXPECT_STR( prompt, "ROW 3" );
    draw_screen( argc > 2 ? argv[ 2 ] : NULL );
    assert( strstr( row_text( 19 ), "JOKER ON" ) && strstr( row_text( 0 ), "JOKER! REMOVE IT" ) );
    held( W( R4, S_DIAMONDS ) );                        /* KEY0 now: the alert first */
    assert( g.phase == SP_ALERT );
    assert( settle() == TICK_SAME );
    take( JOKER );
    settle();
    show_state( "  removed" );
    assert( g.phase == SP_SHOW && g.jokers == 1 );

    /* ... laid exactly over our 5H: same box, a different look */
    put_obj( JOKER, slot_x( 1 ), slot_y( 2 ) );
    settle();
    show_state( "joker exactly on 5H" );
    assert( g.phase == SP_ALERT && g.alert == AL_JOKER && g.acov >= 0 );
    EXPECT_STR( hint, "COVERS YOUR 5" );
    take( JOKER );
    settle();
    assert( g.phase == SP_SHOW && g.jokers == 2 );

    /* ... half over our QS: the boxes fuse, the Q is no longer where it was */
    put_obj( JOKER, slot_x( 0 ) + 24, slot_y( 0 ) + 5 );
    settle();
    show_state( "joker half on QS" );
    assert( g.phase == SP_ALERT && g.alert == AL_JOKER && g.acov >= 0 );
    take( JOKER );
    settle();
    assert( g.phase == SP_SHOW && g.jokers == 3 && g.n == 7 );

    /* ... while a card is waiting to be placed: the alert first, then the
     * placement instruction comes back */
    held( W( RA, S_DIAMONDS ) );
    assert( g.phase == SP_PLACE && g.tcol == 3 && g.trow == 0 );
    put_at( JOKER, 3, 2 );
    settle();
    assert( g.phase == SP_ALERT );
    take( JOKER );
    settle();
    show_state( "  joker gone, back to placing" );
    assert( g.phase == SP_PLACE );
    EXPECT_STR( prompt, "PLACE A" );
    put_at( W( RA, S_DIAMONDS ), 3, 0 );
    settle();
    assert( g.phase == SP_SHOW && g.n == 8 && g.jokers == 4 );

    /* one of ours picked up: quiet for SORT_HIDDEN_MS (a hand?), then pointed
     * at; put back a little off its old place: fine, anywhere in its slot */
    take( W( RK, S_HEARTS ) );
    before = n_reads;
    settle();
    assert( g.phase == SP_SHOW && g.hiding && n_reads == before );
    wait_ms( SORT_HIDDEN_MS );
    show_state( "KH gone 6 s" );
    assert( g.phase == SP_ALERT && g.alert == AL_THING && g.acov >= 0 );
    EXPECT_STR( prompt, "NOT SEEN" );
    put_obj( W( RK, S_HEARTS ), slot_x( 1 ) + 25, slot_y( 0 ) + 4 );
    settle();
    show_state( "  KH back, 25 px off" );
    assert( g.phase == SP_SHOW && g.n == 8 && g.jokers == 4 );
    EXPECT_STR( note, "IS BACK" );

    /* a one-read misread while matching a slid card: the full read settles it */
    held( W( R3, S_HEARTS ) );
    assert( g.tcol == 1 && g.trow == 3 );
    put_at( W( R3, S_HEARTS ), 1, 3 );
    move_to( W( R5, S_HEARTS ), 1, 1 );                 /* moved too: two candidates, two new boxes */
    misread_next = W( R8, S_HEARTS );
    settle();
    show_state( "3H + 5H moved, a misread" );
    assert( g.phase == SP_SHOW && g.n == 9 );

    /* a hand over the board: the boxes change every find, nothing is read */
    before = n_reads;
    for( k = 0; k < 6; k++ ) {
        put_obj( W( R7, S_CLUBS ), 100 + 40 * ( int ) k, 200 );
        render();
        find_cards( &s );
        assert( tick() == TICK_MOVING );
        take( W( R7, S_CLUBS ) );
    }
    assert( n_reads == before );

    /* a card laid straight on the board, no KEY0: read once (no joker), then
     * ignored while it lies there -- only KEY0 puts a card in the memory */
    put_at( W( R7, S_CLUBS ), 2, 2 );
    settle();
    show_state( "7C laid without KEY0" );
    assert( g.phase == SP_SHOW && g.n == 9 && g.nghost == 1 );
    EXPECT_STR( note, "IGNORED 1" );

    /* two more at once: both ignored, no alert -- and the first one, a ghost
     * now, is not read again (6C: 1 + 3 reads, 2D: 1 + 2) */
    before = n_reads;
    put_at( W( R6, S_CLUBS ), 2, 3 );
    put_at( W( R2, S_DIAMONDS ), 3, 3 );
    settle();
    show_state( "  + 6C + 2D, no KEY0" );
    assert( g.phase == SP_SHOW && g.n == 9 && g.nghost == 3 && n_reads == before + 7 );
    EXPECT_STR( note, "IGNORED 2" );

    /* ... then the 7C scanned as it should be: picked up (its ghost forgotten),
     * KEY0, laid again */
    take( W( R7, S_CLUBS ) );
    settle();
    assert( g.nghost == 2 );
    held( W( R7, S_CLUBS ) );
    put_at( W( R7, S_CLUBS ), 2, 2 );
    settle();
    show_state( "  7C scanned, laid again" );
    assert( g.phase == SP_SHOW && g.n == 10 );
    EXPECT_STR( note, "PLACED" );
    take( W( R2, S_DIAMONDS ) );
    take( W( R6, S_CLUBS ) );
    settle();
    assert( g.nghost == 0 );
    held( W( R6, S_CLUBS ) );
    put_at( W( R6, S_CLUBS ), 2, 3 );
    settle();
    assert( g.phase == SP_SHOW && g.n == 11 && g.jokers == 4 );

    /* something that is no card (no read): ignored, no alert */
    put_obj( 0, slot_x( 3 ), slot_y( 2 ) );
    settle();
    show_state( "a non-card" );
    assert( g.phase == SP_SHOW && g.nghost == 1 );
    EXPECT_STR( note, "IGNORED 1" );

    /* KEY1: undo the last placement -- out of the memory, so ignored where it lies */
    before = g.n;
    sort_command( &g, SORT_CMD_UNDO );
    assert( g.n == before - 1 );
    settle();
    show_state( "undone 6C, ignored" );
    assert( g.phase == SP_SHOW && g.n == before - 1 && g.nghost == 2 );
    take( 0 );
    take( W( R6, S_CLUBS ) );

    /* KEY1 tap with a card to place: cancelled */
    held( W( R5, S_DIAMONDS ) );
    assert( g.phase == SP_PLACE );
    sort_command( &g, SORT_CMD_UNDO );
    assert( g.phase == SP_SHOW );
    EXPECT_STR( note, "UNDONE" );

    /* KEY1 held: a new game wants an empty board first */
    sort_command( &g, SORT_CMD_NEW );
    settle();
    show_state( "new game, cards still down" );
    EXPECT_STR( prompt, "CLEAR THE BOARD" );
    assert( g.n == 0 && g.reads == 0 );
    n_obj = 0;
    settle();
    assert( !g.fresh && g.phase == SP_SHOW );

    /* a new game on a mat with a mark locate takes for a card: KEY1 ignores it */
    sort_command( &g, SORT_CMD_NEW );
    put_obj( 0, slot_x( 1 ), slot_y( 1 ) );
    settle();
    EXPECT_STR( prompt, "CLEAR THE BOARD" );
    sort_command( &g, SORT_CMD_UNDO );
    settle();
    show_state( "new game, a mark ignored" );
    assert( !g.fresh && g.phase == SP_SHOW && g.nghost == 1 && g.reads == 0 );
    held( W( R8, S_CLUBS ) );
    assert( g.phase == SP_PLACE && g.tcol == 2 );

    /* the board memory: cards scanned one after another, none laid down yet,
     * all stay -- each in its own slot, on the board panel too */
    held( W( RK, S_DIAMONDS ) );
    held( W( R3, S_HEARTS ) );
    show_state( "3 scanned, none laid" );
    assert( g.n == 3 && g.phase == SP_PLACE && g.last == 2 );
    for( k = 0; k < g.n; k++ ) assert( !g.card[ k ].seen );
    assert( g.card[ 0 ].col == 2 && g.card[ 1 ].col == 3 && g.card[ 2 ].col == 1 );
    draw_screen( argc > 5 ? argv[ 5 ] : NULL );
    {
        int pos[ 3 ][ 2 ] = { { 2, 1 }, { 3, 0 }, { 1, 3 } };     /* (col, row): 8C, KD, 3H */
        const char rank[ 3 ] = { '8', 'K', '3' };

        for( k = 0; k < 3; k++ )
            assert( ( ram[ ( 5 + 6 * pos[ k ][ 1 ] ) * 80 + 48 + 8 * pos[ k ][ 0 ] ] & 0xFFu ) == ( unsigned ) rank[ k ] );
        assert( strstr( row_text( 21 ), "SORTED 3 OF 16" ) && strstr( row_text( 22 ), "ON BOARD 0" ) );
    }
    /* ... then laid down, all at once: each found and kept (seen) */
    put_at( W( R8, S_CLUBS ), 2, 1 );
    put_at( W( RK, S_DIAMONDS ), 3, 0 );
    put_at( W( R3, S_HEARTS ), 1, 3 );
    settle();
    show_state( "  all 3 laid down" );
    assert( g.n == 3 && g.phase == SP_SHOW && g.mistakes == 0 );
    for( k = 0; k < g.n; k++ ) assert( g.card[ k ].seen );
    gui_sort_draw( &g, 1 );
    assert( strstr( row_text( 22 ), "ON BOARD 3" ) );
    assert( ( ram[ 5 * 80 + 47 + 8 * 3 + 5 ] & 0xFFu ) == 0xFB );     /* KD's tick: the camera found it */

    /* the camera too low: cards lying on the board look bigger than a slot and
     * the auto re-scan ignores them -- row 20 says to raise the camera */
    n_obj = 0;
    put_held( W( R5, S_CLUBS ), 100, 190 );
    put_held( W( R6, S_CLUBS ), 330, 190 );
    before = n_reads;
    settle();
    gui_sort_draw( &g, 0 );
    assert( strstr( row_text( 20 ), "2 CARDS TOO BIG" ) && strstr( row_text( 20 ), "RAISE CAMERA" ) );
    assert( n_reads == before );                        /* ignored: none read */
    take( W( R6, S_CLUBS ) );
    settle();
    gui_sort_draw( &g, 0 );
    assert( strstr( row_text( 20 ), "BIG CARD: KEY0 IF HELD" ) );

    /* the photo of 22 Oct: the camera off to one side and higher -- the board
     * in the middle of the frame, its columns 80 px apart, the spades over the
     * hearts' lines and the diamonds over the clubs'. Judged between the
     * cards: all in order, each kept in its suit's column in the memory (the
     * panel), and the 4D planned above the 2D */
    {
        static const int X[ 4 ] = { 140, 220, 300, 380 };       /* the board's columns, spades to diamonds */
        unsigned         i;

        sort_command( &g, SORT_CMD_NEW );
        n_obj = 0;
        settle();
        held( W( RK, S_SPADES ) );
        put_obj( W( RK, S_SPADES ), X[ 0 ], slot_y( 0 ) );
        settle();
        held( W( R8, S_SPADES ) );
        put_obj( W( R8, S_SPADES ), X[ 0 ], slot_y( 1 ) );
        settle();
        held( W( R2, S_DIAMONDS ) );
        put_obj( W( R2, S_DIAMONDS ), X[ 3 ], slot_y( 3 ) );
        settle();
        show_state( "board off-centre: KS 8S 2D" );
        assert( g.phase == SP_SHOW && g.n == 3 && g.mistakes == 0 );
        assert( g.card[ 0 ].col == 0 && g.card[ 1 ].col == 0 && g.card[ 2 ].col == 3 );
        held( W( R4, S_DIAMONDS ) );
        assert( g.tcol == 3 && g.trow == 2 );                   /* above the 2D */
        put_obj( W( R4, S_DIAMONDS ), X[ 3 ], slot_y( 2 ) );
        settle();
        assert( g.phase == SP_SHOW && g.n == 4 && g.mistakes == 0 );

        /* 5H laid under the 8S: out of place -- marked on the panel, in its own column */
        held( W( R5, S_HEARTS ) );
        put_obj( W( R5, S_HEARTS ), X[ 0 ], slot_y( 2 ) );
        settle();
        show_state( "  5H under the 8S" );
        assert( g.phase == SP_PLACE && g.mistakes == 1 && g.card[ 4 ].bad );
        EXPECT_STR( prompt, "MUST GO RIGHT OF K" );
        gui_sort_draw( &g, 1 );
        assert( ( ram[ 11 * 80 + 48 ] & 0xFFu ) == '8' && ( ram[ 11 * 80 + 52 ] & 0xFFu ) == 0xFB );     /* 8S: spades, a tick */
        assert( ( ram[ 23 * 80 + 72 ] & 0xFFu ) == '2' );                                              /* 2D: diamonds, row 4 */
        assert( ( ram[ 17 * 80 + 60 ] & 0xFFu ) == 0x13 );                                             /* 5H: hearts, out of place */

        /* the camera moved: every card found again, the same mistake not counted again */
        for( i = 0; i < n_obj; i++ ) scene[ i ].x -= 20;
        settle();
        show_state( "  camera moved 20 px" );
        assert( g.phase == SP_PLACE && g.mistakes == 1 && g.card[ 4 ].bad );

        /* moved right of the spades: in order */
        take( W( R5, S_HEARTS ) );
        put_obj( W( R5, S_HEARTS ), X[ 1 ] - 20, slot_y( 2 ) );
        settle();
        show_state( "  5H moved right" );
        assert( g.phase == SP_SHOW && g.n == 5 && g.mistakes == 1 && !g.card[ 4 ].bad );
        draw_screen( argc > 6 ? argv[ 6 ] : NULL );
        assert( ( ram[ 17 * 80 + 60 ] & 0xFFu ) == 0xFB );
    }

    /* problem2.jpeg: things the finder boxes on the board that are no card of
     * ours -- glare, a mark, a card nobody scanned, a second copy of one of
     * ours -- were adopted as cards and stayed with !! for good. Now each is
     * read once (no joker), then ignored: never in the memory, no !!, not
     * read again, grey corners on the preview */
    {
        unsigned n0 = g.n, m0 = g.mistakes;

        put_obj( 0, 280, slot_y( 0 ) );                         /* the board's clubs column is empty */
        put_obj( W( RA, S_SPADES ), 280, slot_y( 1 ) );
        put_obj( W( R4, S_DIAMONDS ), 280, slot_y( 3 ) );       /* a second 4D */
        settle();
        show_state( "problem2: 3 things not ours" );
        assert( g.phase == SP_SHOW && g.n == n0 && g.mistakes == m0 && g.nghost == 3 );
        for( k = 0; k < g.n; k++ ) assert( !g.card[ k ].bad );
        EXPECT_STR( note, "IGNORED 3" );

        /* the next turn reads the card laid, not the three */
        held( W( RQ, S_HEARTS ) );
        put_obj( W( RQ, S_HEARTS ), 200, slot_y( 0 ) );
        before = n_reads;
        settle();
        show_state( "  QH placed, the 3 not read again" );
        assert( g.phase == SP_SHOW && g.n == n0 + 1 && n_reads == before + 1 && g.nghost == 3 );
        draw_screen( argc > 7 ? argv[ 7 ] : NULL );
        assert( ( ( ram[ 3 * 80 + 20 ] >> 8 ) & 0xFu ) == GUI_DGREY );     /* the non-card's corner: grey */
        assert( ( ( ram[ 3 * 80 + 15 ] >> 8 ) & 0xFu ) == GUI_WHITE );     /* the QH's: white */
        assert( strstr( row_text( 22 ), "ON BOARD 6" ) );

        /* a scanned card laid where a ghost lay: it looks different, so it is read */
        held( W( R7, S_CLUBS ) );
        take( W( RA, S_SPADES ) );
        put_obj( W( R7, S_CLUBS ), 280, slot_y( 1 ) );
        settle();
        show_state( "  7C laid on the AS's spot" );
        assert( g.phase == SP_SHOW && g.n == n0 + 2 && g.nghost == 2 );
    }

    printf( "sort_host_test: all checks passed (%u CNN calls, %u reads)\n", n_calls, n_reads );
    return 0;
}
