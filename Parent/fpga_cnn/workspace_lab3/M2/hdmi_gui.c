/*
 * HDMI dashboard -- see hdmi_gui.h. Drawn as a card table: green felt inside a
 * wooden rail. Screen map, 80x30 cells of 8x16 px:
 *
 *   row  0                 rail: title + status pill
 *   rows 1-17   cols 0-41  LIVE CAMERA, set into the rail. The preview inside
 *                          it is the fabric's (camera_capture.v PV_X/PV_Y =
 *                          exactly cells 1..40 x 2..16); the layer only frames
 *                          it, and must leave those cells blank.
 *   rows 1-10   cols 44-78 LAST CARD: the card, pips laid out as printed, and
 *                          its numbers
 *   rows 12-17  cols 44-72 TABLE: seen / unseen / repeats / jokers / reads
 *   row  18                aiming hint
 *   rows 19-24             THE BOARD: the last 5 tracked cards, dealt left to
 *                          right (face down until read), a marker over the newest
 *   rows 25-28             the deck, 4 suits x 13 ranks: a chip turns white
 *                          once seen, brass if seen more than once
 *   row  29                rail: key / switch help
 *
 * The colours are text_overlay.v's CGA palette, so a new look needs no
 * recompile -- but white on its felt green is only ~3:1, so card data sits on
 * white card faces and brass plates, and red is never printed on the felt.
 *
 * Text layer registers (ghrd_top.v): img_wr_data = the cell word
 * {bg[15:12], fg[11:8], cp437[7:0]}; img_wr_addr[11:0] = row*80 + col, and
 * img_wr_addr[13] = write enable, a level. So one cell is three accesses: the
 * data, the address with the enable, the address without it. img_wr_ctrl is
 * not touched -- it is CPU1's (infer()'s colour override).
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "app_config.h"
#include "card_pipeline.h"
#include "hdmi_gui.h"

#ifdef GUI_HOST
void alt_write_word( uint32_t addr, uint32_t value );   /* gui_host_test.c emulates the PIOs */
#else
#include "socal.h"
#endif

#define TXT_DATA_PIO_BASE   0xFF200010u     /* img_wr_data */
#define TXT_ADDR_PIO_BASE   0xFF200020u     /* img_wr_addr */
#define TXT_WE              0x2000u         /* img_wr_addr[13] */

#define COLS        80
#define ROWS        30

/* the table */
#define FELT        GUI_GREEN
#define WOOD        GUI_BROWN
#define BRASS       GUI_YELLOW
#define INK         GUI_WHITE       /* text printed on the felt */

/* panels: top-left cell */
#define CAM_ROW     1               /* its frame; the preview is cells 1..40 x 2..16 */
#define CAM_COL     0
#define CAM_H       17
#define CAM_W       42
#define LAST_ROW    1               /* plate; the card is the BIG_H rows under it */
#define LAST_COL    44
#define INFO_COL    59              /* the card's numbers, to column 78 */
#define INFO_W      ( COLS - 1 - INFO_COL )
#define STAT_ROW    12              /* plate; five lines under it */
#define STAT_COL    44
#define HINT_ROW    18
#define BOARD_ROW   19              /* plate + line; the cards are the MINI_H rows under it */
#define BOARD_LINE  15              /* where the line starts, after the plate */
#define BOARD_N     5
#define BOARD_STEP  9               /* MINI_W + a 2-cell gap */
#define BOARD_COL   ( ( COLS - ( BOARD_N * BOARD_STEP - 2 ) ) / 2 )
#define DECK_ROW    25
#define DECK_COL    ( ( COLS - ( 4 + 13 * 4 - 1 ) ) / 2 )   /* suit chip, then 13 rank chips 4 apart */
#define STATUS_COL  50              /* the status pill is right-aligned in columns 50..78 */
#define STATUS_MAX  ( COLS - 3 - STATUS_COL )

/* card sizes in cells: with 8x16 px cells both are a card's 5:7 */
#define BIG_W       13
#define BIG_H       9
#define MINI_W      7
#define MINI_H      5

/* CP437 */
#define CH_SMILE    0x01
#define CH_CHIP     0x0A            /* inverse circle: a poker chip */
#define CH_DOWN     0x1F
#define CH_SHADE    0xB1            /* card back pattern */
#define CH_LOWER    0xDC            /* lower half block */
#define CH_UPPER    0xDF            /* upper half block */
#define BX_H        0xC4
#define BX_V        0xB3
#define BX_TL       0xDA
#define BX_TR       0xBF
#define BX_BL       0xC0
#define BX_BR       0xD9

/* suit index 0..3 = Spades, Clubs, Hearts, Diamonds (cnn_result) */
static const unsigned char SUIT_CH[4] = { 0x06, 0x05, 0x03, 0x04 };
#define SUIT_RED(s) ( ( s ) >= 2u )

/* Where a printed card puts its pips, by rank index (2..10, J, Q, K, A): face
 * rows 1..7 x columns left / centre / right. The courts have none -- they get
 * a framed letter instead. */
#define P( r, c )   ( 1ul << ( ( ( r ) - 1 ) * 3 + ( c ) ) )
static const unsigned long PIPS[ 13 ] = {
    P( 1, 1 ) | P( 7, 1 ),                                                          /* 2  */
    P( 1, 1 ) | P( 4, 1 ) | P( 7, 1 ),                                              /* 3  */
    P( 1, 0 ) | P( 1, 2 ) | P( 7, 0 ) | P( 7, 2 ),                                  /* 4  */
    P( 1, 0 ) | P( 1, 2 ) | P( 4, 1 ) | P( 7, 0 ) | P( 7, 2 ),                      /* 5  */
    P( 1, 0 ) | P( 1, 2 ) | P( 4, 0 ) | P( 4, 2 ) | P( 7, 0 ) | P( 7, 2 ),          /* 6  */
    P( 1, 0 ) | P( 1, 2 ) | P( 2, 1 ) | P( 4, 0 ) | P( 4, 2 ) | P( 7, 0 ) | P( 7, 2 ),              /* 7 */
    P( 1, 0 ) | P( 1, 2 ) | P( 2, 1 ) | P( 4, 0 ) | P( 4, 2 ) | P( 6, 1 ) | P( 7, 0 ) | P( 7, 2 ),  /* 8 */
    P( 1, 0 ) | P( 1, 2 ) | P( 3, 0 ) | P( 3, 2 ) | P( 4, 1 ) | P( 5, 0 ) | P( 5, 2 ) |
        P( 7, 0 ) | P( 7, 2 ),                                                      /* 9  */
    P( 1, 0 ) | P( 1, 2 ) | P( 2, 1 ) | P( 3, 0 ) | P( 3, 2 ) | P( 5, 0 ) | P( 5, 2 ) |
        P( 6, 1 ) | P( 7, 0 ) | P( 7, 2 ),                                          /* 10 */
    0, 0, 0,                                                                        /* J Q K */
    P( 4, 1 ),                                                                      /* A  */
};

/* ponytail: the history keeps the last HIST_MAX tracked cards, and the board,
 * the deck and the counts are recomputed from it on every redraw, so an undo
 * can never leave them disagreeing. Past HIST_MAX the oldest cards drop off;
 * 256 is about five decks. Raise it, or keep separate counts, if a run is ever
 * longer. */
#define HIST_MAX    256

struct det
{
    uint32_t     result;        /* raw cnn_result word */
    uint32_t     red_count;
    uint32_t     inf_ms;
    unsigned     shot;
    const char * where;         /* "CPU1" / "CPU0" / NULL */
};

static struct det hist[ HIST_MAX ];
static unsigned   n_hist;
static struct det last;         /* LAST CARD: the newest tracked card, or an auto capture */
static int        last_valid, last_auto;

/* ---- drawing primitives ------------------------------------------------- */

static void put( int row, int col, unsigned ch, unsigned fg, unsigned bg )
{
    unsigned a;

    if( row < 0 || row >= ROWS || col < 0 || col >= COLS ) return;
    a = ( unsigned ) ( row * COLS + col );
    alt_write_word( TXT_DATA_PIO_BASE, ( bg << 12 ) | ( fg << 8 ) | ( ch & 0xFFu ) );
    alt_write_word( TXT_ADDR_PIO_BASE, TXT_WE | a );
    alt_write_word( TXT_ADDR_PIO_BASE, a );
}

/* `s` from (row, col), padded with spaces to `w` cells (w = 0: no padding, no
 * limit). Returns the column after it. */
static int text( int row, int col, int w, const char * s, unsigned fg, unsigned bg )
{
    int c = col;

    while( *s && ( w == 0 || c < col + w ) ) put( row, c++, ( unsigned char ) *s++, fg, bg );
    while( c < col + w ) put( row, c++, ' ', fg, bg );
    return c;
}

static void fill( int row, int col, int h, int w, unsigned ch, unsigned fg, unsigned bg )
{
    int r, c;

    for( r = row; r < row + h; r++ )
        for( c = col; c < col + w; c++ ) put( r, c, ch, fg, bg );
}

static void frame( int row, int col, int h, int w, unsigned fg, unsigned bg )
{
    int r;

    put( row, col, BX_TL, fg, bg );
    fill( row, col + 1, 1, w - 2, BX_H, fg, bg );
    put( row, col + w - 1, BX_TR, fg, bg );
    for( r = row + 1; r < row + h - 1; r++ ) {
        put( r, col, BX_V, fg, bg );
        put( r, col + w - 1, BX_V, fg, bg );
    }
    put( row + h - 1, col, BX_BL, fg, bg );
    fill( row + h - 1, col + 1, 1, w - 2, BX_H, fg, bg );
    put( row + h - 1, col + w - 1, BX_BR, fg, bg );
}

/* ---- cards -------------------------------------------------------------- */

/* A card-shaped h x w block of `face`: its corner cells are half blocks, so the
 * corners read as rounded against the felt. */
static void card_shape( int row, int col, int h, int w, unsigned face )
{
    fill( row, col, h, w, ' ', GUI_BLACK, face );
    put( row, col, CH_LOWER, face, FELT );
    put( row, col + w - 1, CH_LOWER, face, FELT );
    put( row + h - 1, col, CH_UPPER, face, FELT );
    put( row + h - 1, col + w - 1, CH_UPPER, face, FELT );
}

/* face down: a board slot not dealt yet, or LAST CARD before the first read */
static void card_back( int row, int col, int h, int w )
{
    card_shape( row, col, h, w, GUI_BLUE );
    fill( row + 1, col + 1, h - 2, w - 2, CH_SHADE, GUI_LBLUE, GUI_BLUE );
}

static unsigned ink_of( uint32_t r )        /* its printing colour on a white face */
{
    if( RES_JOKER( r ) ) return GUI_MAGENTA;
    return SUIT_RED( RES_SUIT( r ) ) ? GUI_RED : GUI_BLACK;
}

static unsigned pip_of( uint32_t r )
{
    return RES_JOKER( r ) ? CH_SMILE : SUIT_CH[ RES_SUIT( r ) ];
}

static const char * rank_of( uint32_t r )
{
    return RES_JOKER( r ) ? "" : RANK_NAMES[ RES_RANK( r ) ];
}

/* LAST CARD, BIG_W x BIG_H: the index in two corners, then the pips where a
 * printed card has them -- or, for a court card or the joker, a framed
 * centrepiece. */
static void draw_big( int row, int col, uint32_t r )
{
    unsigned ink = ink_of( r ), pip = pip_of( r ), k;
    const char * rank = rank_of( r );
    unsigned long pips = RES_JOKER( r ) ? 0ul : PIPS[ RES_RANK( r ) ];

    card_shape( row, col, BIG_H, BIG_W, GUI_WHITE );
    text( row + 1, col + 1, 0, rank, ink, GUI_WHITE );
    put( row + 2, col + 1, pip, ink, GUI_WHITE );
    put( row + BIG_H - 3, col + BIG_W - 2, pip, ink, GUI_WHITE );
    text( row + BIG_H - 2, col + BIG_W - 1 - ( int ) strlen( rank ), 0, rank, ink, GUI_WHITE );
    if( pips ) {
        for( k = 0; k < 21u; k++ )
            if( pips & ( 1ul << k ) )
                put( row + 1 + ( int ) ( k / 3u ), col + 4 + 2 * ( int ) ( k % 3u ), pip, ink, GUI_WHITE );
        return;
    }
    frame( row + 2, col + 3, 5, 7, ink, GUI_WHITE );
    put( row + 3, col + 6, pip, ink, GUI_WHITE );
    if( RES_JOKER( r ) ) text( row + 4, col + 4, 0, "JOKER", ink, GUI_WHITE );
    else                 text( row + 4, col + 6, 0, rank, ink, GUI_WHITE );
    put( row + 5, col + 6, pip, ink, GUI_WHITE );
}

/* A board card, MINI_W x MINI_H: the rank in two corners, one pip in the middle. */
static void draw_mini( int row, int col, uint32_t r )
{
    unsigned ink = ink_of( r );
    const char * rank = rank_of( r );

    card_shape( row, col, MINI_H, MINI_W, GUI_WHITE );
    if( RES_JOKER( r ) ) {
        put( row + 1, col + 1, CH_SMILE, ink, GUI_WHITE );
        text( row + 2, col + 1, 0, "JOKER", ink, GUI_WHITE );
        put( row + 3, col + MINI_W - 2, CH_SMILE, ink, GUI_WHITE );
        return;
    }
    text( row + 1, col + 1, 0, rank, ink, GUI_WHITE );
    put( row + 2, col + MINI_W / 2, pip_of( r ), ink, GUI_WHITE );
    text( row + 3, col + MINI_W - 1 - ( int ) strlen( rank ), 0, rank, ink, GUI_WHITE );
}

/* ---- panels ------------------------------------------------------------- */

static const char * card_name( uint32_t r, char * buf, unsigned n )
{
    if( RES_JOKER( r ) ) return "JOKER";
    snprintf( buf, n, "%s of %s", RANK_NAMES[ RES_RANK( r ) ], SUIT_NAMES[ RES_SUIT( r ) ] );
    return buf;
}

static void draw_last( void )
{
    char buf[ 32 ], name[ 24 ];
    const struct det * d = last_valid ? &last : NULL;
    int row = LAST_ROW + 2;

    if( !d ) {
        card_back( LAST_ROW + 1, LAST_COL, BIG_H, BIG_W );
        text( row, INFO_COL, INFO_W, "press KEY0", INK, FELT );
        text( row + 1, INFO_COL, INFO_W, "to read a card", INK, FELT );
        fill( row + 2, INFO_COL, 4, INFO_W, ' ', INK, FELT );
        return;
    }
    draw_big( LAST_ROW + 1, LAST_COL, d->result );
    text( row, INFO_COL, INFO_W, card_name( d->result, name, sizeof name ), INK, FELT );
    snprintf( buf, sizeof buf, "logit %d", RES_SCORE( d->result ) );
    text( row + 1, INFO_COL, INFO_W, buf, INK, FELT );
    snprintf( buf, sizeof buf, "colour %s (%lu)", RES_COLOUR( d->result ) ? "red" : "black",
              ( unsigned long ) d->red_count );
    text( row + 2, INFO_COL, INFO_W, buf, INK, FELT );
    snprintf( buf, sizeof buf, "infer %lu ms%s%s", ( unsigned long ) d->inf_ms,
              d->where ? " " : "", d->where ? d->where : "" );
    text( row + 3, INFO_COL, INFO_W, buf, INK, FELT );
    snprintf( buf, sizeof buf, "shot #%u", d->shot );
    text( row + 4, INFO_COL, INFO_W, buf, INK, FELT );
    if( last_auto ) text( row + 5, INFO_COL, INFO_W, " auto: not tracked", GUI_BLACK, BRASS );
    else            fill( row + 5, INFO_COL, 1, INFO_W, ' ', INK, FELT );
}

/* THE BOARD: the last BOARD_N tracked cards, oldest on the left */
static void draw_board( void )
{
    unsigned first = n_hist > BOARD_N ? n_hist - BOARD_N : 0u, i;

    fill( BOARD_ROW, BOARD_LINE, 1, COLS - 3 - BOARD_LINE, BX_H, BRASS, FELT );
    for( i = 0; i < BOARD_N; i++ ) {
        int col = BOARD_COL + BOARD_STEP * ( int ) i;

        if( first + i < n_hist ) draw_mini( BOARD_ROW + 1, col, hist[ first + i ].result );
        else                     card_back( BOARD_ROW + 1, col, MINI_H, MINI_W );
        if( first + i + 1 == n_hist ) put( BOARD_ROW, col + MINI_W / 2, CH_DOWN, BRASS, FELT );
    }
}

/* the deck grid, and the TABLE counts that come from the same tally */
static void draw_deck( void )
{
    static const unsigned char CHIP[ 4 ] = { GUI_WHITE, GUI_BLUE, GUI_BLACK, BRASS };
    unsigned char cnt[ 4 ][ 13 ];
    unsigned i, s, k, seen = 0, repeats = 0, jokers = 0;
    char buf[ 24 ];
    int row = STAT_ROW + 1;

    memset( cnt, 0, sizeof cnt );
    for( i = 0; i < n_hist; i++ ) {
        uint32_t r = hist[ i ].result;
        if( RES_JOKER( r ) ) jokers++;
        else if( cnt[ RES_SUIT( r ) ][ RES_RANK( r ) ] < 255u ) cnt[ RES_SUIT( r ) ][ RES_RANK( r ) ]++;
    }
    for( s = 0; s < 4; s++ ) {
        for( k = 0; k < 13; k++ ) {
            int r = DECK_ROW + ( int ) s, c = DECK_COL + 4 + 4 * ( int ) k;
            unsigned n = cnt[ s ][ k ];

            seen    += n != 0;
            repeats += n > 1 ? n - 1 : 0;   /* one deck has one of each: likely misreads */
            snprintf( buf, sizeof buf, "%2s ", RANK_NAMES[ k ] );
            if( n ) text( r, c, 3, buf, SUIT_RED( s ) ? GUI_RED : GUI_BLACK, n > 1 ? BRASS : GUI_WHITE );
            else    text( r, c, 3, buf, GUI_LGREEN, FELT );     /* printed on the felt */
        }
    }

    snprintf( buf, sizeof buf, "SEEN    %2u/52", seen );
    text( row, STAT_COL, 16, buf, INK, FELT );
    for( k = 0; k < 13; k++ )               /* a chip for every 4 cards seen */
        put( row, STAT_COL + 16 + ( int ) k, 4 * k < seen ? CH_CHIP : ' ', CHIP[ k % 4 ], FELT );
    snprintf( buf, sizeof buf, "UNSEEN  %2u", 52 - seen );
    text( ++row, STAT_COL, 16, buf, INK, FELT );
    snprintf( buf, sizeof buf, "REPEATS %2u", repeats );
    text( ++row, STAT_COL, 16, buf, INK, FELT );
    snprintf( buf, sizeof buf, "JOKERS  %2u", jokers );
    text( ++row, STAT_COL, 16, buf, INK, FELT );
    snprintf( buf, sizeof buf, "READS   %2u", n_hist );
    text( ++row, STAT_COL, 16, buf, INK, FELT );
}

static void show_newest( void )
{
    last_valid = n_hist > 0;
    last_auto  = 0;
    if( last_valid ) last = hist[ n_hist - 1 ];
    draw_last();
    draw_board();
    draw_deck();
}

/* ---- API ---------------------------------------------------------------- */

void gui_status( const char * msg, enum gui_colour colour )
{
    int n = ( int ) strlen( msg ), c;

    if( n > STATUS_MAX ) n = STATUS_MAX;
    c = COLS - 3 - n;                       /* the pill ends in column 78 */
    fill( 0, STATUS_COL, 1, c - STATUS_COL, ' ', INK, WOOD );
    put( 0, c, ' ', GUI_BLACK, colour );
    text( 0, c + 1, n, msg, GUI_BLACK, colour );
    put( 0, c + 1 + n, ' ', GUI_BLACK, colour );
}

void gui_init( void )
{
    static const struct { const char * key, * what; } HELP[] = {
        { "KEY0", "capture" }, { "KEY1", "undo (hold: clear)" }, { "SW0", "ascii" }, { "SW1", "pgm" },
#if RTOS_MODE
        { "SW2", "stats" }, { "SW3", "auto" },
#endif
    };
    unsigned k, s;
    int r, col;

    fill( 0, 0, ROWS, COLS, ' ', INK, FELT );

    /* the rail: bars top and bottom, both sides, the felt's corners rounded off */
    fill( 0, 0, 1, COLS, ' ', INK, WOOD );
    fill( ROWS - 1, 0, 1, COLS, ' ', INK, WOOD );
    for( r = 1; r < ROWS - 1; r++ ) {
        put( r, 0, ' ', INK, WOOD );
        put( r, COLS - 1, ' ', INK, WOOD );
    }
    put( 1, COLS - 2, CH_LOWER, FELT, WOOD );
    put( HINT_ROW, 1, CH_LOWER, FELT, WOOD );           /* under the camera */
    put( ROWS - 2, 1, CH_UPPER, FELT, WOOD );
    put( ROWS - 2, COLS - 2, CH_UPPER, FELT, WOOD );

    col = text( 0, 2, 0, "\x06\x03 ECE4813  CARD TABLE \x05\x04", BRASS, WOOD );
    text( 0, col + 2, 0, "DE10-Nano CNN", INK, WOOD );

    /* the camera, set into the rail: wood with a brass pinstripe round the preview */
    frame( CAM_ROW, CAM_COL, CAM_H, CAM_W, BRASS, WOOD );
    text( CAM_ROW, CAM_COL + 3, 0, " LIVE CAMERA ", GUI_BLACK, BRASS );
    text( HINT_ROW, 2, CAM_W - 3, "card corner just inside the green box", INK, FELT );

    text( LAST_ROW, LAST_COL, 0, " LAST CARD ", GUI_BLACK, BRASS );
    text( STAT_ROW, STAT_COL, 0, " TABLE ", GUI_BLACK, BRASS );
    text( BOARD_ROW, 3, 0, " THE BOARD ", GUI_BLACK, BRASS );
    for( s = 0; s < 4; s++ ) {
        put( DECK_ROW + ( int ) s, DECK_COL, ' ', GUI_BLACK, GUI_WHITE );
        put( DECK_ROW + ( int ) s, DECK_COL + 1, SUIT_CH[ s ], SUIT_RED( s ) ? GUI_RED : GUI_BLACK, GUI_WHITE );
        put( DECK_ROW + ( int ) s, DECK_COL + 2, ' ', GUI_BLACK, GUI_WHITE );
    }

    col = 1;
    for( k = 0; k < sizeof HELP / sizeof HELP[ 0 ]; k++ ) {
        col = text( ROWS - 1, col, 0, HELP[ k ].key, BRASS, WOOD ) + 1;
        col = text( ROWS - 1, col, 0, HELP[ k ].what, INK, WOOD ) + 2;
    }

    n_hist = 0;
    show_newest();
    gui_status( "READY", GUI_LGREEN );
}

void gui_job_done( unsigned shot, const struct vision_result * v, int track, const char * where )
{
    struct det d;

    switch( v->status ) {
    case VIS_NO_TRIGGER:    gui_status( "CAMERA STOPPED", GUI_LRED );  return;
    case VIS_CAP_TIMEOUT:   gui_status( "NO CAMERA FRAME", GUI_LRED ); return;
    case VIS_INFER_TIMEOUT: gui_status( "CNN TIMEOUT", GUI_LRED );     return;
    default: break;
    }
    if( RES_DDR_ERR( v->result ) )                            { gui_status( "DDR3 PORT ERROR", GUI_LRED ); return; }
    /* a rank the 13-way head cannot produce: never index the name tables with it */
    if( !RES_JOKER( v->result ) && RES_RANK( v->result ) >= 13u ) { gui_status( "BAD RESULT", GUI_LRED ); return; }
    if( v->maxv == 0u )                                       { gui_status( "BLANK FRAME", GUI_YELLOW ); return; }

    d.result    = v->result;
    d.red_count = v->red_count;
    d.inf_ms    = v->inf_us / 1000u;
    d.shot      = shot;
    d.where     = where;

    last       = d;
    last_valid = 1;
    last_auto  = !track;
    if( track ) {
        if( n_hist == HIST_MAX ) {
            memmove( hist, hist + 1, ( HIST_MAX - 1 ) * sizeof hist[ 0 ] );
            n_hist--;
        }
        hist[ n_hist++ ] = d;
        draw_board();
        draw_deck();
    }
    draw_last();
    gui_status( track ? "READY" : "READY (auto)", GUI_LGREEN );
}

void gui_undo( void )
{
    char msg[ 24 ];

    if( !n_hist ) {
        gui_status( "NOTHING TO UNDO", GUI_YELLOW );
        return;
    }
    n_hist--;
    snprintf( msg, sizeof msg, "UNDID #%u", hist[ n_hist ].shot );
    show_newest();
    gui_status( msg, GUI_YELLOW );
}

void gui_clear( void )
{
    n_hist = 0;
    show_newest();
    gui_status( "TRACKER CLEARED", GUI_YELLOW );
}
