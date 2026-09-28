/*
 * HDMI dashboard -- see hdmi_gui.h. Drawn as a card table: green felt inside a
 * wooden rail. Screen map, 80x30 cells of 8x16 px:
 *
 *   row  0                 rail: title + status pill
 *   rows 1-17   cols 0-41  LIVE CAMERA, set into the rail. The preview inside
 *                          it is the fabric's (camera_capture.v PV_X/PV_Y =
 *                          exactly cells 1..40 x 2..16); the layer only frames
 *                          it, and must leave those cells blank.
 *   row  18                aiming hint
 *   rows 19-28  cols 1-41  the prompt (what to do next), then the round's
 *                          progress: pairs found, or turn / team / scores
 *   rows 1-28   cols 43-78 THE GRID: the N x N board (grid.c), card size
 *                          scaled to N -- empty slots, the next cell in brass,
 *                          cards face up, pairs on brass faces, backs face down
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
#include "grid.h"
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
#define HINT_ROW    18
#define INFO_ROW    19              /* the prompt plate; the round's lines under it */
#define INFO_COL    2
#define INFO_W      39
#define GRID_ROW    1               /* plate; column letters and the cards under it */
#define GRID_COL    43              /* the panel runs to column 78 */
#define STATUS_COL  50              /* the status pill is right-aligned in columns 50..78 */
#define STATUS_MAX  ( COLS - 3 - STATUS_COL )

/* card size and pitch in cells by N = 3, 4, 5 (8x16 px cells, so all three are
 * card-shaped), and the panel's card area: columns 46..78, rows 3..28 */
static const struct { int w, h, px, py; } GEO[ 3 ] = {
    { 9, 7, 11, 8 }, { 7, 5, 8, 6 }, { 5, 4, 6, 5 },
};
#define AREA_COL    46
#define AREA_W      33
#define AREA_ROW    3
#define AREA_H      26

/* CP437 */
#define CH_SMILE    0x01
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

static struct grid g;               /* the board being played */

/* ponytail: KEY1 undoes the last UNDO_MAX reads by restoring whole snapshots of
 * the board -- simple and exact, ~3 KB. Deeper history, if a run ever needs it,
 * is a bigger ring. */
#define UNDO_MAX    8
static struct grid undo_ring[ UNDO_MAX ];
static unsigned    undo_head, n_undo;

struct det                          /* the latest read, for the INFO panel */
{
    uint32_t     result;
    unsigned     shot;
    const char * where;             /* "CPU1" / "CPU0" / NULL */
    int          placed;            /* the cell it went to, -1 = none (auto / rejected) */
};
static struct det last;
static int        last_valid;

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

/* face down; `edge` is blue, or brass when it is the cell to turn up next */
static void card_back( int row, int col, int h, int w, unsigned edge )
{
    card_shape( row, col, h, w, edge );
    fill( row + 1, col + 1, h - 2, w - 2, CH_SHADE, GUI_LBLUE, GUI_BLUE );
}

static unsigned ink_of( uint32_t r )        /* its printing colour on a card face */
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

/* A card face up, any of the three grid sizes: the rank in the corners and a
 * pip in the middle; the 9x7 card adds the suit under each index, the 5x4 one
 * has room only for "rank+suit" over a single pip. A nonzero `tag` is its pair
 * number, bottom left, so the two cards of a pair can be told from the others. */
static void card_face( int row, int col, int h, int w, uint32_t r, unsigned face, unsigned tag )
{
    unsigned ink = ink_of( r ), pip = pip_of( r );
    const char * rank = rank_of( r );
    int len = ( int ) strlen( rank );

    card_shape( row, col, h, w, face );
    if( tag ) {
        char t[ 4 ];

        snprintf( t, sizeof t, h < 5 ? "%u" : "#%u", tag );
        text( row + ( h < 5 ? 2 : h - 2 ), col + ( h < 5 ? 0 : 1 ), 0, t, GUI_BLUE, face );
    }
    if( RES_JOKER( r ) ) {
        put( row + 1, col + 1, CH_SMILE, ink, face );
        if( w >= 7 ) text( row + h / 2, col + ( w - 5 ) / 2, 0, "JOKER", ink, face );
        else         text( row + h / 2, col + ( w - 3 ) / 2, 0, "JKR", ink, face );
        return;
    }
    text( row + 1, col + 1, 0, rank, ink, face );
    if( h < 5 ) {
        put( row + 1, col + 1 + len, pip, ink, face );
        put( row + 2, col + w / 2, pip, ink, face );
        return;
    }
    if( h >= 7 ) {
        put( row + 2, col + 1, pip, ink, face );
        put( row + h - 3, col + w - 2, pip, ink, face );
    }
    put( row + h / 2, col + w / 2, pip, ink, face );
    text( row + h - 2, col + w - 1 - len, 0, rank, ink, face );
}

/* a cell with nothing read yet: an outline and its name, brass if it is next */
static void card_slot( int row, int col, int h, int w, const char * name, unsigned fg )
{
    fill( row, col, h, w, ' ', fg, FELT );
    frame( row, col, h, w, fg, FELT );
    text( row + ( h - 1 ) / 2, col + ( w - 2 ) / 2, 0, name, fg, FELT );
}

/* ---- the grid ----------------------------------------------------------- */

static const char * card_name( uint32_t r, char * buf, unsigned n )
{
    if( RES_JOKER( r ) ) return "JOKER";
    snprintf( buf, n, "%s of %s", RANK_NAMES[ RES_RANK( r ) ], SUIT_NAMES[ RES_SUIT( r ) ] );
    return buf;
}

static const char * short_rank( uint32_t r )    /* "7", "10", "JK" */
{
    return RES_JOKER( r ) ? "JK" : RANK_NAMES[ RES_RANK( r ) ];
}

/* the top-left cell of card `i` on the current board */
static void cell_pos( int i, int * row, int * col )
{
    int k = g.n - 3;
    int gw = ( g.n - 1 ) * GEO[ k ].px + GEO[ k ].w, gh = ( g.n - 1 ) * GEO[ k ].py + GEO[ k ].h;

    *col = AREA_COL + ( AREA_W - gw ) / 2 + ( i % g.n ) * GEO[ k ].px;
    *row = AREA_ROW + 1 + ( AREA_H - 1 - gh ) / 2 + ( i / g.n ) * GEO[ k ].py;
}

static void draw_cell( int i )
{
    const struct grid_cell * c = &g.cell[ i ];
    int k = g.n - 3, w = GEO[ k ].w, h = GEO[ k ].h, row, col;
    char name[ 3 ];

    cell_pos( i, &row, &col );
    grid_cell_name( &g, i, name );
    switch( c->state ) {
    case CELL_EMPTY:   card_slot( row, col, h, w, name, i == g.target ? BRASS : GUI_LGREEN ); break;
    case CELL_DOWN:    card_back( row, col, h, w, i == g.target ? BRASS : GUI_BLUE );         break;
    case CELL_OPEN:    card_face( row, col, h, w, c->result, c->pair ? BRASS : GUI_WHITE, c->pair ); break;
    case CELL_MATCHED: card_face( row, col, h, w, c->result, BRASS, 0 );                           break;
    default: break;
    }
}

/* the whole panel: after a new board, since the card size depends on N */
static void draw_board( void )
{
    char buf[ 24 ];
    int i, row, col;

    fill( GRID_ROW, GRID_COL, ROWS - 2, COLS - 1 - GRID_COL, ' ', INK, FELT );
    put( 1, COLS - 2, CH_LOWER, FELT, WOOD );           /* the felt's rounded corners */
    put( ROWS - 2, COLS - 2, CH_UPPER, FELT, WOOD );
    snprintf( buf, sizeof buf, " %s %dx%d ", g.mode == GRID_SCAN ? "FACE-UP" : "FACE-DOWN GAME", g.n, g.n );
    text( GRID_ROW, GRID_COL + 1, 0, buf, GUI_BLACK, BRASS );
    for( i = 0; i < g.n; i++ ) {
        cell_pos( i, &row, &col );                      /* top row: the column letters */
        put( row - 1, col + GEO[ g.n - 3 ].w / 2, ( unsigned ) ( 'A' + i ), INK, FELT );
        cell_pos( i * g.n, &row, &col );                /* left column: the row numbers */
        put( row + ( GEO[ g.n - 3 ].h - 1 ) / 2, col - 2, ( unsigned ) ( '1' + i ), INK, FELT );
    }
    for( i = 0; i < g.n * g.n; i++ ) draw_cell( i );
}

/* ---- INFO: the prompt and the round's progress -------------------------- */

static void draw_info( void )
{
    char buf[ 48 ], a[ 3 ], b[ 3 ];
    int  i, k, row = INFO_ROW + 2, cells = g.n * g.n;

    /* the prompt: what the operator does next */
    if( g.mode == GRID_SCAN ) {
        grid_cell_name( &g, g.target < 0 ? 0 : g.target, a );
        if( g.target >= 0 ) snprintf( buf, sizeof buf, " NEXT: %s - show it at the box, KEY0", a );
        else                snprintf( buf, sizeof buf, " SCAN DONE - %u PAIR%s FOUND", g.pairs, g.pairs == 1 ? "" : "S" );
    }
    else if( grid_left( &g ) <= 1 )
        snprintf( buf, sizeof buf, " GAME OVER - A %u : B %u", g.score[ 0 ], g.score[ 1 ] );
    else if( g.target >= 0 ) {
        grid_cell_name( &g, g.target, a );
        snprintf( buf, sizeof buf, " TEAM %c: turn up %s, show it, KEY0", 'A' + g.team, a );
    }
    else
        snprintf( buf, sizeof buf, " TEAM %c: type the %s card's cell", 'A' + g.team,
                  g.n_open == 1 ? "2nd" : "1st" );
    text( INFO_ROW, INFO_COL, INFO_W, buf, GUI_BLACK, BRASS );
    fill( INFO_ROW + 1, INFO_COL, 8, INFO_W, ' ', INK, FELT );

    if( g.mode == GRID_SCAN ) {
        snprintf( buf, sizeof buf, "READ %u/%d    PAIRS %u", g.n_read, cells, g.pairs );
        text( row++, INFO_COL, INFO_W, buf, INK, FELT );
        /* the pairs, two columns of six: "3  7  A1+C3" */
        for( k = 1; k <= g.pairs && k <= 12; k++ ) {
            int first = -1;

            for( i = 0; i < cells; i++ ) {
                if( g.cell[ i ].pair != k ) continue;
                if( first < 0 ) { first = i; continue; }
                grid_cell_name( &g, first, a );
                grid_cell_name( &g, i, b );
                snprintf( buf, sizeof buf, "%2d %-2s %s+%s", k, short_rank( g.cell[ i ].result ), a, b );
                text( row + ( k - 1 ) % 6, INFO_COL + ( k - 1 ) / 6 * 20, 19, buf, GUI_BLACK, BRASS );
                break;
            }
        }
    }
    else {
        snprintf( buf, sizeof buf, "TURN %u    TEAM %c TO PLAY", g.turn, 'A' + g.team );
        text( row++, INFO_COL, INFO_W, buf, INK, FELT );
        snprintf( buf, sizeof buf, "SCORE   A %u : B %u", g.score[ 0 ], g.score[ 1 ] );
        text( row++, INFO_COL, INFO_W, buf, INK, FELT );
        snprintf( buf, sizeof buf, "LEFT    %d of %d cards", grid_left( &g ), cells );
        text( row++, INFO_COL, INFO_W, buf, INK, FELT );
        if( g.outcome != OUT_NONE && g.n_open == 2 ) {
            grid_cell_name( &g, g.open[ 0 ], a );
            grid_cell_name( &g, g.open[ 1 ], b );
            snprintf( buf, sizeof buf, " %s  %s+%s  (team %c)", g.outcome == OUT_MATCH ? "MATCH" : "NO MATCH",
                      a, b, 'A' + ( g.team ^ 1 ) );
            text( ++row, INFO_COL, 0, buf, GUI_BLACK, g.outcome == OUT_MATCH ? BRASS : GUI_WHITE );
        }
    }

    /* the latest read, wherever it went; blank after a new board or an undo */
    buf[ 0 ] = '\0';
    if( last_valid ) {
        char name[ 24 ];

        if( last.placed >= 0 ) grid_cell_name( &g, last.placed, a );
        snprintf( buf, sizeof buf, "last: %s %s %s", card_name( last.result, name, sizeof name ),
                  last.placed >= 0 ? "->" : "", last.placed >= 0 ? a : "(not placed)" );
    }
    text( ROWS - 2, INFO_COL, INFO_W, buf, INK, FELT );
}

static void redraw( void )
{
    draw_board();
    draw_info();
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

static void new_board( int n, int mode )
{
    char msg[ 24 ];

    grid_new( &g, n, mode );
    n_undo = 0;
    last_valid = 0;
    redraw();
    snprintf( msg, sizeof msg, "NEW %dx%d %s", g.n, g.n, mode == GRID_SCAN ? "SCAN" : "GAME" );
    gui_status( msg, GUI_LGREEN );
}

void gui_init( void )
{
    static const struct { const char * key, * what; } HELP[] = {
        { "KEY0", "read" }, { "KEY1", "undo/hold:new" }, { "SW0", "ascii" }, { "SW1", "pgm" },
#if RTOS_MODE
        { "SW2", "stats" }, { "SW3", "auto" }, { "UART", "h" },
#endif
    };
    unsigned k;
    int r, col;

    fill( 0, 0, ROWS, COLS, ' ', INK, FELT );

    /* the rail: bars top and bottom, both sides, the felt's corners rounded off */
    fill( 0, 0, 1, COLS, ' ', INK, WOOD );
    fill( ROWS - 1, 0, 1, COLS, ' ', INK, WOOD );
    for( r = 1; r < ROWS - 1; r++ ) {
        put( r, 0, ' ', INK, WOOD );
        put( r, COLS - 1, ' ', INK, WOOD );
    }
    put( HINT_ROW, 1, CH_LOWER, FELT, WOOD );           /* under the camera */
    put( ROWS - 2, 1, CH_UPPER, FELT, WOOD );

    col = text( 0, 2, 0, "\x06\x03 ECE4813  CARD TABLE \x05\x04", BRASS, WOOD );
    text( 0, col + 2, 0, "DE10-Nano CNN", INK, WOOD );

    /* the camera, set into the rail: wood with a brass pinstripe round the preview */
    frame( CAM_ROW, CAM_COL, CAM_H, CAM_W, BRASS, WOOD );
    text( CAM_ROW, CAM_COL + 3, 0, " LIVE CAMERA ", GUI_BLACK, BRASS );
    text( HINT_ROW, 2, CAM_W - 3, "fit the whole card in the green box", INK, FELT );

    col = 1;
    for( k = 0; k < sizeof HELP / sizeof HELP[ 0 ]; k++ ) {
        col = text( ROWS - 1, col, 0, HELP[ k ].key, BRASS, WOOD ) + 1;
        col = text( ROWS - 1, col, 0, HELP[ k ].what, INK, WOOD ) + 2;
    }

    new_board( 3, GRID_SCAN );                          /* round 1: face-up 3x3 */
    gui_status( "READY", GUI_LGREEN );
}

void gui_job_done( unsigned shot, const struct vision_result * v, int track, const char * where )
{
    struct grid before;
    char msg[ 28 ], cell[ 3 ], name[ 24 ];
    int ev;

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

    last.result = v->result;
    last.shot   = shot;
    last.where  = where;
    last.placed = -1;
    last_valid  = 1;
    if( !track ) {                          /* SW3 auto: shown, never placed */
        draw_info();
        gui_status( "READY (auto)", GUI_LGREEN );
        return;
    }

    before = g;
    ev = grid_read( &g, v->result, shot );
    if( ev == GRID_NO_CELL || ev == GRID_BAD_CELL ) {
        draw_info();
        gui_status( ev == GRID_BAD_CELL ? "CELL NOT FACE DOWN" :
                    g.mode == GRID_SCAN ? "GRID FULL - type a cell" : "TYPE THE CELL FIRST", GUI_YELLOW );
        return;
    }
    undo_ring[ undo_head ] = before;
    undo_head = ( undo_head + 1 ) % UNDO_MAX;
    if( n_undo < UNDO_MAX ) n_undo++;
    last.placed = g.last;

    redraw();
    grid_cell_name( &g, g.last, cell );
    if( ev == GRID_JUDGED ) gui_status( g.outcome == OUT_MATCH ? "MATCH!" : "NO MATCH", g.outcome == OUT_MATCH ? GUI_LGREEN : GUI_YELLOW );
    else if( ev == GRID_PAIRED ) {
        snprintf( msg, sizeof msg, "PAIR: %s + %s", short_rank( v->result ), cell );
        gui_status( msg, GUI_LGREEN );
    }
    else {
        snprintf( msg, sizeof msg, "%s: %s", cell, card_name( v->result, name, sizeof name ) );
        gui_status( msg, GUI_LGREEN );
    }
}

void gui_undo( void )
{
    char msg[ 24 ], cell[ 3 ];

    if( !n_undo ) {
        gui_status( "NOTHING TO UNDO", GUI_YELLOW );
        return;
    }
    grid_cell_name( &g, g.last, cell );
    undo_head = ( undo_head + UNDO_MAX - 1 ) % UNDO_MAX;
    g = undo_ring[ undo_head ];
    n_undo--;
    last_valid = 0;
    redraw();
    snprintf( msg, sizeof msg, "UNDID %s", cell );
    gui_status( msg, GUI_YELLOW );
}

void gui_clear( void )
{
    new_board( g.n, g.mode );
}

int gui_command( const char * cmd )
{
    char c0 = cmd[ 0 ] | 0x20, msg[ 24 ], name[ 3 ];
    int  cell;

    if( cmd[ 0 ] >= '3' && cmd[ 0 ] <= '5' && !cmd[ 1 ] ) {         /* "4": face-up scan */
        new_board( cmd[ 0 ] - '0', GRID_SCAN );
        return 1;
    }
    if( c0 == 'g' && cmd[ 1 ] >= '3' && cmd[ 1 ] <= '5' && !cmd[ 2 ] ) {   /* "g4": face-down game */
        new_board( cmd[ 1 ] - '0', GRID_GAME );
        return 1;
    }
    if( c0 == 'r' && !cmd[ 1 ] ) {                                  /* restart this board */
        new_board( g.n, g.mode );
        return 1;
    }
    cell = grid_parse_cell( &g, cmd );
    if( cell < 0 ) return 0;
    grid_cell_name( &g, cell, name );
    if( !grid_target( &g, cell ) ) {
        snprintf( msg, sizeof msg, "%s IS NOT FACE DOWN", name );
        gui_status( msg, GUI_YELLOW );
        return 1;
    }
    redraw();
    snprintf( msg, sizeof msg, "NEXT: %s", name );
    gui_status( msg, GUI_LGREEN );
    return 1;
}
