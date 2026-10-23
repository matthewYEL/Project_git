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
 *                          -- or, in the matrix practical, the values read as
 *                          numbers in chips (draw_matrix)
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
#include "sortgame.h"

#ifdef GUI_HOST
void alt_write_word( uint32_t addr, uint32_t value );   /* gui_host_test.c emulates the PIOs */
#else
#include "socal.h"
#endif

/* The bus is shared by two CPU0 tasks since the grid scan: Report draws text,
 * Vision writes conv1's window registers. Each write is three PIO accesses
 * that must not interleave, so hold off task switches across them (the
 * scheduler lock, not a critical section: no ISR touches the bus, and it is
 * safe before the scheduler starts, which gui_init runs). */
#if RTOS_MODE && !defined( GUI_HOST )
#include "FreeRTOS.h"
#include "task.h"
#define BUS_LOCK()      vTaskSuspendAll()
#define BUS_UNLOCK()    ( void ) xTaskResumeAll()
#else
#define BUS_LOCK()
#define BUS_UNLOCK()
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

/* The board came from an automatic grid scan: cells are named by the
 * professor's (row, col) -- "(2,1)" -- instead of the guided scan's "A2" --
 * and the title gives the rows x columns found (a 4x3 is drawn on a 4x4). */
static int s_auto;
static int s_rows, s_cols;

/* The board is the matrix practical's (gui_matrix): a guided face-up board
 * drawn as numbers, its cells named "R2C3" (draw_matrix). */
static int s_matrix;

/* The cell the matrix shows at row r, column c (from 0). The k-th value read
 * goes into cell k, so this one line sets the order on screen: r * g.n + c row
 * by row (the sheet), c * g.n + r column by column. The "R2C3" names and the
 * brass "next" chip follow it. */
static int mx_cell( int r, int c )
{
    return r * g.n + c;
}

/* KEY1 after a whole-grid scan: the board as it was before the latest scan (or
 * clear), one level -- a tap brings that result back, e.g. the 3x3 after a bad
 * 4x4 scan (30 Sep). The undo ring above is for guided one-card reads. */
static struct { struct grid g; struct det last; int last_valid, rows, cols; } scan_prev;
static int scan_prev_ok;

static void scan_keep( void )
{
    if( !s_auto || ( g.mode == GRID_SCAN && !g.n_read ) ) return;   /* nothing scanned to keep */
    scan_prev.g          = g;
    scan_prev.last       = last;
    scan_prev.last_valid = last_valid;
    scan_prev.rows       = s_rows;
    scan_prev.cols       = s_cols;
    scan_prev_ok         = 1;
}

static void cell_label( int i, char * buf, unsigned n )
{
    int r, c;

    if( s_auto ) snprintf( buf, n, "(%d,%d)", i / g.n + 1, i % g.n + 1 );
    else if( s_matrix ) {                       /* where draw_matrix shows cell i */
        buf[ 0 ] = '\0';
        for( r = 0; r < g.n; r++ )
            for( c = 0; c < g.n; c++ )
                if( mx_cell( r, c ) == i ) snprintf( buf, n, "R%dC%d", r + 1, c + 1 );
    }
    else {
        char name[ 3 ];
        grid_cell_name( &g, i, name );
        snprintf( buf, n, "%s", name );
    }
}

/* ---- drawing primitives ------------------------------------------------- */

/* one word on the text-layer bus: data, then the address with the enable, then
 * without it */
static void bus_write( unsigned a, unsigned data )
{
    BUS_LOCK();
    alt_write_word( TXT_DATA_PIO_BASE, data & 0xFFFFu );
    alt_write_word( TXT_ADDR_PIO_BASE, TXT_WE | a );
    alt_write_word( TXT_ADDR_PIO_BASE, a );
    BUS_UNLOCK();
}

static void put( int row, int col, unsigned ch, unsigned fg, unsigned bg )
{
    if( row < 0 || row >= ROWS || col < 0 || col >= COLS ) return;
    bus_write( ( unsigned ) ( row * COLS + col ), ( bg << 12 ) | ( fg << 8 ) | ( ch & 0xFFu ) );
}

/* a cell with no lock: only for gui_fatal, when nothing else runs any more */
static void put_raw( int row, int col, unsigned ch, unsigned fg, unsigned bg )
{
    unsigned a = ( unsigned ) ( row * COLS + col );

    alt_write_word( TXT_DATA_PIO_BASE, ( ( bg << 12 ) | ( fg << 8 ) | ( ch & 0xFFu ) ) & 0xFFFFu );
    alt_write_word( TXT_ADDR_PIO_BASE, TXT_WE | a );
    alt_write_word( TXT_ADDR_PIO_BASE, a );
}

void gui_fatal( const char * msg )
{
    static const char PILL[] = " CPU0 CRASHED ";
    int c;

    for( c = STATUS_COL; c < COLS - 1; c++ ) put_raw( 0, c, ' ', INK, WOOD );   /* the old pill gone */
    for( c = 0; PILL[ c ]; c++ ) put_raw( 0, COLS - 2 - ( int ) sizeof PILL + 1 + c, ( unsigned char ) PILL[ c ], GUI_WHITE, GUI_RED );
    for( c = 0; c < INFO_W; c++ ) put_raw( INFO_ROW, INFO_COL + c, *msg ? ( unsigned char ) *msg++ : ' ', GUI_WHITE, GUI_RED );
}

void win_write( const struct win_regs * w )
{
    const unsigned v[ 8 ] = {
        ( unsigned ) w->x0 & 0x7FFu, ( unsigned ) w->y0 & 0x7FFu, w->step, w->flags,
        w->clip[ 0 ], w->clip[ 1 ], w->clip[ 2 ], w->clip[ 3 ]
    };
    unsigned k;

    for( k = 0; k < 8; k++ ) bus_write( WIN_REG_BASE + k, v[ k ] );
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

static void short_card( uint32_t r, char * buf, unsigned n )    /* "7H", "10S", "JK" */
{
    static const char SUIT_CH[ 4 ] = { 'S', 'C', 'H', 'D' };

    if( RES_JOKER( r ) ) snprintf( buf, n, "JK" );
    else                 snprintf( buf, n, "%s%c", RANK_NAMES[ RES_RANK( r ) ], SUIT_CH[ RES_SUIT( r ) ] );
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
    char name[ 8 ];

    if( s_auto && ( i / g.n >= s_rows || i % g.n >= s_cols ) ) return;   /* not in the scanned grid */
    cell_pos( i, &row, &col );
    cell_label( i, name, sizeof name );
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
    if( s_auto && g.mode == GRID_GAME ) snprintf( buf, sizeof buf, " FACE-DOWN GAME %d x %d ", s_rows, s_cols );
    else if( s_auto ) snprintf( buf, sizeof buf, " GRID SCAN %d x %d ", s_rows, s_cols );
    else         snprintf( buf, sizeof buf, " %s %dx%d ", g.mode == GRID_SCAN ? "FACE-UP" : "FACE-DOWN GAME", g.n, g.n );
    text( GRID_ROW, GRID_COL + 1, 0, buf, GUI_BLACK, BRASS );
    for( i = 0; i < g.n; i++ ) {
        cell_pos( i, &row, &col );                      /* top row: the column letters */
        if( !s_auto || i < s_cols )
            put( row - 1, col + GEO[ g.n - 3 ].w / 2, ( unsigned ) ( ( s_auto ? '1' : 'A' ) + i ), INK, FELT );
        cell_pos( i * g.n, &row, &col );                /* left column: the row numbers */
        if( !s_auto || i < s_rows )
            put( row + ( GEO[ g.n - 3 ].h - 1 ) / 2, col - 2, ( unsigned ) ( '1' + i ), INK, FELT );
    }
    for( i = 0; i < g.n * g.n; i++ ) draw_cell( i );
}

/* ---- the matrix practical ------------------------------------------------
 * The values in white chips, C1.. along the top and R1.. down the left, the
 * next cell in brass, empty ones dark grey. Chip (r, c) shows cell
 * mx_cell( r, c ), which sets the order. Chips at r or c = n -- a sums row /
 * column, say -- still fit for n <= 5. */
#define MX_ROW      3               /* the C1.. labels; chip row r starts at MX_ROW + 1 + r * MX_PY */
#define MX_COL      47              /* column 1's chip; the R1.. labels three to its left */
#define MX_PX       5               /* chip pitch: 4 wide + 1 gap */
#define MX_PY       4               /* 3 tall + 1 gap */

static void mx_pos( int r, int c, int * row, int * col )
{
    *row = MX_ROW + 1 + r * MX_PY;
    *col = MX_COL + c * MX_PX;
}

/* a 4x3 chip at (r, c) with `v` in its middle row */
static void mx_chip( int r, int c, const char * v, unsigned fg, unsigned bg )
{
    int row, col;

    mx_pos( r, c, &row, &col );
    fill( row, col, 3, 4, ' ', fg, bg );
    text( row + 1, col, 0, v, fg, bg );
}

static void draw_matrix( void )
{
    char buf[ 24 ];
    int  r, c, row, col;

    fill( GRID_ROW, GRID_COL, ROWS - 2, COLS - 1 - GRID_COL, ' ', INK, FELT );
    put( 1, COLS - 2, CH_LOWER, FELT, WOOD );           /* the felt's rounded corners */
    put( ROWS - 2, COLS - 2, CH_UPPER, FELT, WOOD );
    snprintf( buf, sizeof buf, " MATRIX %dx%d ", g.n, g.n );
    text( GRID_ROW, GRID_COL + 1, 0, buf, GUI_BLACK, BRASS );
    for( r = 0; r < g.n; r++ ) {
        mx_pos( 0, r, &row, &col );                     /* column r's label on top */
        snprintf( buf, sizeof buf, "C%d", r + 1 );
        text( row - 1, col + 1, 0, buf, INK, FELT );
        mx_pos( r, 0, &row, &col );                     /* row r's on the left */
        snprintf( buf, sizeof buf, "R%d", r + 1 );
        text( row + 1, col - 3, 0, buf, INK, FELT );
        for( c = 0; c < g.n; c++ ) {
            int i = mx_cell( r, c );                    /* the order: see mx_cell */
            const struct grid_cell * k = &g.cell[ i ];

            if( k->state != CELL_OPEN ) {
                mx_chip( r, c, "", GUI_BLACK, i == g.target ? BRASS : GUI_DGREY );
                continue;
            }
            snprintf( buf, sizeof buf, "%3d", grid_value( k->result ) );
            mx_chip( r, c, buf, GUI_BLACK, GUI_WHITE );
        }
    }
}

/* ---- INFO: the prompt and the round's progress -------------------------- */

static void draw_info( void )
{
    char buf[ 48 ], a[ 8 ], b[ 8 ], n1[ 8 ], n2[ 8 ];
    int  i, k, row = INFO_ROW + 2, cells = g.n * g.n;

    if( s_matrix ) {                                    /* the matrix practical */
        cell_label( g.target < 0 ? 0 : g.target, a, sizeof a );
        if( g.target >= 0 ) snprintf( buf, sizeof buf, " NEXT: %s - show the card, KEY0", a );
        else                snprintf( buf, sizeof buf, " MATRIX FULL - hold KEY1 for a new one" );
        text( INFO_ROW, INFO_COL, INFO_W, buf, GUI_BLACK, BRASS );
        fill( INFO_ROW + 1, INFO_COL, 8, INFO_W, ' ', INK, FELT );
        snprintf( buf, sizeof buf, "CAPTURED %u of %d", g.n_read, cells );
        text( row, INFO_COL, INFO_W, buf, INK, FELT );
        buf[ 0 ] = '\0';
        if( last_valid ) {                              /* the latest read and its value */
            char name[ 24 ];

            if( last.placed >= 0 ) cell_label( last.placed, a, sizeof a );
            snprintf( buf, sizeof buf, "last: %s = %d %s %s", card_name( last.result, name, sizeof name ),
                      grid_value( last.result ), last.placed >= 0 ? "->" : "", last.placed >= 0 ? a : "(not placed)" );
        }
        text( ROWS - 2, INFO_COL, INFO_W, buf, INK, FELT );
        return;
    }

    /* the prompt: what the operator does next */
    if( g.mode == GRID_SCAN && s_auto ) {
        snprintf( buf, sizeof buf, " GRID SCAN - %u CARD%s, %u PAIR%s", g.n_read, g.n_read == 1 ? "" : "S",
                  g.pairs, g.pairs == 1 ? "" : "S" );
    }
    else if( g.mode == GRID_SCAN ) {
        cell_label( g.target < 0 ? 0 : g.target, a, sizeof a );
        if( g.target >= 0 ) snprintf( buf, sizeof buf, " NEXT: %s - show it at the box, KEY0", a );
        else                snprintf( buf, sizeof buf, " SCAN DONE - %u PAIR%s FOUND", g.pairs, g.pairs == 1 ? "" : "S" );
    }
    else if( grid_left( &g ) <= 1 && g.players == 1 )
        snprintf( buf, sizeof buf, " GAME OVER - %u PAIR%s IN %u TURNS", g.pairs, g.pairs == 1 ? "" : "S", g.turn - 1u );
    else if( grid_left( &g ) <= 1 )
        snprintf( buf, sizeof buf, " GAME OVER - A %u : B %u", g.score[ 0 ], g.score[ 1 ] );
    else if( g.players == 1 )
        snprintf( buf, sizeof buf, " TURN %u: turn 2 cards up, KEY0", g.turn );
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
        snprintf( buf, sizeof buf, "READ %u/%d    PAIRS %u", g.n_read, s_auto ? s_rows * s_cols : cells, g.pairs );
        text( row++, INFO_COL, INFO_W, buf, INK, FELT );
        /* the pairs, two columns of six: "3  7  A1+C3" */
        for( k = 1; k <= g.pairs && k <= 12; k++ ) {
            int first = -1;

            for( i = 0; i < cells; i++ ) {
                if( g.cell[ i ].pair != k ) continue;
                if( first < 0 ) { first = i; continue; }
                cell_label( first, a, sizeof a );
                cell_label( i, b, sizeof b );
                snprintf( buf, sizeof buf, "%2d %-2s %s+%s", k, short_rank( g.cell[ i ].result ), a, b );
                text( row + ( k - 1 ) % 6, INFO_COL + ( k - 1 ) / 6 * 20, 19, buf, GUI_BLACK, BRASS );
                break;
            }
        }
    }
    else if( g.players == 1 ) {
        snprintf( buf, sizeof buf, "TURN %u    PAIRS %u    LEFT %d of %d", g.turn, g.pairs, grid_left( &g ), cells );
        text( row++, INFO_COL, INFO_W, buf, INK, FELT );
        /* the pairs found, in the order found, two columns of six: "7H(1,1)+7D(3,2)" */
        for( k = 1; k <= g.pairs && k <= 12; k++ ) {
            int first = -1;

            for( i = 0; i < cells; i++ ) {
                if( g.cell[ i ].state != CELL_MATCHED || g.cell[ i ].pair != k ) continue;
                if( first < 0 ) { first = i; continue; }
                cell_label( first, a, sizeof a );
                cell_label( i, b, sizeof b );
                short_card( g.cell[ first ].result, n1, sizeof n1 );
                short_card( g.cell[ i ].result, n2, sizeof n2 );
                snprintf( buf, sizeof buf, "%s%s+%s%s", n1, a, n2, b );
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

    if( g.mode == GRID_GAME && g.players == 1 ) {     /* the camera game: its latest turn instead */
        text( ROWS - 2, INFO_COL, INFO_W, "", INK, FELT );
        if( g.outcome != OUT_NONE && g.n_open == 2 ) {
            cell_label( g.open[ 0 ], a, sizeof a );
            cell_label( g.open[ 1 ], b, sizeof b );
            snprintf( buf, sizeof buf, g.outcome == OUT_MATCH ? " PAIR  %s+%s" : " NO PAIR  %s+%s - turn them back",
                      a, b );
            text( ROWS - 2, INFO_COL, 0, buf, GUI_BLACK, g.outcome == OUT_MATCH ? BRASS : GUI_WHITE );
        }
        return;
    }

    /* the latest read, wherever it went; blank after a new board or an undo */
    buf[ 0 ] = '\0';
    if( last_valid ) {
        char name[ 24 ];

        if( last.placed >= 0 ) cell_label( last.placed, a, sizeof a );
        snprintf( buf, sizeof buf, "last: %s %s %s", card_name( last.result, name, sizeof name ),
                  last.placed >= 0 ? "->" : "", last.placed >= 0 ? a : "(not placed)" );
    }
    text( ROWS - 2, INFO_COL, INFO_W, buf, INK, FELT );
}

static void redraw( void )
{
    if( s_matrix ) draw_matrix();
    else           draw_board();
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
    snprintf( msg, sizeof msg, "NEW %dx%d %s", g.n, g.n, s_matrix ? "MATRIX" : mode == GRID_SCAN ? "SCAN" : "GAME" );
    gui_status( msg, GUI_LGREEN );
}

void gui_init( void )
{
    static const struct { const char * key, * what; } HELP[] = {
#if SORT_GAME && RTOS_MODE
        { "KEY0", "read held card" }, { "KEY1", "undo / ignore" }, { "hold KEY1", "new game" }, { "SW1", "save picture" },
#elif RTOS_MODE
        { "KEY0", "read" }, { "KEY1", "undo/hold:new" },
        { "SW0", "turn" }, { "SW1", "save" }, { "SW2", "game" }, { "SW3", "auto" }, { "UART", "h" },
#else
        { "KEY0", "read" }, { "KEY1", "undo/hold:new" },
        { "SW0", "ascii" }, { "SW1", "pgm" },
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

#if SORT_GAME && RTOS_MODE
    col = text( 0, 2, 0, "\x06\x03 RAMA'S SUIT SORT \x05\x04", BRASS, WOOD );
#else
    col = text( 0, 2, 0, "\x06\x03 RAMA'S GAME OF POKER \x05\x04", BRASS, WOOD );
#endif
    text( 0, col + 2, 0, "DE10-Nano CNN", INK, WOOD );

    /* the camera, set into the rail: wood with a brass pinstripe round the preview */
    frame( CAM_ROW, CAM_COL, CAM_H, CAM_W, BRASS, WOOD );
    text( CAM_ROW, CAM_COL + 3, 0, " LIVE CAMERA ", GUI_BLACK, BRASS );
#if SORT_GAME && RTOS_MODE
    text( HINT_ROW, 2, CAM_W - 3, "starting the camera...", INK, FELT );
#else
    text( HINT_ROW, 2, CAM_W - 3, "fit the whole card in the green box", INK, FELT );
#endif

    col = 1;
    for( k = 0; k < sizeof HELP / sizeof HELP[ 0 ]; k++ ) {
        col = text( ROWS - 1, col, 0, HELP[ k ].key, BRASS, WOOD ) + 1;
        col = text( ROWS - 1, col, 0, HELP[ k ].what, INK, WOOD ) + 2;
    }

#if SORT_GAME && RTOS_MODE
    text( CAM_ROW, CAM_COL + 18, 0, " KEY0: READ CARD ", GUI_BLACK, BRASS );
#else
    new_board( 3, GRID_SCAN );                          /* round 1: face-up 3x3 */
#endif
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
    if( s_matrix && grid_value( v->result ) > 9 ) {     /* the practical's digits are 0..9 */
        draw_info();
        gui_status( "NOT A DIGIT (A-9, JOKER)", GUI_YELLOW );
        return;
    }

    before = g;
    ev = grid_read( &g, v->result, shot );
    if( ev == GRID_NO_CELL || ev == GRID_BAD_CELL ) {
        draw_info();
        gui_status( ev == GRID_BAD_CELL ? "CELL NOT FACE DOWN" : s_matrix ? "MATRIX FULL - HOLD KEY1" :
                    g.mode == GRID_SCAN ? "GRID FULL - type a cell" : "TYPE THE CELL FIRST", GUI_YELLOW );
        return;
    }
    undo_ring[ undo_head ] = before;
    undo_head = ( undo_head + 1 ) % UNDO_MAX;
    if( n_undo < UNDO_MAX ) n_undo++;
    last.placed = g.last;

    redraw();
    if( s_matrix ) {                        /* "R2C3 = 7 (7 of Hearts)" */
        char lbl[ 8 ];

        cell_label( g.last, lbl, sizeof lbl );
        snprintf( msg, sizeof msg, "%s = %d (%s)", lbl, grid_value( v->result ), card_name( v->result, name, sizeof name ) );
        gui_status( msg, GUI_LGREEN );
        return;
    }
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
    char msg[ 24 ], cell[ 8 ];

    if( s_auto ) {                              /* a scanned board: back to the one before */
        if( !scan_prev_ok ) {
            gui_status( "NOTHING TO UNDO", GUI_YELLOW );
            return;
        }
        g            = scan_prev.g;
        last         = scan_prev.last;
        last_valid   = scan_prev.last_valid;
        s_rows       = scan_prev.rows;
        s_cols       = scan_prev.cols;
        scan_prev_ok = 0;
        redraw();
        gui_status( g.mode == GRID_GAME ? "UNDID LAST TURN" : "UNDID LAST SCAN", GUI_YELLOW );
        return;
    }
    if( !n_undo ) {
        gui_status( "NOTHING TO UNDO", GUI_YELLOW );
        return;
    }
    cell_label( g.last, cell, sizeof cell );
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
    int players = g.players;

    scan_keep();                                /* a tap brings a cleared scan back */
    new_board( g.n, g.mode );
    if( g.mode == GRID_GAME && players == 1 ) { /* the camera game stays one-player */
        g.players = 1;
        redraw();
    }
}

int gui_command( const char * cmd )
{
    char c0 = cmd[ 0 ] | 0x20, msg[ 24 ], name[ 3 ];
    int  cell;

    if( cmd[ 0 ] >= '3' && cmd[ 0 ] <= '5' && !cmd[ 1 ] ) {         /* "4": face-up scan */
        s_auto = 0;
        new_board( cmd[ 0 ] - '0', GRID_SCAN );
        return 1;
    }
    if( c0 == 'g' && cmd[ 1 ] >= '3' && cmd[ 1 ] <= '5' && !cmd[ 2 ] ) {   /* "g4": face-down game */
        s_auto = 0;
        s_matrix = 0;
        new_board( cmd[ 1 ] - '0', GRID_GAME );
        return 1;
    }
    if( c0 == 'r' && !cmd[ 1 ] ) {                                  /* restart this board */
        s_auto = 0;
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

/* ---- whole-grid scan ---------------------------------------------------- */

void gui_scan_begin( const struct scan_result * s )
{
    char msg[ 28 ];
    int  n = ( int ) ( s->nrows > s->ncols ? s->nrows : s->ncols );

    scan_keep();                                /* KEY1 can bring the last result back */
    s_auto = 1;
    s_matrix = 0;
    s_rows = ( int ) s->nrows;
    s_cols = ( int ) s->ncols;
    new_board( n < 3 ? 3 : n > GRID_MAX ? GRID_MAX : n, GRID_SCAN );
    g.target = -1;                          /* nothing to prompt: the scan places cards */
    draw_info();
    snprintf( msg, sizeof msg, "SCANNING %u CARD%s", ( unsigned ) s->n, s->n == 1 ? "" : "S" );
    gui_status( msg, GUI_YELLOW );
}

void gui_scan_card( unsigned row, unsigned col, uint32_t result, unsigned shot )
{
    char msg[ 28 ], name[ 24 ];
    int  cell;

    if( !row || !col || row > g.n || col > g.n ) return;    /* off the drawable board */
    cell = ( int ) ( ( row - 1 ) * g.n + ( col - 1 ) );
    if( !RES_DONE( result ) || RES_DDR_ERR( result ) ||
        ( !RES_JOKER( result ) && RES_RANK( result ) >= 13u ) ) {
        snprintf( msg, sizeof msg, "NO READ AT (%u,%u)", row, col );
        gui_status( msg, GUI_LRED );
        return;
    }
    grid_target( &g, cell );                /* a filled cell is read again: a re-read replaces it */
    grid_read( &g, result, shot );
    g.target = -1;
    last.result = result;
    last.shot   = shot;
    last.where  = NULL;
    last.placed = cell;
    last_valid  = 1;
    redraw();
    snprintf( msg, sizeof msg, "(%u,%u) %s", row, col, card_name( result, name, sizeof name ) );
    gui_status( msg, GUI_YELLOW );
}

void gui_scan_done( void )
{
    char msg[ 28 ];

    g.target = -1;
    draw_info();
    snprintf( msg, sizeof msg, "SCAN DONE - %u PAIR%s", g.pairs, g.pairs == 1 ? "" : "S" );
    gui_status( msg, g.pairs >= 2 ? GUI_LGREEN : GUI_YELLOW );
}

/* The scan frame is the green box plus 64 camera px (32 preview px) each side,
 * top and bottom the same as the box. In AUTO mode a thin line just inside
 * each side edge (text columns 5 and 36 -- glyph pixels print over the video)
 * shows where the whole grid must fit; a card cut by the frame edge is not
 * counted. Off again in guided mode, where the preview must stay clear. */
#define GUIDE_COL_L     5
#define GUIDE_COL_R     36
#define GUIDE_ROW_0     4
#define GUIDE_ROW_1     14
#define CH_VLINE        0xB3u

void gui_mode( int auto_scan, const char * orient_name )
{
    char buf[ 48 ];
    int  r;

    if( auto_scan ) snprintf( buf, sizeof buf, "AUTO: grid inside the lines (%s)", orient_name );
    else            snprintf( buf, sizeof buf, "fit the whole card in the green box" );
    text( HINT_ROW, 2, CAM_W - 3, buf, INK, FELT );
    for( r = GUIDE_ROW_0; r <= GUIDE_ROW_1; r++ ) {
        put( r, GUIDE_COL_L, auto_scan ? CH_VLINE : ' ', GUI_LGREEN, FELT );
        put( r, GUIDE_COL_R, auto_scan ? CH_VLINE : ' ', GUI_LGREEN, FELT );
    }
}

/* ---- face-down game, camera-driven (SW2) ----------------------------------
 * The board comes from a scan with every card face down; each turn the camera
 * finds the two cards turned up and gui_game_turn() judges them with grid.c's
 * rules. One player: pairs and turns are counted, no teams. Found pairs stay
 * face up where they lie, and the camera skips their cells (gui_game_matched). */
void gui_game_begin( int n )
{
    char msg[ 24 ];

    scan_keep();                                /* KEY1 can bring the last scan back */
    s_auto = 1;
    s_matrix = 0;
    s_rows = s_cols = n;
    new_board( n, GRID_GAME );
    g.players = 1;
    redraw();
    snprintf( msg, sizeof msg, "GAME %dx%d READY", g.n, g.n );
    gui_status( msg, GUI_LGREEN );
}

int gui_game_turn( unsigned r1, unsigned c1, uint32_t w1, unsigned r2, unsigned c2, uint32_t w2, unsigned shot )
{
    char        msg[ 40 ], a[ 8 ], b[ 8 ], n1[ 8 ], n2[ 8 ];
    int         k1, k2;
    struct grid before = g;

    if( g.mode != GRID_GAME || !s_auto || !r1 || !c1 || !r2 || !c2 ||
        r1 > g.n || c1 > g.n || r2 > g.n || c2 > g.n ) return OUT_NONE;
    k1 = ( int ) ( ( r1 - 1 ) * g.n + ( c1 - 1 ) );
    k2 = ( int ) ( ( r2 - 1 ) * g.n + ( c2 - 1 ) );
    if( !grid_target( &g, k1 ) || grid_read( &g, w1, shot ) != GRID_OPENED ||
        !grid_target( &g, k2 ) || grid_read( &g, w2, shot ) != GRID_JUDGED ) {
        g = before;                             /* a found pair's cell, or one cell twice: nothing happened */
        gui_status( "NOT FACE-DOWN CELLS", GUI_LRED );
        return OUT_NONE;
    }
    scan_prev.g          = before;              /* KEY1 tap undoes this turn */
    scan_prev.last       = last;
    scan_prev.last_valid = last_valid;
    scan_prev.rows       = s_rows;
    scan_prev.cols       = s_cols;
    scan_prev_ok         = 1;
    last.result = w2;
    last.shot   = shot;
    last.where  = NULL;
    last.placed = k2;
    last_valid  = 1;
    redraw();
    cell_label( k1, a, sizeof a );
    cell_label( k2, b, sizeof b );
    short_card( w1, n1, sizeof n1 );
    short_card( w2, n2, sizeof n2 );
    snprintf( msg, sizeof msg, "%s %s%s %s%s", g.outcome == OUT_MATCH ? "PAIR!" : "NO PAIR", n1, a, n2, b );
    gui_status( msg, g.outcome == OUT_MATCH ? GUI_LGREEN : GUI_YELLOW );
    return g.outcome;
}

void gui_game_note( const char * msg )
{
    gui_status( msg, GUI_YELLOW );
}

int gui_game_n( void )
{
    return s_auto && g.mode == GRID_GAME ? g.n : 0;
}

uint32_t gui_game_matched( void )
{
    uint32_t m = 0;
    int      i;

    if( !gui_game_n() ) return 0;
    for( i = 0; i < g.n * g.n; i++ )
        if( g.cell[ i ].state == CELL_MATCHED ) m |= 1u << ( ( i / g.n ) * GRID_MAX + i % g.n );
    return m;
}

/* ---- the matrix practical ------------------------------------------------ */

void gui_matrix( int n )
{
    s_auto   = 0;
    s_matrix = 1;
    new_board( n, GRID_SCAN );
}

/* ---- M3: suit sort (sortgame.c, app_config.h SORT_GAME) ---------------------
 * The live preview carries the board's lines -- a column per suit, the rows,
 * the frame's edge -- as glyphs over the video, in exactly the text cells
 * sortgame.h's bands are cut to (frame px (x, y) is cell 5 + x/16,
 * (56 + y/2)/16). They are light cyan: the fabric's own green square is the
 * KEY0 read window. Every card the camera finds gets white corners, find by
 * find (brass: held up; grey: ignored, no card of ours); the slot the card to
 * place goes to gets brass edges and its name; whatever must come off gets
 * red corners. The panel on the right is the board as the game holds it. */

#define PV_ROW0     2               /* the preview's cells: rows 2..16 x cols 1..40 */
#define PV_COL0     1
#define PV_ROWS     15
#define PV_COLS     40
#define CH_CROSS    0xC5            /* box-drawing cross, T pieces */
#define CH_T_DOWN   0xC2
#define CH_T_UP     0xC1
#define CH_T_RIGHT  0xC3
#define CH_DBL_EXCL 0x13            /* CP437 double exclamation mark */
#define CH_ARROW_UP 0x18
#define CH_ARROW_DN 0x19
#define CH_CHECK    0xFB            /* CP437 square root: a tick */
#define LINE        GUI_LCYAN       /* the board's lines over the preview */
#define SORT_TOP    4               /* the board panel: row 1's cards, pitch GEO[ 1 ].py */
#define SORT_LEFT   47              /* ... column 1's, pitch GEO[ 1 ].px */
#define LIVE_ROW    ( INFO_ROW + 1 )    /* what the camera sees, every find */
#define SORT_EDGE_X ( FRAME_W - 8 ) /* the frame's right edge as drawn (text col 36) */

static uint16_t ov[ PV_ROWS ][ PV_COLS ];   /* the preview's overlay, built whole, then written */

static void ov_put( int row, int col, unsigned ch, unsigned fg )
{
    if( row < PV_ROW0 || row >= PV_ROW0 + PV_ROWS || col < PV_COL0 || col >= PV_COL0 + PV_COLS ) return;
    ov[ row - PV_ROW0 ][ col - PV_COL0 ] = ( uint16_t ) ( ( FELT << 12 ) | ( ( fg & 0xFu ) << 8 ) | ( ch & 0xFFu ) );
}

static unsigned ov_get( int row, int col )
{
    return ov[ row - PV_ROW0 ][ col - PV_COL0 ] & 0xFFu;
}

static int ov_col( int fx ) { return 5 + fx / 16; }
static int ov_row( int fy ) { return ( 56 + fy / 2 ) / 16; }

static void ov_text( int row, int col, const char * s, unsigned fg )
{
    while( *s ) ov_put( row, col++, ( unsigned char ) *s++, fg );
}

/* corners round frame box b, and `mark` in its middle */
static void ov_corners( const struct scan_card * b, unsigned fg, const char * mark )
{
    int c0 = ov_col( b->x0 ), c1 = ov_col( b->x1 ), r0 = ov_row( b->y0 ), r1 = ov_row( b->y1 );

    ov_put( r0, c0, BX_TL, fg ); ov_put( r0, c1, BX_TR, fg );
    ov_put( r1, c0, BX_BL, fg ); ov_put( r1, c1, BX_BR, fg );
    ov_text( ( r0 + r1 ) / 2, ( c0 + c1 + 1 ) / 2 - ( int ) strlen( mark ) / 2, mark, fg );
}

static void draw_sort_preview( const struct sort_game * sg )
{
    int      r, c, k, r1, c1;
    unsigned i;
    char     name[ 8 ];

    for( r = 0; r < PV_ROWS; r++ )
        for( c = 0; c < PV_COLS; c++ ) ov[ r ][ c ] = ( uint16_t ) ( ( FELT << 12 ) | ( INK << 8 ) | ' ' );

    /* the lines: between the rows, between the suit columns, the frame's edge
     * (locate drops a card touching it, so cards stay inside) */
    r = ov_row( 0 ); r1 = ov_row( FRAME_H );
    c = ov_col( SORT_X0 ); c1 = ov_col( SORT_EDGE_X );
    for( k = 1; k < SORT_ROWS; k++ )
        for( i = ( unsigned ) c + 1; i < ( unsigned ) c1; i++ ) ov_put( ov_row( k * SORT_BAND_H ), ( int ) i, BX_H, LINE );
    for( k = 1; k < SORT_COLS; k++ ) {
        int col = ov_col( SORT_X0 + k * SORT_BAND_W ), row;

        for( row = r + 1; row < r1; row++ ) ov_put( row, col, ov_get( row, col ) == BX_H ? CH_CROSS : BX_V, LINE );
    }
    for( k = c; k <= c1; k++ ) {
        ov_put( r, k, ov_get( r + 1, k ) == BX_V ? CH_T_DOWN : BX_H, LINE );
        ov_put( r1, k, ov_get( r1 - 1, k ) == BX_V ? CH_T_UP : BX_H, LINE );
    }
    for( k = r + 1; k < r1; k++ ) {
        ov_put( k, c, ov_get( k, c + 1 ) == BX_H ? CH_T_RIGHT : BX_V, LINE );
        ov_put( k, c1, BX_V, LINE );
    }
    ov_put( r, c, BX_TL, LINE ); ov_put( r, c1, BX_TR, LINE );
    ov_put( r1, c, BX_BL, LINE ); ov_put( r1, c1, BX_BR, LINE );
    for( k = 0; k < SORT_COLS; k++ ) {                  /* over the frame: the suits */
        unsigned suit = ( unsigned ) sort_suit_col( ( unsigned ) k );

        ov_put( PV_ROW0, ov_col( SORT_X0 + k * SORT_BAND_W + SORT_BAND_W / 2 ), SUIT_CH[ suit ],
                SUIT_RED( suit ) ? GUI_LRED : GUI_WHITE );
    }
    for( k = 0; k < SORT_ROWS; k++ )                    /* left of it: the row numbers */
        ov_put( ov_row( k * SORT_BAND_H + SORT_BAND_H / 2 ), c - 1, ( unsigned ) ( '1' + k ), LINE );

    for( i = 0; i < sg->nprev; i++ )                    /* what this find saw (grey: ignored, no card of ours) */
        ov_corners( &sg->prev[ i ], sort_in_air( &sg->prev[ i ] ) ? BRASS : sort_ignored( sg, &sg->prev[ i ] ) ? GUI_DGREY : GUI_WHITE, "" );

    if( sg->phase == SP_PLACE && sg->trow >= 0 && sg->last >= 0 ) {    /* where the card to place goes: brass edges */
        struct scan_card t;

        memset( &t, 0, sizeof t );
        t.x0 = ( uint16_t ) ( SORT_X0 + sg->tcol * SORT_BAND_W );
        t.x1 = ( uint16_t ) ( t.x0 + SORT_BAND_W > SORT_EDGE_X ? SORT_EDGE_X : t.x0 + SORT_BAND_W );
        t.y0 = ( uint16_t ) ( sg->trow * SORT_BAND_H );
        t.y1 = ( uint16_t ) ( t.y0 + SORT_BAND_H - 1 );
        r  = ov_row( t.y0 ); r1 = ov_row( t.y1 );
        c  = ov_col( t.x0 ); c1 = ov_col( t.x1 );
        for( k = c; k <= c1; k++ ) { ov_put( r, k, BX_H, BRASS ); ov_put( r1, k, BX_H, BRASS ); }
        for( k = r; k <= r1; k++ ) { ov_put( k, c, BX_V, BRASS ); ov_put( k, c1, BX_V, BRASS ); }
        sort_name( sg->card[ sg->last ].word, name, sizeof name );
        ov_corners( &t, BRASS, name );
        if( sg->card[ sg->last ].seen ) ov_corners( &sg->card[ sg->last ].box, BRASS, "" );    /* laid elsewhere: there it is */
        for( k = sg->slide_a; sg->slide_dir && k <= sg->slide_z; k++ )   /* the cards to slide first */
            ov_put( ov_row( k * SORT_BAND_H + SORT_BAND_H / 2 ), c1 - 1,
                    sg->slide_dir > 0 ? CH_ARROW_DN : CH_ARROW_UP, BRASS );
    }
    if( sg->phase == SP_ALERT ) {
        static const char MARK[ 2 ] = { CH_DBL_EXCL, 0 };

        ov_corners( &sg->abox, GUI_LRED, sg->alert == AL_JOKER ? "JOKER" : MARK );
    }

    for( r = 0; r < PV_ROWS; r++ )
        for( c = 0; c < PV_COLS; c++ ) bus_write( ( unsigned ) ( ( PV_ROW0 + r ) * COLS + PV_COL0 + c ), ov[ r ][ c ] );
}

static void sort_slot_pos( int col, int row, int * prow, int * pcol )
{
    *prow = SORT_TOP + row * GEO[ 1 ].py;
    *pcol = SORT_LEFT + col * GEO[ 1 ].px;
}

/* a red card where something must come off */
static void alert_card( int row, int col, int joker )
{
    card_shape( row, col, GEO[ 1 ].h, GEO[ 1 ].w, GUI_LRED );
    text( row + 1, col + 1, 0, joker ? "JOKER" : "?", GUI_BLACK, GUI_LRED );
    text( row + 3, col + 2, 0, "OFF!", GUI_BLACK, GUI_LRED );
}

static void draw_sort_board( const struct sort_game * sg )
{
    int      c, r, row, col;
    unsigned k;

    fill( GRID_ROW, GRID_COL, ROWS - 2, COLS - 1 - GRID_COL, ' ', INK, FELT );
    put( 1, COLS - 2, CH_LOWER, FELT, WOOD );           /* the felt's rounded corners */
    put( ROWS - 2, COLS - 2, CH_UPPER, FELT, WOOD );
    text( GRID_ROW, GRID_COL + 1, 0, " THE BOARD: HIGH TO LOW ", GUI_BLACK, BRASS );
    for( c = 0; c < SORT_COLS; c++ ) {
        unsigned suit = ( unsigned ) sort_suit_col( ( unsigned ) c );

        sort_slot_pos( c, 0, &row, &col );              /* the suit on a white plate over its column */
        fill( row - 2, col + 1, 1, GEO[ 1 ].w - 2, ' ', GUI_BLACK, GUI_WHITE );
        put( row - 2, col + GEO[ 1 ].w / 2, SUIT_CH[ suit ], SUIT_RED( suit ) ? GUI_RED : GUI_BLACK, GUI_WHITE );
        for( r = 0; r < SORT_ROWS; r++ ) {
            sort_slot_pos( c, r, &row, &col );
            if( !c ) put( row + GEO[ 1 ].h / 2, col - 2, ( unsigned ) ( '1' + r ), INK, FELT );
            card_slot( row, col, GEO[ 1 ].h, GEO[ 1 ].w, "", GUI_DGREY );
        }
    }
    for( k = 0; k < sg->n; k++ ) {                      /* the memory: every card scanned, in its slot */
        const struct sort_card * p = &sg->card[ k ];
        unsigned face = ( int ) k != sg->last ? GUI_WHITE : sg->phase == SP_PLACE ? BRASS : GUI_LCYAN;

        sort_slot_pos( p->col, p->row, &row, &col );
        card_face( row, col, GEO[ 1 ].h, GEO[ 1 ].w, p->word, face, 0 );
        if( p->seen ) put( row + 1, col + GEO[ 1 ].w - 2, p->bad ? CH_DBL_EXCL : CH_CHECK, p->bad ? GUI_RED : GUI_GREEN, face );   /* found: in order, or out of place */
    }
    for( r = sg->slide_a; sg->phase == SP_PLACE && sg->slide_dir && r <= sg->slide_z; r++ ) {
        sort_slot_pos( sg->tcol, r, &row, &col );       /* the cards that slide to make room */
        put( row + GEO[ 1 ].h / 2, col + GEO[ 1 ].w, sg->slide_dir > 0 ? CH_ARROW_DN : CH_ARROW_UP, BRASS, FELT );
    }
    if( sg->phase == SP_ALERT ) {
        if( sg->acov >= 0 ) { c = sg->card[ sg->acov ].col; r = sg->card[ sg->acov ].row; }
        else                sort_slot( &sg->abox, &c, &r );
        sort_slot_pos( c, r, &row, &col );
        alert_card( row, col, sg->alert == AL_JOKER );
    }
}

/* row 20, every find: what the camera sees -- and whether the camera is at a
 * height where cards fit the lines (SORT_ROWS rows of SORT_BAND_H px) */
static void draw_sort_live( const struct sort_game * sg )
{
    char     buf[ 64 ];
    unsigned i, j, k, n = 0, air = 0, h[ SCAN_MAX ], a[ SCAN_MAX ], t;

    for( i = 0; i < sg->nprev; i++ ) {                  /* heights, sorted: lying on the board / bigger than a slot */
        int        big = sort_in_air( &sg->prev[ i ] );
        unsigned * v   = big ? a : h;

        k = big ? air++ : n++;
        t = ( unsigned ) ( sg->prev[ i ].y1 - sg->prev[ i ].y0 + 1 );
        for( j = k; j > 0 && v[ j - 1 ] > t; j-- ) v[ j ] = v[ j - 1 ];
        v[ j ] = t;
    }
    /* One card is held up at a time: several bigger than a slot are cards
     * lying on the board seen too big -- the camera is too low, and the auto
     * re-scan ignores them all (sort_in_air), jokers too. */
    if( !sg->nprev )                        snprintf( buf, sizeof buf, "SEES NO CARD" );
    else if( air >= 2 )                     snprintf( buf, sizeof buf, "%u CARDS TOO BIG (%u PX): RAISE CAMERA", air, a[ air / 2 ] );
    else if( air && !n )                    snprintf( buf, sizeof buf, "BIG CARD: KEY0 IF HELD, ELSE RAISE CAM" );
    else if( air )                          snprintf( buf, sizeof buf, "CARD HELD UP: READY, PRESS KEY0" );
    else if( h[ n / 2 ] > SORT_BAND_H - 10 ) snprintf( buf, sizeof buf, "CARDS %u PX TALL: RAISE THE CAMERA", h[ n / 2 ] );
    else if( h[ n / 2 ] < 56u )              snprintf( buf, sizeof buf, "CARDS %u PX TALL: LOWER THE CAMERA", h[ n / 2 ] );
    else
        snprintf( buf, sizeof buf, "SEES %u CARD%s, %u PX TALL%s", n, n == 1 ? "" : "S", h[ n / 2 ],
                  sg->hiding ? ", SOME HIDDEN" : sg->still + 1u >= SORT_STILL ? ", STILL" : "" );
    text( LIVE_ROW, INFO_COL, INFO_W, buf, INK, FELT );
}

static void draw_sort_info( const struct sort_game * sg )
{
    char     buf[ 64 ];
    unsigned plate = sg->tone == TONE_ALERT ? GUI_LRED : BRASS, k, seen;

    text( HINT_ROW, 2, CAM_W - 3, sg->hint, INK, FELT );
    fill( INFO_ROW, 1, ROWS - 1 - INFO_ROW, CAM_W - 1, ' ', INK, FELT );
    text( INFO_ROW, INFO_COL, INFO_W, sg->prompt, GUI_BLACK, plate );
    snprintf( buf, sizeof buf, "TURN %u   SORTED %u OF %u   MISTAKES %u", sg->turn, sg->n, ( unsigned ) SORT_MAX, sg->mistakes );
    text( INFO_ROW + 2, INFO_COL, 0, buf, INK, FELT );
    for( k = seen = 0; k < sg->n; k++ ) seen += sg->card[ k ].seen;
    snprintf( buf, sizeof buf, "ON BOARD %u (SEEN)   JOKERS REMOVED %u", seen, sg->jokers );
    text( INFO_ROW + 3, INFO_COL, 0, buf, INK, FELT );
    if( sg->reads ) {
        snprintf( buf, sizeof buf, "LAST LOOK: %u CNN READ%s, %u.%u S", sg->reads, sg->reads == 1 ? "" : "S",
                  sg->ms / 1000u, sg->ms % 1000u / 100u );
        text( INFO_ROW + 4, INFO_COL, 0, buf, INK, FELT );
    }
    if( sg->phase == SP_PLACE && sg->last >= 0 ) {     /* the card to place */
        text( INFO_ROW + 6, INFO_COL, 0, "TO PLACE", INK, FELT );
        card_face( INFO_ROW + 5, INFO_COL + 10, GEO[ 2 ].h, GEO[ 2 ].w, sg->card[ sg->last ].word, GUI_WHITE, 0 );
    }
    snprintf( buf, sizeof buf, "last: %s", sg->note );
    text( ROWS - 2, INFO_COL, INFO_W, buf, INK, FELT );
}

void gui_sort_draw( const struct sort_game * sg, int full )
{
    static const unsigned TONE_COLOUR[ 4 ] = { GUI_LGREEN, GUI_YELLOW, GUI_LMAGENTA, GUI_LRED };

    draw_sort_preview( sg );
    if( full ) {
        draw_sort_board( sg );
        draw_sort_info( sg );
        gui_status( sg->pill, TONE_COLOUR[ sg->tone & 3u ] );
    }
    draw_sort_live( sg );
}
