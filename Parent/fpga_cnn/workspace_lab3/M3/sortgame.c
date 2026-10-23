/* M3 suit sort -- see sortgame.h. */
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "sortgame.h"

/* Every scan_card in the game word-aligned: with the MMU off a misaligned word
 * access is a data abort, and memcpy / struct copies use words (card_pipeline.h).
 * The shared layouts unchanged. */
#define SORT_ALIGNED( f )   ( offsetof( struct sort_game, f ) % 4u == 0u )
typedef char sort_game_aligned[ SORT_ALIGNED( card ) && SORT_ALIGNED( abox ) && SORT_ALIGNED( prev ) &&
                                SORT_ALIGNED( acted ) && SORT_ALIGNED( ghost ) && SORT_ALIGNED( gsig ) &&
                                offsetof( struct sort_card, box ) % 4u == 0u && sizeof( struct sort_card ) % 4u == 0u ? 1 : -1 ];
typedef char scan_layout_kept[ sizeof( struct scan_card ) == 12u && sizeof( struct scan_result ) == 316u &&
                               sizeof( struct win_regs ) == 16u ? 1 : -1 ];

#define ABS( a )    ( ( a ) < 0 ? -( a ) : ( a ) )
#define STILL_PX    8           /* a box this close in the next find has not moved */
#define MATCH_PX    16          /* one of our cards nudged this far is still that card */
#define NO_READ     0xFFu
#define ID_JOKER    0x40u

/* what each box of a scene is: >= 0 one of ours (card[] index), else */
#define W_NEW       ( -1 )      /* on the board and not seen there before: laid, or an intruder */
#define W_AIR       ( -2 )      /* bigger than a slot: a card held up, not on the board */
#define W_GHOST     ( -3 )      /* dismissed with KEY1: never read again */
#define W_DONE      ( -4 )      /* a new box already accounted for in this scene */

static const char SUIT_GLYPH[ 4 ] = { 0x06, 0x05, 0x03, 0x04 };    /* CP437, cnn_result suit order */

static void say( char * dst, unsigned n, const char * fmt, ... ) __attribute__( ( format( printf, 3, 4 ) ) );
static void say( char * dst, unsigned n, const char * fmt, ... )
{
    va_list ap;

    va_start( ap, fmt );
    vsnprintf( dst, n, fmt, ap );
    va_end( ap );
}
#define SAY( g, field, ... )    say( ( g )->field, sizeof ( g )->field, __VA_ARGS__ )

/* rank and suit, ID_JOKER, or NO_READ -- app_rtos.c prvIdentity */
static unsigned ident( uint32_t w )
{
    if( !RES_DONE( w ) || RES_DDR_ERR( w ) || ( !RES_JOKER( w ) && RES_RANK( w ) >= 13u ) ) return NO_READ;
    return RES_JOKER( w ) ? ID_JOKER : ( RES_RANK( w ) << 2 ) | RES_SUIT( w );
}

int sort_suit_col( unsigned suit )
{
    static const int COL[ 4 ] = { 0, 2, 1, 3 };     /* its own inverse: also column -> suit */

    return COL[ suit & 3u ];
}

void sort_name( uint32_t word, char * buf, unsigned n )
{
    if( ident( word ) == NO_READ ) snprintf( buf, n, "?" );
    else if( RES_JOKER( word ) )   snprintf( buf, n, "JOKER" );
    else                           snprintf( buf, n, "%s%c", RANK_NAMES[ RES_RANK( word ) ], SUIT_GLYPH[ RES_SUIT( word ) ] );
}

void sort_slot( const struct scan_card * b, int * col, int * row )
{
    int x = ( b->x0 + b->x1 ) / 2, y = ( b->y0 + b->y1 ) / 2;

    *col = x < SORT_X0 ? 0 : ( x - SORT_X0 ) / SORT_BAND_W;
    *row = y / SORT_BAND_H;
    if( *col >= SORT_COLS ) *col = SORT_COLS - 1;
    if( *row >= SORT_ROWS ) *row = SORT_ROWS - 1;
}

int sort_in_air( const struct scan_card * b )
{
    return b->y1 - b->y0 + 1 > SORT_BAND_H - 4 || b->x1 - b->x0 + 1 > SORT_BAND_W - 16;
}

/* "<suit> ROW 2" */
static void where( const struct scan_card * b, char * buf, unsigned n )
{
    int col, row;

    sort_slot( b, &col, &row );
    snprintf( buf, n, "%c ROW %d", SUIT_GLYPH[ sort_suit_col( ( unsigned ) col ) ], row + 1 );
}

void sort_sig( const struct scan_card * b, sort_gray_fn gray, uint8_t sig[ SORT_SIG_N ] )
{
    int      w = b->x1 - b->x0 + 1, h = b->y1 - b->y0 + 1;
    int      ix = b->x0 + w / 10, iy = b->y0 + h / 10, iw = w - 2 * ( w / 10 ), ih = h - 2 * ( h / 10 );
    unsigned v[ SORT_SIG_N ], top = 1, k;
    int      gx, gy, q;

    for( gy = 0; gy < 4; gy++ )
        for( gx = 0; gx < 4; gx++ ) {
            unsigned s = 0;

            for( q = 0; q < 4; q++ )        /* the cell's quarter points, 2 x 2 */
                s += gray( ix + iw * ( 8 * gx + 2 + 4 * ( q & 1 ) ) / 32,
                           iy + ih * ( 8 * gy + 2 + 4 * ( q >> 1 ) ) / 32 );
            v[ gy * 4 + gx ] = s;
            if( s > top ) top = s;
        }
    for( k = 0; k < SORT_SIG_N; k++ ) sig[ k ] = ( uint8_t ) ( v[ k ] * 255u / top );
}

unsigned sort_sig_diff( const uint8_t a[ SORT_SIG_N ], const uint8_t b[ SORT_SIG_N ] )
{
    unsigned d0 = 0, d1 = 0, k;

    for( k = 0; k < SORT_SIG_N; k++ ) {
        d0 += ( unsigned ) ABS( ( int ) a[ k ] - ( int ) b[ k ] );
        d1 += ( unsigned ) ABS( ( int ) a[ k ] - ( int ) b[ SORT_SIG_N - 1 - k ] );   /* turned 180 deg */
    }
    return ( d0 < d1 ? d0 : d1 ) / SORT_SIG_N;
}

/* the same box give or take px: centre and size */
static int near( const struct scan_card * a, const struct scan_card * b, int px )
{
    return ABS( ( a->x0 + a->x1 ) - ( b->x0 + b->x1 ) ) <= 2 * px &&
           ABS( ( a->y0 + a->y1 ) - ( b->y0 + b->y1 ) ) <= 2 * px &&
           ABS( ( a->x1 - a->x0 ) - ( b->x1 - b->x0 ) ) <= px &&
           ABS( ( a->y1 - a->y0 ) - ( b->y1 - b->y0 ) ) <= px;
}

/* two finds of the same boxes, in any order */
static int same_boxes( const struct scan_card * a, unsigned na, const struct scan_card * b, unsigned nb )
{
    uint32_t used = 0;
    unsigned i, j;

    if( na != nb ) return 0;
    for( i = 0; i < na; i++ ) {
        for( j = 0; j < nb && ( ( ( used >> j ) & 1u ) || !near( &a[ i ], &b[ j ], STILL_PX ) ); j++ ) ;
        if( j == nb ) return 0;
        used |= 1u << j;
    }
    return 1;
}

/* the scene already acted on: the same boxes, each looking the same */
static int seen_before( const struct sort_game * g, const struct scan_result * s )
{
    uint32_t used = 0;
    unsigned i, j;

    if( !g->acted_ok || s->n != g->nacted ) return 0;
    for( i = 0; i < s->n; i++ ) {
        for( j = 0; j < g->nacted; j++ )
            if( !( ( used >> j ) & 1u ) && near( &s->card[ i ], &g->acted[ j ], STILL_PX ) &&
                sort_sig_diff( g->look[ i ], g->alook[ j ] ) <= SORT_SIG_DIFF ) break;
        if( j == g->nacted ) return 0;
        used |= 1u << j;
    }
    return 1;
}

/* Ghosts: boxes that are no card of ours -- dismissed with KEY1, or read once
 * and found to be neither a joker nor a scanned card. Not read again while
 * they lie there; forgotten once gone (judge). */

/* the ghost box b is: the same place and the same look (a ghost without a
 * look, all 0, by place alone); -1 if none. A card laid on a ghost's spot
 * looks different, so it is read. */
static int ghost_of( const struct sort_game * g, const struct scan_card * b, const uint8_t * look )
{
    unsigned i, k;

    for( i = 0; i < g->nghost; i++ ) {
        for( k = 0; k < SORT_SIG_N && !g->gsig[ i ][ k ]; k++ ) ;
        if( near( b, &g->ghost[ i ], MATCH_PX ) && ( k == SORT_SIG_N || sort_sig_diff( look, g->gsig[ i ] ) <= SORT_SIG_DIFF ) )
            return ( int ) i;
    }
    return -1;
}

/* b, looking like `look` (NULL: any look), is a ghost from now on */
static void add_ghost( struct sort_game * g, const struct scan_card * b, const uint8_t * look )
{
    if( g->nghost == SORT_GHOSTS ) {                    /* full: the oldest goes */
        memmove( &g->ghost[ 0 ], &g->ghost[ 1 ], ( SORT_GHOSTS - 1 ) * sizeof g->ghost[ 0 ] );
        memmove( g->gsig[ 0 ], g->gsig[ 1 ], ( SORT_GHOSTS - 1 ) * sizeof g->gsig[ 0 ] );
        g->nghost--;
    }
    g->ghost[ g->nghost ] = *b;
    if( look ) memcpy( g->gsig[ g->nghost ], look, SORT_SIG_N );
    else       memset( g->gsig[ g->nghost ], 0, SORT_SIG_N );
    g->nghost++;
}

/* only the ghosts this find saw (bit i: ghost i) stay */
static void keep_ghosts( struct sort_game * g, uint32_t seen )
{
    unsigned i, n = 0;

    for( i = 0; i < g->nghost; i++ ) {
        if( !( ( seen >> i ) & 1u ) ) continue;
        if( n != i ) {
            g->ghost[ n ] = g->ghost[ i ];
            memcpy( g->gsig[ n ], g->gsig[ i ], SORT_SIG_N );
        }
        n++;
    }
    g->nghost = n;
}

int sort_ignored( const struct sort_game * g, const struct scan_card * b )
{
    unsigned i;

    for( i = 0; i < g->nghost; i++ )
        if( near( b, &g->ghost[ i ], MATCH_PX ) ) return 1;
    return 0;
}

static uint32_t read_card( struct sort_game * g, sort_read_fn rd, const struct scan_card * c, int how, unsigned * agree )
{
    unsigned n = 0, a = 0;
    uint32_t w = rd( c, how, &n, &a );

    g->reads += n;
    if( agree ) *agree = a;
    return w;
}

static int col_of( const struct scan_card * b )
{
    int col, row;

    sort_slot( b, &col, &row );
    return col;
}

static int row_of( const struct scan_card * b )
{
    int col, row;

    sort_slot( b, &col, &row );
    return row;
}

static void forget( struct sort_game * g, unsigned k )
{
    memmove( &g->card[ k ], &g->card[ k + 1 ], ( g->n - k - 1 ) * sizeof g->card[ 0 ] );
    g->n--;
    if( g->last == ( int ) k ) g->last = -1;
    else if( g->last > ( int ) k ) g->last--;
}

int sort_on_board( const struct sort_game * g, uint32_t word )
{
    unsigned k;

    if( ident( word ) == NO_READ || ident( word ) == ID_JOKER ) return 0;
    for( k = 0; k < g->n; k++ )
        if( ident( g->card[ k ].word ) == ident( word ) ) return 1;
    return 0;
}

static void say_idle( struct sort_game * g )
{
    SAY( g, prompt, "HOLD A CARD UP, PRESS KEY0" );
    if( g->n ) SAY( g, hint, "%u SORTED. BRASS CORNERS = READY", g->n );
    else       SAY( g, hint, "BRASS CORNERS ON IT = READY TO READ" );
    SAY( g, pill, "YOUR TURN" );
    g->tone = TONE_OK;
}

/* card[ last ] is the card to place: where it goes */
static void say_place( struct sort_game * g )
{
    char name[ 8 ];
    const struct sort_card * c = &g->card[ g->last ];

    sort_name( c->word, name, sizeof name );
    SAY( g, prompt, "PLACE %s: %c COLUMN, ROW %d", name, SUIT_GLYPH[ RES_SUIT( c->word ) ], g->trow + 1 );
    SAY( g, hint, "%s", g->phint );
    SAY( g, pill, "PLACE %s", name );
    g->tone = TONE_BUSY;
}

/* ---- the turn ------------------------------------------------------------ */

/* Where `word` goes in the board memory: its suit's column, any free row below
 * every higher card and above every lower one -- the free row nearest to where
 * its rank would sit on a full column (A K Q J top ... 4 3 2 bottom), so later
 * cards keep room. No free row between them: the run below slides down one (or
 * the run above up one). Sets tcol / trow, the slide and the hint (phint);
 * moves nothing -- store() does. 0 if the column is full. */
static int plan( struct sort_game * g, uint32_t word )
{
    int      col = sort_suit_col( RES_SUIT( word ) ), rank = ( int ) RES_RANK( word );
    int      occ[ SORT_ROWS ], above = -1, below = SORT_ROWS, ideal, best = -1, r, fr;
    char     name[ 8 ], up[ 8 ] = "", down[ 8 ] = "", slide[ 24 ] = "";
    char     glyph = SUIT_GLYPH[ RES_SUIT( word ) ];
    unsigned k;

    for( r = 0; r < SORT_ROWS; r++ ) occ[ r ] = -1;
    for( k = 0; k < g->n; k++ ) {
        if( g->card[ k ].col != col ) continue;
        r        = g->card[ k ].row;
        occ[ r ] = ( int ) k;
        if( ( int ) RES_RANK( g->card[ k ].word ) > rank ) { if( r > above ) above = r; }
        else if( r < below ) below = r;
    }
    sort_name( word, name, sizeof name );
    ideal = ( 12 - rank ) * SORT_ROWS / 13;
    for( r = above + 1; r < below; r++ )
        if( occ[ r ] < 0 && ( best < 0 || ABS( r - ideal ) < ABS( best - ideal ) ) ) best = r;

    g->slide_dir = 0;
    if( best < 0 ) {                                    /* no room between: make some */
        int a, z, n = 0;

        for( fr = below; fr < SORT_ROWS && occ[ fr ] >= 0; fr++ ) ;
        if( fr < SORT_ROWS ) { a = below; z = fr - 1; best = below; }
        else {
            for( fr = above; fr >= 0 && occ[ fr ] >= 0; fr-- ) ;
            if( fr < 0 ) {
                SAY( g, prompt, "%c COLUMN IS FULL: SET %s ASIDE", glyph, name );
                SAY( g, hint, "%d ROWS PER SUIT", SORT_ROWS );
                SAY( g, pill, "COLUMN FULL" );
                g->tone = TONE_WARN;
                return 0;
            }
            a = fr + 1; z = above; best = above;
        }
        for( r = a; r <= z; r++ ) {
            char nm[ 8 ];

            sort_name( g->card[ occ[ r ] ].word, nm, sizeof nm );
            n += snprintf( slide + n, sizeof slide - ( unsigned ) n, "%s ", nm );
            if( n >= ( int ) sizeof slide - 6 ) break;
        }
        SAY( g, phint, "FIRST SLIDE %s%s 1 ROW", slide, best == below ? "DOWN" : "UP" );
        g->slide_a   = ( int8_t ) a;
        g->slide_z   = ( int8_t ) z;
        g->slide_dir = ( int8_t ) ( best == below ? 1 : -1 );
    }
    else {
        for( r = best - 1; r >= 0 && occ[ r ] < 0; r-- ) ;
        if( r >= 0 ) sort_name( g->card[ occ[ r ] ].word, up, sizeof up );
        for( r = best + 1; r < SORT_ROWS && occ[ r ] < 0; r++ ) ;
        if( r < SORT_ROWS ) sort_name( g->card[ occ[ r ] ].word, down, sizeof down );
        if( *up && *down ) SAY( g, phint, "BELOW %s, ABOVE %s", up, down );
        else if( *up )     SAY( g, phint, "BELOW %s", up );
        else if( *down )   SAY( g, phint, "ABOVE %s", down );
        else               SAY( g, phint, "FIRST %c CARD: ANY ROW IS FINE", glyph );
    }
    g->tcol = ( int8_t ) col;
    g->trow = ( int8_t ) best;
    return 1;
}

/* Into the board memory at the planned slot -- its column's run moved first,
 * on the map at once, on the board by hand -- not seen by the camera yet. It
 * is the card to place from now on. */
static void store( struct sort_game * g, uint32_t word )
{
    struct sort_card * c = &g->card[ g->n ];
    unsigned           k;

    for( k = 0; g->slide_dir && k < g->n; k++ )
        if( g->card[ k ].col == g->tcol && g->card[ k ].row >= g->slide_a && g->card[ k ].row <= g->slide_z )
            g->card[ k ].row = ( int8_t ) ( g->card[ k ].row + g->slide_dir );
    if( g->slide_dir ) {                                /* the run's rows as moved: arrows, undo */
        g->slide_a = ( int8_t ) ( g->slide_a + g->slide_dir );
        g->slide_z = ( int8_t ) ( g->slide_z + g->slide_dir );
    }
    memset( c, 0, sizeof *c );
    c->word = word;
    c->col  = g->tcol;
    c->row  = g->trow;
    g->last = ( int8_t ) g->n++;
    g->turn++;
    g->phase = SP_PLACE;
    say_place( g );
}

/* card[ last ] out of the board memory again; its slide undone if it was
 * still to be placed */
static void unstore( struct sort_game * g )
{
    unsigned k;

    if( g->last < 0 ) return;
    for( k = 0; g->phase == SP_PLACE && g->slide_dir && k < g->n; k++ )
        if( ( int ) k != g->last && g->card[ k ].col == g->tcol &&
            g->card[ k ].row >= g->slide_a && g->card[ k ].row <= g->slide_z )
            g->card[ k ].row = ( int8_t ) ( g->card[ k ].row - g->slide_dir );
    forget( g, ( unsigned ) g->last );
    if( g->turn ) g->turn--;
    g->slide_dir = 0;
    g->phase     = SP_SHOW;
}

/* card k found on the board at box b, looking like `look` -- in the memory it
 * keeps its suit's column: where it lies is judged by board_problem */
static void found( struct sort_game * g, unsigned k, const struct scan_card * b, const uint8_t * look )
{
    struct sort_card * c = &g->card[ k ];

    c->box = *b;
    memcpy( c->sig, look, SORT_SIG_N );
    c->seen = 1;
}

/* box b is one of this find's: its card lies there now (not hidden, and not
 * where it lay before the camera moved) */
static int in_find( const struct scan_result * s, const struct scan_card * b )
{
    unsigned i;

    for( i = 0; i < s->n; i++ )
        if( s->card[ i ].x0 == b->x0 && s->card[ i ].y0 == b->y0 && s->card[ i ].x1 == b->x1 && s->card[ i ].y1 == b->y1 )
            return 1;
    return 0;
}

/* The first rule the cards on the board break, as an instruction; 0 if they
 * are in order. Judged between the cards in this find, not by the lines over
 * the preview, so a camera off to one side, higher or lower, or moved since,
 * judges the same: a suit stacked in one column (centres less than a card
 * width apart), higher ranks above lower ones, the suits' columns left to
 * right as on the panel (a card width apart at least). Of two cards out of
 * order the one scanned later is out of place (bad); a rule card[ first ]
 * breaks is named before the rest. Cards not in this find keep their mark. */
static int board_problem( struct sort_game * g, const struct scan_result * s, int first, char * msg, unsigned n )
{
    int      here[ SORT_MAX ];
    unsigned i, j, named = 0;                           /* 2: a rule card[ first ] breaks */

    for( i = 0; i < g->n; i++ ) {
        here[ i ] = g->card[ i ].seen && in_find( s, &g->card[ i ].box );
        if( here[ i ] ) g->card[ i ].bad = 0;
    }
    for( j = 1; j < g->n; j++ )
        for( i = 0; i < j; i++ ) {
            const struct scan_card * p = &g->card[ i ].box, * q = &g->card[ j ].box;
            int      cp = sort_suit_col( RES_SUIT( g->card[ i ].word ) ), cq = sort_suit_col( RES_SUIT( g->card[ j ].word ) );
            int      dx = ( q->x0 + q->x1 - p->x0 - p->x1 ) / 2, dy = ( q->y0 + q->y1 - p->y0 - p->y1 ) / 2;
            int      w = ( q->x1 - q->x0 + p->x1 - p->x0 ) / 2, h = ( q->y1 - q->y0 + p->y1 - p->y0 ) / 2;
            int      up = RES_RANK( g->card[ j ].word ) > RES_RANK( g->card[ i ].word );    /* card j goes above card i */
            unsigned pri = ( int ) j == first ? 2u : 1u;
            char     a[ 8 ], c[ 8 ], m[ 40 ];

            if( !here[ i ] || !here[ j ] ) continue;
            sort_name( g->card[ j ].word, a, sizeof a );
            sort_name( g->card[ i ].word, c, sizeof c );
            if( cp == cq && ABS( dx ) > w )                     snprintf( m, sizeof m, "%s GOES IN %s'S COLUMN", a, c );
            else if( cp == cq && ( up ? -dy : dy ) < h / 2 )    snprintf( m, sizeof m, "%s MUST GO %s %s", a, up ? "ABOVE" : "BELOW", c );
            else if( cp != cq && ( cq > cp ? dx : -dx ) < w )   snprintf( m, sizeof m, "%s MUST GO %s OF %s", a, cq > cp ? "RIGHT" : "LEFT", c );
            else                                                continue;
            g->card[ j ].bad = 1;
            if( named < pri ) {
                snprintf( msg, n, "%s", m );
                named = pri;
            }
        }
    return named != 0;
}

/* card[ last ], the card to place, has just been found on the board: in
 * order -- done; else what to fix (a mistake, counted once until it is in
 * order again: found out of place again -- the camera moved, say -- it is
 * the same mistake) */
static void check_placed( struct sort_game * g, const struct scan_result * s )
{
    struct sort_card * c = &g->card[ g->last ];
    char               name[ 8 ], why[ 40 ];
    int                was = c->bad;

    sort_name( c->word, name, sizeof name );
    board_problem( g, s, g->last, why, sizeof why );
    if( c->bad ) {
        if( !was ) g->mistakes++;
        SAY( g, prompt, "%s", why );
        SAY( g, hint, "TARGET: %c COLUMN, ROW %d", SUIT_GLYPH[ RES_SUIT( c->word ) ], g->trow + 1 );
        SAY( g, pill, "WRONG PLACE" );
        g->tone = TONE_WARN;
        return;
    }
    g->phase = SP_SHOW;
    SAY( g, note, "TURN %u: %s PLACED, %c ROW %d", g->turn, name, SUIT_GLYPH[ RES_SUIT( c->word ) ], c->row + 1 );
    SAY( g, prompt, "GOOD! NEXT: HOLD A CARD UP, KEY0" );
    SAY( g, hint, "%u CARD%s SORTED", g->n, g->n == 1 ? "" : "S" );
    SAY( g, pill, "PLACED %s", name );
    g->tone = TONE_OK;
}

/* Point at something that must come off the board */
static void raise_alert( struct sort_game * g, int kind, const struct scan_card * b, uint32_t word, int cov )
{
    char at[ 16 ], name[ 8 ], ours[ 8 ] = "";

    if( g->phase != SP_ALERT ) g->resume = g->phase;
    g->phase = SP_ALERT;
    g->alert = ( uint8_t ) kind;
    g->abox  = *b;
    g->aword = word;
    g->acov  = ( int8_t ) cov;
    where( b, at, sizeof at );
    sort_name( word, name, sizeof name );
    if( cov >= 0 ) sort_name( g->card[ cov ].word, ours, sizeof ours );
    g->tone = TONE_ALERT;
    if( kind == AL_JOKER ) {
        SAY( g, prompt, "JOKER ON %s: REMOVE IT!", at );
        if( cov >= 0 ) SAY( g, hint, "IT COVERS YOUR %s", ours );
        else           SAY( g, hint, "THE OPPONENT PLAYED A JOKER" );
        SAY( g, pill, "JOKER! REMOVE IT" );
        SAY( g, note, "JOKER FOUND ON %s", at );
    }
    else if( kind == AL_CARD ) {
        SAY( g, prompt, "TAKE %s OFF %s", name, at );
        SAY( g, hint, "ONE CARD AT A TIME, KEY0 FIRST" );
        SAY( g, pill, "TAKE %s OFF", name );
    }
    else {
        if( cov >= 0 ) SAY( g, prompt, "%s NOT SEEN: UNCOVER IT", ours );
        else           SAY( g, prompt, "SOMETHING ON %s: CLEAR IT", at );
        SAY( g, hint, "KEY1: IGNORE IT" );
        SAY( g, pill, "CHECK %s", at );
    }
}

/* the alert is over: what it pointed at is gone, or our card it covered is
 * back -- where it was and looking like itself, or reading as itself
 * anywhere in its slot */
static int alert_gone( struct sort_game * g, const struct scan_result * s, sort_read_fn rd, const int * who, const int * at )
{
    unsigned i;

    if( g->acov >= 0 ) {
        struct sort_card * c = &g->card[ g->acov ];

        if( at[ g->acov ] >= 0 ) return 1;
        for( i = 0; i < s->n; i++ ) {
            uint32_t w;

            if( who[ i ] != W_NEW || col_of( &s->card[ i ] ) != col_of( &c->box ) ||
                row_of( &s->card[ i ] ) != row_of( &c->box ) ) continue;
            w = read_card( g, rd, &s->card[ i ], READ_VERIFY, NULL );
            if( ident( w ) != ident( c->word ) ) w = read_card( g, rd, &s->card[ i ], READ_FAST, NULL );
            if( ident( w ) != ident( c->word ) ) return 0;
            found( g, ( unsigned ) g->acov, &s->card[ i ], g->look[ i ] );     /* itself, in another light */
            return 1;
        }
        return 0;
    }
    for( i = 0; i < s->n; i++ )
        if( who[ i ] != W_GHOST && near( &s->card[ i ], &g->abox, MATCH_PX ) ) return 0;
    return 1;
}

/* `removed`: it came off the board (a joker removed counts), not dismissed */
static void end_alert( struct sort_game * g, int removed )
{
    char name[ 8 ] = "";

    if( g->acov >= 0 ) sort_name( g->card[ g->acov ].word, name, sizeof name );
    if( removed && g->alert == AL_JOKER ) g->jokers++;
    if( !removed )                  SAY( g, note, "IGNORED, KEY1" );
    else if( g->alert == AL_JOKER ) SAY( g, note, "JOKER REMOVED" );
    else if( g->alert == AL_CARD )  SAY( g, note, "CARD TAKEN OFF" );
    else if( g->acov >= 0 )         SAY( g, note, "%s IS BACK", name );
    else                            SAY( g, note, "CLEARED" );
    g->alert = AL_NONE;
    g->acov  = -1;
    g->phase = g->resume;
    if( g->phase == SP_PLACE && g->last >= 0 ) say_place( g );     /* again: where the card to place goes */
    else {
        g->phase = SP_SHOW;
        say_idle( g );
    }
}

/* one of ours, seen on the board but missing from where it was, whose box
 * overlaps b; -1 if none */
static int covered( const struct sort_game * g, const int * at, const struct scan_card * b )
{
    unsigned k;

    for( k = 0; k < g->n; k++ ) {
        const struct scan_card * c = &g->card[ k ].box;

        if( g->card[ k ].seen && at[ k ] < 0 && c->x0 <= b->x1 && b->x0 <= c->x1 && c->y0 <= b->y1 && b->y0 <= c->y1 )
            return ( int ) k;
    }
    return -1;
}

/* The card to place, laid down, read as another suit of its rank: the
 * board's read -- the card's own window, its colour from the index corners --
 * is the better one and replaces the held one in the board memory. Judged
 * where it lies. 0 if its column is full (an alert). */
static int reread( struct sort_game * g, const struct scan_result * s, uint32_t word, const struct scan_card * b, const uint8_t * look )
{
    char was[ 8 ], now[ 8 ];

    sort_name( g->card[ g->last ].word, was, sizeof was );
    sort_name( word, now, sizeof now );
    unstore( g );
    SAY( g, note, "THE BOARD READ %s, NOT %s", now, was );
    if( !plan( g, word ) ) {
        raise_alert( g, AL_CARD, b, word, -1 );
        return 0;
    }
    store( g, word );                                   /* a slide too: one card per slot in the memory */
    found( g, ( unsigned ) g->last, b, look );
    check_placed( g, s );
    return 1;
}

/* The new boxes on the board are, if anything, our cards: moved ones (a slide
 * to make room) and scanned ones not seen on the board yet -- the card to
 * place among them. One box and one candidate that look alike need no read;
 * otherwise each box is read once and matched (a column holds one suit, so
 * the rank decides), and a box that matches nothing is read properly: a joker
 * raises an alert; the card to place read as another suit of its rank is
 * that card (reread); anything else is no card of ours -- one laid without
 * KEY0, glare, a mark -- and becomes a ghost, never part of the memory. 0 if
 * an alert was raised. */
static int account_new( struct sort_game * g, const struct scan_result * s, sort_read_fn rd, int * who, int * at )
{
    struct { int k; uint32_t word; const uint8_t * sig; int box; } c[ SORT_MAX ];
    unsigned                 nc = 0, nnew = 0, nx = 0, i, j, k;
    int                      xbox = -1, placed = 0;
    uint32_t                 xword = 0;
    const struct sort_card * p = g->last >= 0 ? &g->card[ g->last ] : NULL;    /* the card to place ... */
    int                      waiting = g->phase == SP_PLACE && p && !p->seen && !RES_JOKER( p->word );  /* ... not found yet */

    for( k = 0; k < g->n; k++ )
        if( at[ k ] < 0 ) { c[ nc ].k = ( int ) k; c[ nc ].word = g->card[ k ].word; c[ nc ].sig = g->card[ k ].sig; c[ nc++ ].box = -1; }
    for( i = 0; i < s->n; i++ ) nnew += who[ i ] == W_NEW;

    for( i = 0; i < s->n; i++ ) {
        int      pick = -1;
        unsigned agree = 0;
        uint32_t w;

        if( who[ i ] != W_NEW ) continue;
        if( nnew == 1 && nc == 1 && g->card[ c[ 0 ].k ].seen &&
            sort_sig_diff( g->look[ i ], c[ 0 ].sig ) <= SORT_SIG_DIFF ) pick = 0;
        else {
            w = read_card( g, rd, &s->card[ i ], READ_VERIFY, NULL );
            for( j = 0; j < nc && pick < 0; j++ ) if( c[ j ].box < 0 && ident( c[ j ].word ) == ident( w ) ) pick = ( int ) j;
            if( pick < 0 ) {
                w = read_card( g, rd, &s->card[ i ], READ_FAST, &agree );
                for( j = 0; j < nc && pick < 0; j++ ) if( c[ j ].box < 0 && ident( c[ j ].word ) == ident( w ) ) pick = ( int ) j;
            }
            if( pick < 0 ) {
                if( ident( w ) == ID_JOKER ) {
                    raise_alert( g, AL_JOKER, &s->card[ i ], w, covered( g, at, &s->card[ i ] ) );
                    return 0;
                }
                who[ i ] = W_DONE;
                if( xbox < 0 && waiting && ident( w ) != NO_READ && agree >= 2 &&
                    RES_RANK( w ) == RES_RANK( p->word ) && !sort_on_board( g, w ) ) {
                    xbox  = ( int ) i;
                    xword = w;
                }
                else if( covered( g, at, &s->card[ i ] ) < 0 ) {     /* over one of ours: step 3 reports it */
                    add_ghost( g, &s->card[ i ], g->look[ i ] );
                    nx++;
                }
                continue;
            }
        }
        c[ pick ].box = ( int ) i;
        who[ i ] = W_DONE;
    }

    for( j = 0; j < nc; j++ ) {                         /* ours, found: moved, or laid as told */
        char name[ 8 ];

        if( c[ j ].box < 0 ) continue;
        k = ( unsigned ) c[ j ].k;
        if( !g->card[ k ].seen && ( int ) k != g->last ) {
            sort_name( g->card[ k ].word, name, sizeof name );
            SAY( g, note, "%s IS ON THE BOARD", name );
        }
        found( g, k, &s->card[ c[ j ].box ], g->look[ c[ j ].box ] );
        at[ k ] = c[ j ].box;
        placed |= ( int ) k == g->last && g->phase == SP_PLACE;
    }
    if( placed ) check_placed( g, s );
    if( xbox >= 0 && !placed ) {                        /* the card to place, misread while held */
        if( !reread( g, s, xword, &s->card[ xbox ], g->look[ xbox ] ) ) return 0;
        at[ g->last ] = xbox;
    }
    else if( xbox >= 0 ) {                              /* it was found elsewhere: this is not ours */
        add_ghost( g, &s->card[ xbox ], g->look[ xbox ] );
        nx++;
    }
    if( nx ) SAY( g, note, "IGNORED %u NOT SCANNED: KEY0 FIRST", nx );
    return 1;
}

static void judge( struct sort_game * g, const struct scan_result * s, sort_read_fn rd )
{
    int      who[ SCAN_MAX ], at[ SORT_MAX ], miss = -1;
    unsigned i, k, nnew = 0;
    uint32_t gseen = 0;
    char     why[ 40 ];

    for( i = 0; i < s->n; i++ ) {
        int gh = ghost_of( g, &s->card[ i ], g->look[ i ] );

        if( gh >= 0 ) gseen |= 1u << gh;
        who[ i ] = gh >= 0 ? W_GHOST : sort_in_air( &s->card[ i ] ) ? W_AIR : W_NEW;
    }
    keep_ghosts( g, gseen );                            /* gone: forgotten -- what lies there next is read */
    /* ours on the board: where they were, give or take a nudge, and looking the
     * same. A different look there is another card -- ours slid into its
     * place, or something laid over it -- and is read below like any new box.
     * Cards scanned but not seen on the board yet are matched there too. */
    for( k = 0; k < SORT_MAX; k++ ) at[ k ] = -1;
    for( k = 0; k < g->n; k++ ) {
        if( !g->card[ k ].seen ) continue;
        for( i = 0; i < s->n && at[ k ] < 0; i++ )
            if( who[ i ] == W_NEW && near( &s->card[ i ], &g->card[ k ].box, MATCH_PX ) &&
                sort_sig_diff( g->look[ i ], g->card[ k ].sig ) <= SORT_SIG_DIFF ) {
                who[ i ] = ( int ) k;
                at[ k ]  = ( int ) i;
            }
    }
    for( i = 0; i < s->n; i++ ) nnew += who[ i ] == W_NEW;

    if( g->fresh ) {                                    /* a new game starts on an empty board */
        if( nnew ) {
            SAY( g, prompt, "CLEAR THE BOARD: %u CARD%s ON IT", nnew, nnew == 1 ? "" : "S" );
            SAY( g, hint, "NOTHING THERE? KEY1 IGNORES IT" );
            SAY( g, pill, "CLEAR THE BOARD" );
            g->tone = TONE_WARN;
            return;
        }
        g->fresh = 0;
    }

    if( g->phase == SP_ALERT ) {
        if( !alert_gone( g, s, rd, who, at ) ) return;
        end_alert( g, 1 );
    }

    /* 1: ours that stayed follow small nudges (the memory keeps their slot) */
    for( k = 0; k < g->n; k++ )
        if( at[ k ] >= 0 ) g->card[ k ].box = s->card[ at[ k ] ];

    /* 2: new boxes on the board */
    if( nnew && !account_new( g, s, rd, who, at ) ) return;

    /* 3: ours gone from where the camera saw them, with nothing new to explain
     * it: most likely a hand or a card held up over them, so only after
     * SORT_HIDDEN_MS is it reported -- covered by something locate cannot box
     * (a joker at an angle), or taken away. Cards slid out of the target
     * column wait for the placement. Cards never seen on the board are not
     * missed: they are only in the memory so far. */
    for( k = 0; k < g->n && miss < 0; k++ )
        if( g->card[ k ].seen && at[ k ] < 0 && !( g->phase == SP_PLACE && g->card[ k ].col == g->tcol ) )
            miss = ( int ) k;
    if( miss < 0 ) g->hiding = 0;
    else {
        uint32_t w;

        if( !g->hiding ) {
            g->hiding    = 1;
            g->hidden_at = g->now;
        }
        if( g->now - g->hidden_at < SORT_HIDDEN_MS ) return;
        g->hiding = 0;
        w = read_card( g, rd, &g->card[ miss ].box, READ_FAST, NULL );
        raise_alert( g, ident( w ) == ID_JOKER ? AL_JOKER : AL_THING, &g->card[ miss ].box,
                     ident( w ) == ID_JOKER ? w : 0u, miss );
        return;
    }

    /* 4: the board as it now lies must still be in order (a slide can break it) */
    if( board_problem( g, s, g->phase == SP_PLACE ? g->last : -1, why, sizeof why ) ) {
        SAY( g, prompt, "FIX: %s", why );
        SAY( g, pill, "FIX THE ORDER" );
        g->tone = TONE_WARN;
        return;
    }

    /* 5: nothing to place: say what to do next (keeping "GOOD!" until then) */
    if( g->phase == SP_SHOW && ( g->tone != TONE_OK || !g->n ) ) say_idle( g );
}

/* ---- API ------------------------------------------------------------------ */

void sort_new( struct sort_game * g )
{
    memset( g, 0, sizeof *g );
    g->phase = g->resume = SP_SHOW;
    g->fresh = 1;
    g->last  = -1;
    g->acov  = -1;
    g->tcol  = g->trow = -1;
    say_idle( g );
    SAY( g, note, "NEW GAME" );
    SAY( g, pill, "NEW GAME" );
}

int sort_tick( struct sort_game * g, const struct scan_result * s, unsigned now_ms, sort_read_fn rd, sort_gray_fn gray )
{
    unsigned i, n = s->n < SCAN_MAX ? s->n : SCAN_MAX;
    int      same = same_boxes( s->card, n, g->prev, g->nprev ), due;

    memcpy( g->prev, s->card, n * sizeof s->card[ 0 ] );
    g->nprev = n;
    g->now   = now_ms;
    g->reads = 0;
    g->still = same ? g->still + 1u : 0u;
    if( g->still + 1u < SORT_STILL ) return TICK_MOVING;
    for( i = 0; i < n; i++ ) sort_sig( &s->card[ i ], gray, g->look[ i ] );
    due = g->hiding && now_ms - g->hidden_at >= SORT_HIDDEN_MS;     /* hidden long enough: look again */
    if( !due && seen_before( g, s ) ) return TICK_SAME;

    memcpy( g->acted, s->card, n * sizeof s->card[ 0 ] );
    memcpy( g->alook, g->look, n * sizeof g->look[ 0 ] );
    g->nacted   = n;
    g->acted_ok = 1;
    judge( g, s, rd );
    g->events++;
    return TICK_EVENT;
}

void sort_held( struct sort_game * g, uint32_t word, unsigned reads, unsigned agree )
{
    char name[ 8 ];

    g->reads = reads;
    g->events++;
    sort_name( word, name, sizeof name );
    if( g->phase == SP_ALERT ) {
        SAY( g, note, "KEY0: CLEAR THE ALERT FIRST" );
        return;
    }
    if( ident( word ) == ID_JOKER ) {
        SAY( g, prompt, "THAT'S A JOKER: NOT YOURS TO SORT" );
        SAY( g, hint, "HOLD ONE OF YOUR CARDS UP, KEY0" );
        SAY( g, pill, "JOKER?" );
    }
    else if( ident( word ) == NO_READ || agree < 2 ) {
        SAY( g, prompt, "CAN'T READ IT: TRY AGAIN, KEY0" );
        SAY( g, hint, "WHOLE CARD IN VIEW, BRASS CORNERS" );
        SAY( g, pill, "READ AGAIN" );
    }
    else if( sort_on_board( g, word ) ) {
        SAY( g, prompt, "%s IS ALREADY ON THE BOARD", name );
        SAY( g, hint, "KEY0 TO READ IT AGAIN" );
        SAY( g, pill, "ALREADY PLACED" );
    }
    else {
        if( plan( g, word ) ) {
            SAY( g, note, "SCANNED %s (%u READ%s)", name, reads, reads == 1 ? "" : "S" );
            store( g, word );
        }
        return;
    }
    g->tone = TONE_WARN;
}

void sort_command( struct sort_game * g, int cmd )
{
    char name[ 8 ];

    if( cmd == SORT_CMD_NEW ) {
        sort_new( g );
        g->events++;
        return;
    }
    if( cmd == SORT_CMD_UNDO ) {
        if( g->phase == SP_ALERT ) {                    /* not a real problem: stop pointing at it */
            if( g->acov >= 0 ) forget( g, ( unsigned ) g->acov );
            else               add_ghost( g, &g->abox, NULL );
            end_alert( g, 0 );
        }
        else if( g->fresh ) {                           /* what lies on the "empty" board is no card */
            unsigned i, n = 0;

            for( i = 0; i < g->nacted; i++ )
                if( !sort_in_air( &g->acted[ i ] ) ) { add_ghost( g, &g->acted[ i ], g->alook[ i ] ); n++; }
            g->fresh = 0;
            say_idle( g );
            SAY( g, note, "IGNORING %u THING%s ON THE BOARD", n, n == 1 ? "" : "S" );
        }
        else if( g->last >= 0 ) {                       /* the latest card scanned: out of the memory */
            sort_name( g->card[ g->last ].word, name, sizeof name );
            unstore( g );
            say_idle( g );
            SAY( g, note, "UNDONE %s", name );
        }
    }
    g->acted_ok = 0;                                    /* look at the scene again */
    g->events++;
}
