/* The M2 card grid -- see grid.h. */
#include <string.h>

#include "card_pipeline.h"
#include "grid.h"

int grid_rank_key( uint32_t result )
{
    return RES_JOKER( result ) ? 13 : ( int ) RES_RANK( result );
}

void grid_new( struct grid * g, int n, int mode )
{
    int i;

    memset( g, 0, sizeof *g );
    g->n      = ( uint8_t ) ( n < 3 ? 3 : n > GRID_MAX ? GRID_MAX : n );
    g->mode   = ( uint8_t ) mode;
    g->target = mode == GRID_SCAN ? 0 : -1;     /* a scan starts at A1; a game waits for a cell */
    g->last   = -1;
    g->turn   = 1;
    g->players = 2;
    g->open[ 0 ] = g->open[ 1 ] = -1;
    for( i = 0; i < g->n * g->n; i++ )
        g->cell[ i ].state = mode == GRID_SCAN ? CELL_EMPTY : CELL_DOWN;
}

/* Scan: number the pairs in cell order. The first two cards of a rank are a
 * pair, the next two another, and an odd one out stays unpaired. */
static void repair( struct grid * g )
{
    int pending[ 14 ], i, k = 0;

    for( i = 0; i < 14; i++ ) pending[ i ] = -1;
    for( i = 0; i < g->n * g->n; i++ ) {
        struct grid_cell * c = &g->cell[ i ];
        int r;

        c->pair = 0;
        if( c->state != CELL_OPEN ) continue;
        r = grid_rank_key( c->result );
        if( pending[ r ] < 0 ) {
            pending[ r ] = i;
            continue;
        }
        g->cell[ pending[ r ] ].pair = c->pair = ( uint8_t ) ++k;
        pending[ r ] = -1;
    }
    g->pairs = ( uint8_t ) k;
}

/* the next empty cell after `from` in cell order, wrapping; -1 if none */
static int next_empty( const struct grid * g, int from )
{
    int k, cells = g->n * g->n;

    for( k = 1; k <= cells; k++ ) {
        int i = ( from + k ) % cells;
        if( g->cell[ i ].state == CELL_EMPTY ) return i;
    }
    return -1;
}

/* game: a judged no-match pair goes back face down when the next turn starts */
static void end_turn( struct grid * g )
{
    int k;

    if( g->n_open < 2 ) return;
    for( k = 0; k < 2; k++ )
        if( g->cell[ g->open[ k ] ].state == CELL_OPEN ) g->cell[ g->open[ k ] ].state = CELL_DOWN;
    g->open[ 0 ] = g->open[ 1 ] = -1;
    g->n_open = 0;
}

int grid_read( struct grid * g, uint32_t result, unsigned shot )
{
    struct grid_cell * c;
    int before;

    if( g->mode == GRID_GAME && g->target >= 0 ) end_turn( g );
    if( g->target < 0 ) return GRID_NO_CELL;
    c = &g->cell[ g->target ];

    if( g->mode == GRID_SCAN ) {
        if( c->state == CELL_EMPTY ) g->n_read++;
        c->state  = CELL_OPEN;
        c->result = result;
        c->shot   = ( uint16_t ) shot;
        g->last   = g->target;
        before    = g->pairs;
        repair( g );
        g->target = ( int8_t ) next_empty( g, g->last );
        return c->pair && g->pairs > before ? GRID_PAIRED : GRID_PLACED;
    }

    if( c->state != CELL_DOWN ) return GRID_BAD_CELL;
    c->state  = CELL_OPEN;
    c->result = result;
    c->shot   = ( uint16_t ) shot;
    g->last   = g->target;
    g->open[ g->n_open++ ] = g->target;
    g->target = -1;
    if( g->n_open == 1 ) return GRID_OPENED;

    if( grid_rank_key( g->cell[ g->open[ 0 ] ].result ) == grid_rank_key( result ) ) {
        g->cell[ g->open[ 0 ] ].state = c->state = CELL_MATCHED;
        g->score[ g->team ]++;
        g->pairs++;
        g->outcome = OUT_MATCH;
    }
    else
        g->outcome = OUT_NO_MATCH;
    g->turn++;
    if( g->players == 2 ) g->team ^= 1u;    /* the spec: teams take alternate turns */
    return GRID_JUDGED;
}

int grid_target( struct grid * g, int cell )
{
    if( cell < 0 || cell >= g->n * g->n ) return 0;
    if( g->mode == GRID_GAME ) {
        end_turn( g );
        if( g->cell[ cell ].state != CELL_DOWN ) return 0;
    }
    g->target = ( int8_t ) cell;
    return 1;
}

int grid_left( const struct grid * g )
{
    int i, n = 0;

    for( i = 0; i < g->n * g->n; i++ ) n += g->cell[ i ].state != CELL_MATCHED;
    return n;
}

int grid_parse_cell( const struct grid * g, const char * s )
{
    int col, row;

    if( !s[ 0 ] || !s[ 1 ] || s[ 2 ] ) return -1;
    col = ( s[ 0 ] | 0x20 ) - 'a';                  /* either case */
    row = s[ 1 ] - '1';
    if( col < 0 || col >= g->n || row < 0 || row >= g->n ) return -1;
    return row * g->n + col;
}

void grid_cell_name( const struct grid * g, int cell, char * buf )
{
    buf[ 0 ] = ( char ) ( 'A' + cell % g->n );
    buf[ 1 ] = ( char ) ( '1' + cell / g->n );
    buf[ 2 ] = '\0';
}
