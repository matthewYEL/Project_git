/*
 * The M2 card grid: an N x N board (N = 3, 4, 5) and the rules of the rubric
 * rounds played on it. Pure logic -- no I/O, no FPGA -- so the host test runs
 * it unchanged, and hdmi_gui.c draws whatever state it leaves.
 *
 *   GRID_SCAN  the face-up rounds: every card is read once, in cell order
 *              (A1, B1, C1, A2, ... -- letters are columns, numbers rows), and
 *              cards of the same rank pair up.
 *   GRID_GAME  the face-down round: two cards per turn, teams alternate, and a
 *              same-rank pair is matched and scores for the team whose turn it is.
 *
 * Positions come from the operator (a guided scan): the display names the next
 * cell, or the operator types one ("b3" on the UART), and the next read goes
 * there. A pair is two cards of the same rank; two jokers pair too.
 */
#ifndef GRID_H
#define GRID_H

#include <stdint.h>

#define GRID_MAX    5
#define GRID_CELLS  ( GRID_MAX * GRID_MAX )

enum grid_mode  { GRID_SCAN, GRID_GAME };
enum cell_state { CELL_EMPTY, CELL_DOWN, CELL_OPEN, CELL_MATCHED };
enum grid_outcome { OUT_NONE, OUT_MATCH, OUT_NO_MATCH };

/* What a read did; the display turns it into the status line. The first four
 * changed the grid, the rejections left it untouched. */
enum grid_event
{
    GRID_PLACED,        /* scan: the card went into its cell                  */
    GRID_PAIRED,        /* scan: ... and completed a new pair                 */
    GRID_OPENED,        /* game: the first card of the turn is up             */
    GRID_JUDGED,        /* game: the second is up; see outcome                */
    GRID_NO_CELL,       /* rejected: no cell chosen (game), or the scan is full */
    GRID_BAD_CELL       /* rejected: that cell is matched or already open     */
};

struct grid_cell
{
    uint8_t  state;     /* enum cell_state */
    uint8_t  pair;      /* scan: pair number from 1, 0 = unpaired */
    uint16_t shot;
    uint32_t result;    /* raw cnn_result word of the card read there */
};

struct grid
{
    uint8_t  n, mode;
    int8_t   target;    /* the cell the next read goes to; -1 = none */
    int8_t   last;      /* the cell of the latest read; -1 = none */
    uint8_t  n_read;    /* scan: cells filled */
    uint8_t  pairs;     /* scan: pairs among them; game: pairs matched */
    uint8_t  team;      /* game: whose turn, 0 = A, 1 = B */
    uint8_t  players;   /* game: 2 = teams A/B alternate (grid_new), 1 = one player */
    uint8_t  turn;      /* game: from 1 */
    uint8_t  score[ 2 ];
    int8_t   open[ 2 ]; /* game: the cells turned up this turn */
    uint8_t  n_open;
    uint8_t  outcome;   /* game: the last judged turn, enum grid_outcome */
    struct grid_cell cell[ GRID_CELLS ];
};

void grid_new( struct grid * g, int n, int mode );

/* Record a read in the target cell. Returns an enum grid_event. */
int  grid_read( struct grid * g, uint32_t result, unsigned shot );

/* Make `cell` the target of the next read: in a scan any cell (a filled one is
 * read again, to correct it), in a game only a face-down one. Starting a game
 * turn also turns the previous no-match pair face down. 0 if refused. */
int  grid_target( struct grid * g, int cell );

/* the cells not yet matched (game) */
int  grid_left( const struct grid * g );

/* 0..12 by rank, 13 for a joker: what "same rank" compares */
int  grid_rank_key( uint32_t result );

/* "b3" / "B3" -> cell index for this board, -1 if it is not a cell of it */
int  grid_parse_cell( const struct grid * g, const char * s );

/* cell index -> "B3" (buf holds 3 bytes) */
void grid_cell_name( const struct grid * g, int cell, char * buf );

#endif /* GRID_H */
