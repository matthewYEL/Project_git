/*
 * M3 "suit sort": the rules, played from what the camera finds (app_config.h
 * SORT_GAME). Pure logic -- no FPGA, no RTOS -- so the host test runs it
 * unchanged; the camera comes in through two callbacks (a CNN read of a card,
 * the grey of a frame pixel) and hdmi_gui.c draws whatever state it leaves.
 *
 * The frame (512x384) is the board: 4 suit columns x SORT_ROWS rows of
 * 128 x 96 px bands. The band edges sit exactly under the guide lines
 * hdmi_gui.c draws over the live preview (one text cell = 16 x 32 frame px).
 * The lines are a guide: a placement is judged between the cards themselves
 * (a suit stacked in one column, higher ranks above, the suits' columns left
 * to right), so a camera off to one side or at another height judges the
 * same, and the board memory keeps every card in its own suit's column.
 *
 *   a turn  KEY0 reads the card held up under the camera (app_rtos.c, then
 *           sort_held); the game says which column and row it goes to (any
 *           free row between its neighbours: only the top-to-bottom order is
 *           checked) and keeps it in the board memory from then on. The
 *           camera keeps finding the cards on the board, so the card is read
 *           again where it is laid, confirmed (seen) and checked there; read
 *           there as another suit of its rank, the board's read replaces the
 *           held one. Only KEY0 puts a card in the memory: anything else the
 *           finder boxes on the board -- a card laid without KEY0, glare, a
 *           mark -- is read once, for a joker, then ignored while it lies there.
 *   alerts  the opponent's joker, anywhere -- in a gap, half over one of ours,
 *           exactly over one -- is pointed at until it is taken off.
 *
 * It acts only on a scene that has stood still for SORT_STILL card finds
 * (hands moving cards change it), and only once per scene, so a waiting
 * board costs no CNN reads. A box bigger than a slot is a card held up, not
 * one on the board: never read or judged. Cards hidden for a moment (a hand,
 * a held card) are only reported after SORT_HIDDEN_MS.
 */
#ifndef SORTGAME_H
#define SORTGAME_H

#include <stdint.h>

#include "app_config.h"
#include "card_pipeline.h"

#define SORT_COLS       4
#define SORT_MAX        ( SORT_COLS * SORT_ROWS )
#define SORT_BAND_W     128                                 /* frame px per suit column: 8 text cells */
#define SORT_BAND_H     ( FRAME_H / SORT_ROWS )             /* ... per row */
#define SORT_X0         8                                   /* the board's left edge (text col 5) */
#define SORT_SIG_N      16                                  /* a card's look: 4 x 4 cells */
#define SORT_GHOSTS     8
#define SORT_HIDDEN_MS  6000u                               /* ours out of sight this long: report it */

/* How a card is read (app_rtos.c): FULL = M2's 5-read vote + club look, FAST =
 * stop as soon as the first windows agree, VERIFY = one window (only to pick
 * between cards the game already knows). */
enum sort_read  { READ_FULL, READ_FAST, READ_VERIFY };

enum sort_phase { SP_SHOW, SP_PLACE, SP_ALERT };
enum sort_alert { AL_NONE, AL_JOKER, AL_CARD, AL_THING };
enum sort_tone  { TONE_OK, TONE_BUSY, TONE_WARN, TONE_ALERT };   /* hdmi_gui.c colours them */
enum sort_tick  { TICK_MOVING, TICK_SAME, TICK_EVENT };
enum sort_cmd   { SORT_CMD_NONE, SORT_CMD_READ, SORT_CMD_UNDO, SORT_CMD_NEW };

/* A CNN read of card c (enum sort_read): the cnn_result word (0 = no read),
 * how many windows it took and how many agreed. */
typedef uint32_t ( *sort_read_fn )( const struct scan_card * c, int how, unsigned * reads, unsigned * agree );
/* The grey (0..255) of frame pixel (x, y) -- valid between CNN runs only. */
typedef unsigned ( *sort_gray_fn )( int x, int y );

struct sort_card
{
    uint32_t         word;                  /* its cnn_result */
    struct scan_card box;                   /* where the camera found it (seen only) ... */
    uint8_t          sig[ SORT_SIG_N ];     /* ... and how it looked there */
    int8_t           col, row;              /* its slot in the board memory: its suit's column, the row planned */
    uint8_t          seen;                  /* the camera has found it on the board ... */
    uint8_t          bad;                   /* ... out of order with the cards around it */
};

struct sort_game
{
    struct sort_card card[ SORT_MAX ];      /* the board memory: every card scanned, in that order */
    unsigned         n;

    uint8_t          phase, resume;         /* resume: the phase an alert interrupted */
    uint8_t          fresh;                 /* a new game: wait for an empty board */

    int8_t           tcol, trow;            /* SP_PLACE: the slot of card[ last ], the card to place */
    int8_t           slide_a, slide_z;      /* ... rows of its column that slide to make room (as moved) ... */
    int8_t           slide_dir;             /* ... +1 down, -1 up, 0 none */
    char             phint[ 40 ];           /* ... and the hint that came with it */

    uint8_t          alert;                 /* enum sort_alert */
    struct scan_card abox;                  /* where the alert points */
    uint32_t         aword;                 /* what was read there (0 = nothing) */
    int8_t           acov;                  /* our card it lies over, -1 = none */

    struct scan_card prev[ SCAN_MAX ];      /* the last find: the stillness check, and the live view */
    unsigned         nprev, still;
    uint8_t          look[ SCAN_MAX ][ SORT_SIG_N ];    /* this find's boxes: how each looks */
    struct scan_card acted[ SCAN_MAX ];     /* the scene last acted on ... */
    uint8_t          alook[ SCAN_MAX ][ SORT_SIG_N ];   /* ... and its looks */
    unsigned         nacted;
    uint8_t          acted_ok;
    struct scan_card ghost[ SORT_GHOSTS ];  /* no cards of ours, not read again while they lie there ... */
    uint8_t          gsig[ SORT_GHOSTS ][ SORT_SIG_N ];  /* ... and how each looked (all 0: any look) */
    unsigned         nghost;

    unsigned         now;                   /* ms, from the caller's latest tick */
    uint8_t          hiding;                /* ours out of sight since hidden_at */
    unsigned         hidden_at;

    int8_t           last;                  /* card[] index of the latest card scanned, -1 = none */
    unsigned         turn, mistakes, jokers;
    unsigned         reads, ms;             /* the latest look's CNN reads; ms is set by the caller */
    unsigned         events;                /* bumped whenever the whole screen should be redrawn */

    char             prompt[ 40 ];          /* what to do now (HDMI prompt plate) */
    char             hint[ 40 ];            /* under it: neighbours, slides */
    char             note[ 40 ];            /* the latest event */
    char             pill[ 28 ];            /* the status pill ... */
    uint8_t          tone;                  /* ... and its enum sort_tone */
};

void sort_new( struct sort_game * g );

/* One card find (locate_cards' result) at time now_ms: may read cards through
 * `rd` (each a ~1.7 s CNN run) and look at pixels through `gray`. Returns enum
 * sort_tick. */
int  sort_tick( struct sort_game * g, const struct scan_result * s, unsigned now_ms, sort_read_fn rd, sort_gray_fn gray );

/* KEY0's read of the card held up (app_rtos.c): word, CNN reads, how many
 * agreed. Says where it goes, or why not. */
void sort_held( struct sort_game * g, uint32_t word, unsigned reads, unsigned agree );

/* Is this card (not a joker) on the board already? One deck: a second one is a misread. */
int  sort_on_board( const struct sort_game * g, uint32_t word );

/* KEY1 tap: dismiss an alert, else (a new game) ignore what lies on the
 * board, else take the latest card scanned out of the memory -- if it lies
 * on the board it is read again there. KEY1 held: a new game. */
void sort_command( struct sort_game * g, int cmd );

/* Where a box lies: its board column 0..3 and row 0..SORT_ROWS-1 */
void sort_slot( const struct scan_card * b, int * col, int * row );

/* Bigger than a slot: a card held up over the board, not one lying on it */
int  sort_in_air( const struct scan_card * b );

/* A box the game ignores: where a ghost lies (not a card of ours) */
int  sort_ignored( const struct sort_game * g, const struct scan_card * b );

/* The board column of a suit index (cnn_result: 0 spades, 1 clubs, 2 hearts,
 * 3 diamonds): spades, hearts, clubs, diamonds left to right */
int  sort_suit_col( unsigned suit );

/* "J\x06", "10\x03", "JOKER" -- CP437 suit glyphs, as HDMI prints them */
void sort_name( uint32_t word, char * buf, unsigned n );

/* A card's look: 4 x 4 cells over the middle of its box, each the mean of 4
 * pixels, scaled so the brightest is 255 */
void sort_sig( const struct scan_card * b, sort_gray_fn gray, uint8_t sig[ SORT_SIG_N ] );

/* How far two looks are apart (mean per cell), the closer of as-is and turned
 * 180 deg -- a card can lie either way up */
unsigned sort_sig_diff( const uint8_t a[ SORT_SIG_N ], const uint8_t b[ SORT_SIG_N ] );

#endif /* SORTGAME_H */
