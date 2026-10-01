/*
 * HDMI dashboard: the live camera preview (top left, drawn by the fabric) plus
 * a panel of detected cards, written into the fabric's 80x30 text layer
 * (text_overlay.v). The layout is all here, so it changes without a recompile.
 *
 * ONE WRITER. Every call writes the text layer with a 3-access sequence that
 * must not interleave with another, so only one task may call these: the
 * Report task in the RTOS build (app_rtos.c), main() in the single-core loop.
 * CPU0 only -- CPU1 never touches the GUI.
 */
#ifndef HDMI_GUI_H
#define HDMI_GUI_H

#include <stdint.h>

struct vision_result;

/* The text layer's 16-colour CGA palette, in text_overlay.v's order. */
enum gui_colour
{
    GUI_BLACK, GUI_BLUE, GUI_GREEN, GUI_CYAN, GUI_RED, GUI_MAGENTA, GUI_BROWN, GUI_LGREY,
    GUI_DGREY, GUI_LBLUE, GUI_LGREEN, GUI_LCYAN, GUI_LRED, GUI_LMAGENTA, GUI_YELLOW, GUI_WHITE
};

/* Clear the layer and draw the frame: title bar, preview border, empty panels. */
void gui_init( void );

/* The title bar's status field, e.g. ("READY", GUI_LGREEN). */
void gui_status( const char * msg, enum gui_colour colour );

/* A finished job. When `track` is set a real card goes into the grid's target
 * cell (grid.c); an untracked one (SW3 auto) is only shown. Camera/CNN/DDR3
 * failures and blank frames only set the status. */
void gui_job_done( unsigned shot, const struct vision_result * v, int track, const char * where );

/* KEY1: undo the last read (tap) / start the same board again (held). */
void gui_undo( void );
void gui_clear( void );

/* An operator command line (UART): "3".."5" a new face-up scan of that size,
 * "g3".."g5" a new face-down game, "r" the same board again, "a1".."e5" the
 * cell the next read goes to. 0 if it is none of these. */
int gui_command( const char * cmd );

/* ---- whole-grid scan (M2 demo) --------------------------------------------
 * The board then shows the grid the camera found, labelled by the professor's
 * (row, col); pairs are numbered by grid.c as in a guided scan. */
struct scan_result;
struct win_regs;
void gui_scan_begin( const struct scan_result * s );
void gui_scan_card( unsigned row, unsigned col, uint32_t result, unsigned shot );
void gui_scan_done( void );

/* The hint under the camera: what KEY0 does now, and the camera orientation. */
void gui_mode( int auto_scan, const char * orient_name );

/* ---- face-down game, camera-driven (SW2, 30 Sep) ---------------------------
 * gui_game_begin: a new one-player n x n game (the camera saw n x n, all face
 * down). gui_game_turn: the two cards turned up, by the professor's (row, col),
 * judged by grid.c; returns enum grid_outcome, OUT_NONE if a cell was not face
 * down. gui_game_note: a problem with the snapshot, on the status line.
 * gui_game_n: the game's n, 0 if the board is not a camera game (after an
 * undo, say). gui_game_matched: matched cells as bits (row-1)*GRID_MAX+(col-1),
 * which the camera then skips (found pairs stay face up). */
void     gui_game_begin( int n );
int      gui_game_turn( unsigned r1, unsigned c1, uint32_t w1, unsigned r2, unsigned c2, uint32_t w2, unsigned shot );
void     gui_game_note( const char * msg );
int      gui_game_n( void );
uint32_t gui_game_matched( void );

/* conv1's window registers (ghrd_top.v, img_wr_addr 0xE00 + n). They share
 * the text layer's bus, so like the text they are written from CPU0 only; the
 * bus is locked against the other CPU0 task for each 3-access write. */
void win_write( const struct win_regs * w );

#endif /* HDMI_GUI_H */
