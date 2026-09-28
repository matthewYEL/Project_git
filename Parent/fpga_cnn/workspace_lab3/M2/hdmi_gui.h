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

#endif /* HDMI_GUI_H */
