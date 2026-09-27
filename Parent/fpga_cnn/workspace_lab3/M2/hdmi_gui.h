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

/* A finished job. A real card goes to LAST CARD and, when `track` is set, onto
 * the board (the last 5) and into the deck grid; camera/CNN/DDR3 failures and
 * blank frames only set the status. `where` ("CPU1"/"CPU0" or NULL) is shown
 * with the card. */
void gui_job_done( unsigned shot, const struct vision_result * v, int track, const char * where );

/* Drop the newest tracked card (KEY1 tap) / all of them (KEY1 held). */
void gui_undo( void );
void gui_clear( void );

#endif /* HDMI_GUI_H */
