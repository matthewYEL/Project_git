/* Compiled only in the FreeRTOS build (RTOS_MODE in app_config.h). */
#include "app_config.h"
#if RTOS_MODE
/*
 * The FreeRTOS application for the 384x384 snapshot build: FreeRTOS on CPU0,
 * the vision job on CPU1.
 *
 *   CPU1  (bare metal, amp.c)  runs vision_run(): trigger the camera, wait for
 *                              the frame, sample it, start the CNN, poll the
 *                              ~1.7 s inference, decode. It owns the camera/CNN
 *                              PIOs while a job is in flight.
 *
 *   CPU0  (FreeRTOS, 3 tasks):
 *     prio 3  Input   every 10 ms reads the KEY presses latched in button_pio's
 *                     edge-capture register and issues one job per KEY0 press
 *                     (or one every AUTO_PERIOD_MS with SW3 up). A press while a
 *                     job is running is answered at once with "busy" -- CPU0
 *                     stays responsive while CPU1 works. KEY1 edits the HDMI
 *                     tracker: a tap undoes the newest card, holding it
 *                     KEY1_CLEAR_MS clears them all.
 *     prio 2  Vision  hands the job to CPU1 and sleeps until it is done (or runs
 *                     it on CPU0 if CPU1 never came up / stops answering).
 *     prio 1  Report  the only task that prints: result line, the picture
 *                     kept for Arm DS (SW1, pics.h),
 *                     run-time statistics (every
 *                     STATS_EVERY results), the SW2 face-down game.
 *                     Lowest priority on purpose:
 *                     semihosted printf is slow and nothing else waits on it.
 *                     Also the only task that draws the HDMI dashboard
 *                     (hdmi_gui.c) -- a cell is a 3-access sequence that must
 *                     not interleave -- and it draws before it prints.
 *
 * FPGA ownership: the SW1 picture copy and the camera diagnostics read the image RAM
 * from CPU0, which is only safe with no job in flight. So Vision takes the next
 * job only after Report has finished with the current one (task notification).
 *
 * Nothing here touches CPU1 except through amp.c's req/done handshake --
 * FreeRTOS is not SMP-safe and CPU1 must never see a kernel object.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"

#include "card_pipeline.h"
#include "hdmi_gui.h"
#include "amp.h"
#include "locate.h"
#include "pics.h"
#include "grid.h"

extern void     vInstallVectorTable( void );
extern void     vInitialiseGIC( void );
extern void     vInitPeriphClock( void );
extern uint32_t ulGetPeriphClockHz( void );

#define INPUT_PRIO      3
#define VISION_PRIO     2
#define REPORT_PRIO     1

#define INPUT_STACK     ( configMINIMAL_STACK_SIZE * 2 )
#define VISION_STACK    ( configMINIMAL_STACK_SIZE * 4 )
#define REPORT_STACK    ( configMINIMAL_STACK_SIZE * 8 )   /* printf lives here */

#define KEY_POLL_MS     10

struct job_req
{
    uint32_t shot;
    unsigned sw;
    uint8_t  auto_job;                  /* issued by SW3's timer, not a KEY0 press */
    uint8_t  scan;                      /* a whole-grid scan, not one guided read */
};

/* Only REPORT_RESULT is answered with the notification Vision waits on. */
enum { REPORT_RESULT, REPORT_BUSY, REPORT_JOB_START, REPORT_UNDO, REPORT_CLEAR, REPORT_CMD,
       REPORT_SCAN_FOUND, REPORT_SCAN_CARD, REPORT_SCAN_DONE, REPORT_SCAN_FAIL,
       REPORT_GAME_SW,                          /* SW2 moved: shot = 1 game on, 0 off */
       REPORT_TURN,                             /* SW0 moved: shot = 1 camera turned, 0 not */
       REPORT_GAME_BOARD,                       /* game: the snapshot that sets the board up */
       REPORT_GAME_TURN,                        /* game: xScan cards idx and aux are the two up */
       REPORT_GAME_NOTE };                      /* game: idx = GAME_BAD_GRID / GAME_NOT_TWO, aux = count */
enum { GAME_BAD_GRID = 1, GAME_NOT_TWO };
/* A scan answers with the notification Vision waits on at SCAN_DONE / SCAN_FAIL. */

#define CMD_MAX         8                   /* an operator command line, NUL included */

struct report_msg
{
    uint8_t                      kind;
    uint8_t                      on_core1;
    uint8_t                      auto_job;  /* shown on HDMI, not tracked */
    uint32_t                     shot;
    unsigned                     sw;
    const struct vision_result * res;   /* stays valid until Report notifies Vision */
    char                         cmd[ CMD_MAX ];    /* REPORT_CMD: the typed line */
    uint8_t                      idx;       /* REPORT_SCAN_CARD: which xScan card */
    uint32_t                     aux;       /* SCAN_FOUND: locate us; SCAN_FAIL: status */
};

static QueueHandle_t xJobs;         /* Input  -> Vision, depth 1 */
static QueueHandle_t xReports;      /* Input/Vision -> Report    */
static TaskHandle_t  xVisionTask;

/* Set by Input when it issues a job, cleared by Vision once Report is done with
 * it. One writer each side, so a plain volatile is enough. */
static volatile int  xJobInFlight;
static volatile int  xCore1Lost;    /* CPU1 stopped answering: run jobs on CPU0 */

static struct vision_result xLocalResult;   /* CPU0 fallback target */

static uint32_t ulJobsCore1, ulJobsCore0;

/* ---- whole-grid scan state ------------------------------------------------
 * UART m toggles what KEY0 does (one guided read / scan the whole grid) and
 * UART o cycles how the camera sits over the table; Report owns both (it
 * handles the commands), Input and Vision only read them. */
static volatile int      xAutoScan = SCAN_AUTO_DEFAULT;
static volatile unsigned uOrientIdx = SCAN_ORIENT_DEFAULT;

static const struct { unsigned bits; const char * name; } ORIENTS[] = {
    { 0,                             "upright" },
    { ORIENT_SWAP | ORIENT_FLIP_R,   "cam CW" },       /* camera turned clockwise */
    { ORIENT_SWAP | ORIENT_FLIP_C,   "cam CCW" },      /* ... anticlockwise */
    { ORIENT_FLIP_R | ORIENT_FLIP_C, "cam 180" },
};
#define N_ORIENTS   ( sizeof ORIENTS / sizeof ORIENTS[ 0 ] )

/* SW0 in this app: the camera is turned 90 deg clockwise, ORIENTS[ 1 ] -- for a
 * 4x4, whose four rows then run along the frame's 512 px, not its 384 (cards
 * ~82 px wide instead of ~61). Up, it overrides UART o. SW0's old job, the
 * ASCII preview, went out over the UART, which the demo no longer has (30 Sep). */
#define SW_TURN         SW_PREVIEW
#define ORIENT_TURNED   1u

/* The orientation the next scan uses: SW0 up = turned, else the o setting */
static unsigned prvOrient( void )
{
    return ( read_switches() & SW_TURN ) ? ORIENT_TURNED : uOrientIdx;
}

/* ... and the one the running scan uses, from SW0 as it was at KEY0: set by
 * prvScan before it locates, read by prvReadCard and by Report's header line */
static volatile unsigned uScanOrient = SCAN_ORIENT_DEFAULT;

/* The scan in progress: written by Vision, read by Report through the card
 * index in each message. Vision starts the next job only after Report has
 * answered SCAN_DONE, so neither changes under the other. */
static struct scan_result xScan;

/* The face-down game (SW2): written by Report only, read by Vision between
 * jobs (Vision takes the next job only after Report's answer). uGameN is the
 * board's n, 0 = none yet -- the next KEY0 sets it up; ulGameMatched holds the
 * found pairs' cells, bit (row-1)*GRID_MAX + col-1: they stay face up where
 * they lie, and the camera skips them. */
static volatile unsigned uGameN;
static volatile uint32_t ulGameMatched;
static uint32_t           xScanWord[ SCAN_MAX ];    /* each card's final read */
static uint8_t            xScanReads[ SCAN_MAX ];   /* windows read: SCAN_READS, all 7 for a duplicate */
static uint8_t            xScanAgree[ SCAN_MAX ];   /* of them, how many gave the answer */

/* conv1's default window: the old centred 384x384 crop, 1:1 -- what guided
 * reads use (ghrd_top.v's reset values) */
static const struct win_regs xGuideWin = {
    GUIDE_X0, 0, 256, 0, { 0, 0, FRAME_W - 1, FRAME_H - 1 }
};

/* A display-only message for Report. Never blocks the sender: with the queue
 * full, the message is dropped rather than stalling input or vision. */
static void prvPost( uint8_t ucKind, uint32_t ulShot, uint8_t ucOnCore1 )
{
    struct report_msg xMsg = { ucKind, ucOnCore1, 0, ulShot, 0, NULL, { 0 } };

    xQueueSend( xReports, &xMsg, 0 );
}

/* Gather the operator's typed characters (UART) into a line; Enter posts it to
 * Report, which owns the grid and the display. Backspace edits; anything past
 * CMD_MAX - 1 characters is dropped. Returns 1 for "s": a scan, which is a job
 * and so Input's to issue. */
static int prvPollCommand( char * pcLine, unsigned * puLen )
{
    int iCh, iScan = 0;

    while( ( iCh = uart_getc() ) >= 0 )
    {
        if( iCh == '\r' || iCh == '\n' )
        {
            if( *puLen == 1u && ( pcLine[ 0 ] | 0x20 ) == 's' )
            {
                iScan  = 1;
                *puLen = 0;
            }
            else if( *puLen )
            {
                struct report_msg xMsg = { REPORT_CMD, 0, 0, 0, 0, NULL, { 0 } };

                memcpy( xMsg.cmd, pcLine, *puLen );
                xQueueSend( xReports, &xMsg, 0 );
                *puLen = 0;
            }
        }
        else if( ( iCh == 0x08 || iCh == 0x7F ) && *puLen )
        {
            ( *puLen )--;
        }
        else if( iCh > ' ' && iCh < 0x7F && *puLen < CMD_MAX - 1u )
        {
            pcLine[ ( *puLen )++ ] = ( char ) iCh;
        }
    }
    return iScan;
}

/* ---- tasks -------------------------------------------------------------- */

static void prvInputTask( void * pvParameters )
{
    uint32_t   ulShot      = 0;
    TickType_t xLastIssued = xTaskGetTickCount();
    TickType_t xKey1Down   = 0;
    int        iKey1Timing = 0;
    char       acLine[ CMD_MAX ];
    unsigned   uLineLen    = 0;
    unsigned   uLastGame   = ~0u;           /* SW2 as last shown; ~0 = not yet */
    unsigned   uLastTurn   = ~0u;           /* SW0 as last shown; ~0 = not yet */

    ( void ) pvParameters;

    for( ;; )
    {
        int iScanCmd = prvPollCommand( acLine, &uLineLen );

        /* Presses are latched in button_pio's edge-capture register, so one
         * made while CPU0 was halted in a semihosted printf is still here. */
        unsigned uPressed = key_presses();
        unsigned uSw      = read_switches();
        int      iAuto    = ( uSw & SW_AUTO ) && !xJobInFlight &&
                            ( xTaskGetTickCount() - xLastIssued ) >= pdMS_TO_TICKS( AUTO_PERIOD_MS );

        /* SW2: the face-down game on or off -- Report starts a new game each
         * time it goes up (the text layer's one writer). Only marked shown once
         * the message is queued, so a full queue just means another try on the
         * next poll. */
        if( ( uSw & SW_GAME ) != uLastGame )
        {
            struct report_msg xMsg = { REPORT_GAME_SW, 0, 0, ( uSw & SW_GAME ) ? 1u : 0u, 0, NULL, { 0 }, 0, 0 };

            if( xQueueSend( xReports, &xMsg, 0 ) == pdPASS )
            {
                uLastGame = uSw & SW_GAME;
            }
        }

        /* SW0: the camera turned or not, shown in the HDMI hint the same way */
        if( ( uSw & SW_TURN ) != uLastTurn )
        {
            struct report_msg xMsg = { REPORT_TURN, 0, 0, ( uSw & SW_TURN ) ? 1u : 0u, 0, NULL, { 0 }, 0, 0 };

            if( xQueueSend( xReports, &xMsg, 0 ) == pdPASS )
            {
                uLastTurn = uSw & SW_TURN;
            }
        }

        /* KEY1: released before KEY1_CLEAR_MS = undo, still down then = clear.
         * The level is only sampled here, so a press already over by the time
         * it is seen (e.g. made during a printf halt) counts as a tap. While a
         * job runs the board belongs to it: KEY1 only says busy, so an undo
         * can never swap boards under a scan in progress. */
        if( uPressed & KEY_UNDO )
        {
            iKey1Timing = 1;
            xKey1Down   = xTaskGetTickCount();
        }
        if( iKey1Timing && !( keys_held() & KEY_UNDO ) )
        {
            if( xJobInFlight ) prvPost( REPORT_BUSY, ulShot - 1u, 0 );
            else               prvPost( REPORT_UNDO, 0, 0 );
            iKey1Timing = 0;
        }
        else if( iKey1Timing && ( xTaskGetTickCount() - xKey1Down ) >= pdMS_TO_TICKS( KEY1_CLEAR_MS ) )
        {
            if( xJobInFlight ) prvPost( REPORT_BUSY, ulShot - 1u, 0 );
            else               prvPost( REPORT_CLEAR, 0, 0 );
            iKey1Timing = 0;
        }
        uPressed &= KEY_CAPTURE;

        if( ( uPressed || iScanCmd ) && xJobInFlight )
        {
            prvPost( REPORT_BUSY, ulShot - 1u, 0 );
        }
        else if( uPressed || iScanCmd || iAuto )
        {
            struct job_req xReq;

            xReq.shot     = ulShot++;
            xReq.sw       = uSw;
            xReq.auto_job = !uPressed && !iScanCmd;
            /* UART s always scans; KEY0 scans in AUTO mode; SW3's timer never */
            xReq.scan     = ( uint8_t ) ( iScanCmd || ( uPressed && xAutoScan ) );
            xLastIssued   = xTaskGetTickCount();
            xJobInFlight  = 1;
            xQueueSend( xJobs, &xReq, portMAX_DELAY );
        }

        vTaskDelay( pdMS_TO_TICKS( KEY_POLL_MS ) );
    }
}

/* Run one job on CPU1 -- or on CPU0 if CPU1 never came up or has stopped
 * answering -- and return where its result is. The result stays valid until
 * the next job. */
static const struct vision_result * prvJob( const struct vision_job * pxCmd, uint8_t * pucOnCore1 )
{
    if( ampCore1Running() && !xCore1Lost )
    {
        AMP->job.cmd = *pxCmd;
        ampCacheClean( &AMP->job.cmd, sizeof( AMP->job.cmd ) );
        ampSubmit();
        if( ampWait( CORE1_JOB_TIMEOUT_US ) )
        {
            ulJobsCore1++;
            *pucOnCore1 = 1;
            return &AMP->job.res;
        }
        /* Longer than any job can take (capture timeouts + the 5 s inference
         * timeout): CPU1 is wedged, so stop giving it work rather than risk
         * two cores driving the FPGA at once. */
        xCore1Lost = 1;
    }
    /* vision_do polls without blocking, so Report only gets to show progress
     * after the job -- fallback mode only. */
    vision_do( pxCmd, &xLocalResult );
    ulJobsCore0++;
    *pucOnCore1 = 0;
    return &xLocalResult;
}

/* ---- whole-grid scan ------------------------------------------------------ */

static void prvPostScan( uint8_t ucKind, const struct job_req * pxReq, uint8_t ucIdx, uint32_t ulAux )
{
    struct report_msg xMsg = { ucKind, 0, 0, pxReq->shot, pxReq->sw, NULL, { 0 }, ucIdx, ulAux };

    xQueueSend( xReports, &xMsg, portMAX_DELAY );      /* never dropped: Report must see them all */
}

/* One CNN read through `w` with the colour forced. 0 = no result. */
static uint32_t prvInferAt( const struct win_regs * w, int iRed )
{
    struct vision_job              xCmd = { VJOB_INFER, 0, ( uint32_t ) iRed };
    const struct vision_result *   pxRes;
    uint8_t                        ucOn1;

    win_write( w );                         /* CPU0: the bus is ours */
    pxRes = prvJob( &xCmd, &ucOn1 );
    return pxRes->status == VIS_OK && !RES_DDR_ERR( pxRes->result ) ? pxRes->result : 0u;
}

/* What a read says, for voting: rank and suit, 0x40 for a joker, 0xFF for none. */
static unsigned prvIdentity( uint32_t r )
{
    if( !RES_DONE( r ) || RES_DDR_ERR( r ) || ( !RES_JOKER( r ) && RES_RANK( r ) >= 13u ) ) return 0xFFu;
    return RES_JOKER( r ) ? 0x40u : ( RES_RANK( r ) << 2 ) | RES_SUIT( r );
}

/* The vote's windows: each a slightly different look at the same stored frame
 * (turned, shifted, zoomed), so a borderline card is read several ways. The
 * first SCAN_READS are used; a card whose identity turns up twice in the grid
 * (impossible with one deck) is read with all of them. The 29 Sep photo test,
 * 243 card reads: 1 read 237, the old 3-read rule on J/Q/K/A 237 (it fixed kings
 * but flipped aces), 5 reads on every card 241. sim_card_cnn.VARIANTS is the
 * same table. */
static const struct { int8_t dx, dy; uint16_t zoom; uint8_t flip; } VARIANTS[] = {
    {  0,  0, 1000, 0 },                    /* as is */
    {  0,  0, 1000, WIN_FX | WIN_FY },      /* turned 180 deg */
    {  3,  0, 1000, 0 },                    /* shifted x + 3 px */
    {  0, -3, 1000, WIN_FX | WIN_FY },      /* shifted y - 3 px, turned */
    {  0,  0, 1050, 0 },                    /* zoomed out 5 % */
    { -3,  0, 1000, WIN_FX | WIN_FY },      /* duplicates only: shifted x - 3 px, turned */
    {  0,  0,  950, 0 },                    /* duplicates only: zoomed in 5 % */
};
#define N_VARIANTS  ( sizeof VARIANTS / sizeof VARIANTS[ 0 ] )

/* One card, read through the first uReads windows. The most frequent identity
 * wins, ties to the higher summed rank logit, then to the earlier read; the
 * return is that identity's highest-logit read. *pucAgree = reads behind it. */
static uint32_t prvReadCard( const struct scan_card * c, unsigned uReads, uint8_t * pucReads, uint8_t * pucAgree )
{
    struct win_regs w, v;
    uint32_t        r[ N_VARIANTS ];
    unsigned        id[ N_VARIANTS ], k, j, best = 0, best_n = 0;
    long            best_s = 0;

    if( uReads < 1u ) uReads = 1u;
    if( uReads > N_VARIANTS ) uReads = N_VARIANTS;
    window_for_card( c, ORIENTS[ uScanOrient ].bits, FRAME_W, FRAME_H, &w );
    for( k = 0; k < uReads; k++ )
    {
        v = w;
        if( VARIANTS[ k ].zoom != 1000u )       /* rescale, centred on the card again */
        {
            int step = ( ( int ) w.step * VARIANTS[ k ].zoom + 500 ) / 1000, span;

            step   = step < 1 ? 1 : step > 511 ? 511 : step;
            span   = ( ( 384 - 1 ) * step ) >> 8;
            v.step = ( uint16_t ) step;
            v.x0   = ( int16_t ) ( ( c->x0 + c->x1 - span + 1 ) >> 1 );
            v.y0   = ( int16_t ) ( ( c->y0 + c->y1 - span + 1 ) >> 1 );
        }
        v.x0    = ( int16_t ) ( v.x0 + VARIANTS[ k ].dx );
        v.y0    = ( int16_t ) ( v.y0 + VARIANTS[ k ].dy );
        v.flags = ( uint16_t ) ( v.flags ^ VARIANTS[ k ].flip );
        r[ k ]  = prvInferAt( &v, c->red );
        id[ k ] = prvIdentity( r[ k ] );
    }

    for( k = 0; k < uReads; k++ )               /* each identity once, at its first read */
    {
        unsigned n = 0;
        long     s = 0;

        if( id[ k ] == 0xFFu ) continue;
        for( j = 0; j < k && id[ j ] != id[ k ]; j++ ) ;
        if( j < k ) continue;
        for( j = k; j < uReads; j++ )
        {
            if( id[ j ] == id[ k ] )
            {
                n++;
                s += RES_SCORE( r[ j ] );
            }
        }
        if( n > best_n || ( n == best_n && s > best_s ) )
        {
            best   = k;
            best_n = n;
            best_s = s;
        }
    }
    *pucReads = ( uint8_t ) uReads;
    *pucAgree = ( uint8_t ) best_n;
    if( !best_n ) return r[ 0 ];
    for( k = best, j = best + 1; j < uReads; j++ )
    {
        if( id[ j ] == id[ best ] && RES_SCORE( r[ j ] ) > RES_SCORE( r[ k ] ) ) k = j;
    }
    return r[ k ];
}

/* Capture, find the cards, read each one; Report shows them as they come.
 * Report answers the final SCAN_DONE / SCAN_FAIL with the notification. */
static void prvScan( const struct job_req * pxReq )
{
    struct vision_job              xCmd = { VJOB_CAPTURE, 0, 0 };
    const struct vision_result *   pxRes;
    uint8_t                        ucOn1;
    unsigned                       i, j;

    pxRes = prvJob( &xCmd, &ucOn1 );
    if( pxRes->status != VIS_OK || pxRes->maxv == 0u )
    {
        prvPostScan( REPORT_SCAN_FAIL, pxReq, 0, pxRes->status == VIS_OK ? 0xFFu : pxRes->status );
        return;
    }

    uScanOrient = ( pxReq->sw & SW_TURN ) ? ORIENT_TURNED : uOrientIdx;
    xCmd.kind   = VJOB_LOCATE;
    xCmd.orient = ORIENTS[ uScanOrient ].bits;
    pxRes = prvJob( &xCmd, &ucOn1 );
    xScan = pxRes->scan;
    prvPostScan( REPORT_SCAN_FOUND, pxReq, 0, pxRes->inf_us );

    for( i = 0; i < xScan.n; i++ )
    {
        xScanWord[ i ] = prvReadCard( &xScan.card[ i ], SCAN_READS, &xScanReads[ i ], &xScanAgree[ i ] );
        prvPostScan( REPORT_SCAN_CARD, pxReq, ( uint8_t ) i, 0 );
    }

    /* One deck: the same card twice is a misread. Read both again the careful
     * way, unless they already were (a re-read of the same windows would only
     * repeat itself). Two jokers are fine. */
    for( i = 0; i < xScan.n; i++ )
    {
        for( j = i + 1; j < xScan.n; j++ )
        {
            unsigned id = prvIdentity( xScanWord[ i ] ), k;

            if( id == 0xFFu || id == 0x40u || id != prvIdentity( xScanWord[ j ] ) ) continue;
            for( k = 0; k < 2; k++ )
            {
                unsigned m = k ? j : i;

                if( xScanReads[ m ] >= N_VARIANTS ) continue;
                xScanWord[ m ] = prvReadCard( &xScan.card[ m ], N_VARIANTS, &xScanReads[ m ], &xScanAgree[ m ] );
                prvPostScan( REPORT_SCAN_CARD, pxReq, ( uint8_t ) m, 1 );
            }
        }
    }

    win_write( &xGuideWin );                /* guided reads use the old crop again */
    prvPostScan( REPORT_SCAN_DONE, pxReq, 0, 0 );
}

/* The face-down game (SW2), one KEY0: one snapshot and the cards found. The
 * first one of a game only sets the board up (Report checks every card is face
 * down); after that the two cards turned up -- face up, and not on a found
 * pair's cell -- are read (the same 5-read vote) and Report judges them. Report
 * answers the final GAME_* / SCAN_FAIL message with the notification. */
static void prvGameTurn( const struct job_req * pxReq )
{
    struct vision_job              xCmd = { VJOB_CAPTURE, 0, 0 };
    const struct vision_result *   pxRes;
    uint8_t                        ucOn1;
    unsigned                       i, n = uGameN, nUp = 0, aUp[ 2 ] = { 0, 0 };

    pxRes = prvJob( &xCmd, &ucOn1 );
    if( pxRes->status != VIS_OK || pxRes->maxv == 0u )
    {
        prvPostScan( REPORT_SCAN_FAIL, pxReq, 0, pxRes->status == VIS_OK ? 0xFFu : pxRes->status );
        return;
    }
    uScanOrient = ( pxReq->sw & SW_TURN ) ? ORIENT_TURNED : uOrientIdx;
    xCmd.kind   = VJOB_LOCATE;
    xCmd.orient = ORIENTS[ uScanOrient ].bits;
    pxRes = prvJob( &xCmd, &ucOn1 );
    xScan = pxRes->scan;

    if( !n )
    {
        prvPostScan( REPORT_GAME_BOARD, pxReq, 0, 0 );
        return;
    }
    if( xScan.nrows != n || xScan.ncols != n || xScan.n != n * n )
    {
        prvPostScan( REPORT_GAME_NOTE, pxReq, GAME_BAD_GRID, xScan.n );
        return;
    }
    for( i = 0; i < xScan.n; i++ )          /* in (row, col) order */
    {
        const struct scan_card * c = &xScan.card[ i ];

        if( !c->up || ( ( ulGameMatched >> ( ( c->row - 1u ) * GRID_MAX + c->col - 1u ) ) & 1u ) ) continue;
        if( nUp < 2 ) aUp[ nUp ] = i;
        nUp++;
    }
    if( nUp != 2 )
    {
        prvPostScan( REPORT_GAME_NOTE, pxReq, GAME_NOT_TWO, nUp );
        return;
    }
    for( i = 0; i < 2; i++ )
    {
        xScanWord[ aUp[ i ] ] = prvReadCard( &xScan.card[ aUp[ i ] ], SCAN_READS,
                                             &xScanReads[ aUp[ i ] ], &xScanAgree[ aUp[ i ] ] );
    }
    win_write( &xGuideWin );                /* guided reads use the old crop again */
    prvPostScan( REPORT_GAME_TURN, pxReq, ( uint8_t ) aUp[ 0 ], aUp[ 1 ] );
}

static void prvVisionTask( void * pvParameters )
{
    ( void ) pvParameters;

    for( ;; )
    {
        struct job_req    xReq;

        if( xQueueReceive( xJobs, &xReq, portMAX_DELAY ) != pdPASS )
        {
            continue;
        }

        prvPost( REPORT_JOB_START, xReq.shot, ( uint8_t ) ( ampCore1Running() && !xCore1Lost ) );
        if( xReq.scan && ( xReq.sw & SW_GAME ) )
        {
            prvGameTurn( &xReq );
        }
        else if( xReq.scan )
        {
            prvScan( &xReq );
        }
        else
        {
            struct vision_job xCmd = { VJOB_FULL, 0, 0 };
            struct report_msg xMsg = { REPORT_RESULT, 0, xReq.auto_job, xReq.shot, xReq.sw, NULL, { 0 }, 0, 0 };

            win_write( &xGuideWin );        /* in case a scan was cut short */
            xMsg.res = prvJob( &xCmd, &xMsg.on_core1 );
            xQueueSend( xReports, &xMsg, portMAX_DELAY );
        }

        /* Hold the next job until Report is done: its picture copy / camera
         * diagnostics read the FPGA from CPU0, and it reads the result block. */
        ulTaskNotifyTake( pdTRUE, portMAX_DELAY );
        xJobInFlight = 0;
    }
}

static char cStatsBuffer[ 640 ];

/* The scheduling evidence: per-task CPU0 time straight out of the kernel, and
 * how much work CPU1 has done. */
static void prvPrintStats( void )
{
    printf( "\n  FreeRTOS run-time stats (CPU0)\n"
            "  Task            Abs time     %%\n" );
    vTaskGetRunTimeStats( cStatsBuffer );
    fputs( cStatsBuffer, stdout );
    printf( "  CPU1 vision worker: %lu jobs, %lu ms busy",
            ( unsigned long ) AMP->core1_jobs,
            ( unsigned long ) ( AMP->core1_busy_us / 1000UL ) );
    if( AMP->core1_jobs )
    {
        printf( " (%lu ms/job)",
                ( unsigned long ) ( AMP->core1_busy_us / 1000UL / AMP->core1_jobs ) );
    }
    printf( "\n  jobs run on CPU1: %lu, on CPU0 (fallback): %lu%s\n\n",
            ( unsigned long ) ulJobsCore1, ( unsigned long ) ulJobsCore0,
            xCore1Lost ? "   ! CPU1 stopped answering" : "" );
}

static void prvPrintHelp( void )
{
    printf( "  commands (type, then Enter):\n"
            "    s             scan the whole grid: find every card, read it, list the pairs\n"
            "    m             KEY0: whole-grid scan <-> one card at the green box (now: %s)\n"
            "    o             next camera orientation (now: %s)\n"
            "    3 / 4 / 5     new guided face-up scan of that size\n"
            "    g3 / g4 / g5  new face-down game (round 3): teams alternate\n"
            "    a1 .. e5      the cell the next guided read goes to (column letter, row number)\n"
            "    r             start this board again\n"
            "  KEY1 tap undoes the last scan (or guided read), hold %u ms for a new board.\n",
            xAutoScan ? "scan" : "one card", ORIENTS[ prvOrient() ].name, ( unsigned ) KEY1_CLEAR_MS );
}

/* The scan's pairs, grouped by rank in reading order: "7: 7H (1,1) + 7D (2,2)".
 * Three of a kind is listed as a group of three rather than a pair plus a
 * silently dropped card. Same rank pairs; two jokers pair. */
static void prvPrintPairs( void )
{
    static const char SUIT_LETTER[ 4 ] = { 'S', 'C', 'H', 'D' };
    unsigned i, j, groups = 0, pairs = 0;
    uint8_t  done[ SCAN_MAX ] = { 0 };

    printf( "  pairs (same rank, as (row,col)):\n" );
    for( i = 0; i < xScan.n; i++ )
    {
        unsigned id = prvIdentity( xScanWord[ i ] ), key, cnt = 0;

        if( done[ i ] || id == 0xFFu ) continue;
        key = RES_JOKER( xScanWord[ i ] ) ? 13u : RES_RANK( xScanWord[ i ] );
        for( j = i; j < xScan.n; j++ )
        {
            uint32_t r = xScanWord[ j ];
            if( prvIdentity( r ) != 0xFFu && ( RES_JOKER( r ) ? 13u : RES_RANK( r ) ) == key ) cnt++;
        }
        if( cnt < 2u ) continue;
        groups++;
        pairs += cnt / 2u;
        printf( "    %-5s", key == 13u ? "JOKER" : RANK_NAMES[ key ] );
        for( j = i; j < xScan.n; j++ )
        {
            uint32_t r = xScanWord[ j ];
            if( prvIdentity( r ) == 0xFFu || ( RES_JOKER( r ) ? 13u : RES_RANK( r ) ) != key ) continue;
            done[ j ] = 1;
            if( key == 13u ) printf( " %sJK (%u,%u)", j == i ? "" : "+ ", xScan.card[ j ].row, xScan.card[ j ].col );
            else printf( " %s%s%c (%u,%u)", j == i ? "" : "+ ", RANK_NAMES[ key ], SUIT_LETTER[ RES_SUIT( r ) ],
                         xScan.card[ j ].row, xScan.card[ j ].col );
        }
        if( cnt > 2u ) printf( "   [%u of a kind]", cnt );
        printf( "\n" );
    }
    if( !groups ) printf( "    none\n" );
    printf( "  %u pair%s found\n\n", pairs, pairs == 1u ? "" : "s" );
}

/* The face-down game's answer to one KEY0 (prvGameTurn), on HDMI and the UART:
 * the board set up, the turn judged, or what was wrong with the snapshot. */
static void prvGameReport( const struct report_msg * pxMsg )
{
    char     msg[ 28 ];
    unsigned i, ups = 0, n = xScan.nrows > xScan.ncols ? xScan.nrows : xScan.ncols;

    if( pxMsg->kind == REPORT_GAME_BOARD )
    {
        for( i = 0; i < xScan.n; i++ ) ups += xScan.card[ i ].up;
        if( xScan.nrows != xScan.ncols || n < 3 || n > GRID_MAX || xScan.n != n * n )
        {
            snprintf( msg, sizeof msg, "SAW %lux%lu, %lu CARDS", ( unsigned long ) xScan.nrows,
                      ( unsigned long ) xScan.ncols, ( unsigned long ) xScan.n );
        }
        else if( ups )
        {
            snprintf( msg, sizeof msg, "%u UP - TURN ALL DOWN", ups );
        }
        else
        {
            gui_game_begin( ( int ) n );
            uGameN        = n;
            ulGameMatched = 0;
            printf( "\n[%3lu] face-down game %ux%u set up: turn 2 cards up, KEY0\n",
                    ( unsigned long ) pxMsg->shot, n, n );
            return;
        }
        gui_game_note( msg );
        printf( "\n[%3lu] face-down game: %s -- need a full 3x3, 4x4 or 5x5, all face down\n",
                ( unsigned long ) pxMsg->shot, msg );
        return;
    }
    if( pxMsg->kind == REPORT_GAME_NOTE )
    {
        if( pxMsg->idx == GAME_BAD_GRID )
            snprintf( msg, sizeof msg, "SAW %lu CARDS, NOT %ux%u", ( unsigned long ) pxMsg->aux, uGameN, uGameN );
        else
            snprintf( msg, sizeof msg, "SHOW 2 CARDS - SAW %lu", ( unsigned long ) pxMsg->aux );
        gui_game_note( msg );
        printf( "\n[%3lu] game: %s\n", ( unsigned long ) pxMsg->shot, msg );
        return;
    }
    {                                               /* REPORT_GAME_TURN */
        const struct scan_card * a = &xScan.card[ pxMsg->idx ], * b = &xScan.card[ pxMsg->aux ];
        uint32_t                 wa = xScanWord[ pxMsg->idx ], wb = xScanWord[ pxMsg->aux ];
        int                      out = gui_game_turn( a->row, a->col, wa, b->row, b->col, wb, pxMsg->shot );

        ulGameMatched = gui_game_matched();
        printf( "\n[%3lu] game turn: (%u,%u) ", ( unsigned long ) pxMsg->shot, a->row, a->col );
        if( prvIdentity( wa ) == 0xFFu ) printf( "no read" ); else print_result_name( wa );
        printf( " + (%u,%u) ", b->row, b->col );
        if( prvIdentity( wb ) == 0xFFu ) printf( "no read" ); else print_result_name( wb );
        printf( "  -> %s\n", out == OUT_MATCH ? "PAIR" : out == OUT_NO_MATCH ? "NOT A PAIR, turn them back"
                                                  : "refused: not face-down cells" );
    }
}

/* SW1: keep the frame and the reads in DDR for Arm DS to save (pics.h) -- a
 * 1-2 s copy, where the UART dump took 70 s and needed the cable. Red once the
 * ring has wrapped: the oldest picture is gone, save the ring sooner. */
static void prvSavePicture( const struct scan_result * pxScan, const uint32_t * pulWords )
{
    char     msg[ 28 ];
    unsigned n;

    gui_status( "SAVING PICTURE", GUI_YELLOW );
    n = pic_store( pxScan, pulWords );
    snprintf( msg, sizeof msg, n > PIC_SLOTS ? "PIC %u: OLDEST LOST" : "PIC %u SAVED", n );
    gui_status( msg, n > PIC_SLOTS ? GUI_LRED : GUI_LGREEN );
    printf( "  picture %u kept in memory for Arm DS (slot %u of %u, pics.h)\n\n",
            n, ( n - 1u ) % PIC_SLOTS + 1u, ( unsigned ) PIC_SLOTS );
}

static void prvReportTask( void * pvParameters )
{
    uint32_t ulResults = 0;

    ( void ) pvParameters;

    for( ;; )
    {
        struct report_msg            xMsg;
        const struct vision_result * pxRes;

        if( xQueueReceive( xReports, &xMsg, portMAX_DELAY ) != pdPASS )
        {
            continue;
        }

        if( xMsg.kind == REPORT_BUSY )
        {
            printf( "       busy: CPU%d is still on shot %lu -- press ignored\n",
                    ( ampCore1Running() && !xCore1Lost ) ? 1 : 0,
                    ( unsigned long ) xMsg.shot );
            continue;
        }
        if( xMsg.kind == REPORT_JOB_START )
        {
            char acStatus[ 24 ];

            snprintf( acStatus, sizeof acStatus, "READING #%lu (CPU%d)",
                      ( unsigned long ) xMsg.shot, xMsg.on_core1 ? 1 : 0 );
            gui_status( acStatus, GUI_YELLOW );
            continue;
        }
        if( xMsg.kind == REPORT_UNDO )
        {
            gui_undo();
            uGameN        = ( unsigned ) gui_game_n();      /* the game as the board now shows it */
            ulGameMatched = gui_game_matched();
            continue;
        }
        if( xMsg.kind == REPORT_GAME_SW )           /* SW2: every time it goes up, a new game */
        {
            uGameN        = 0;
            ulGameMatched = 0;
            if( xMsg.shot ) gui_game_note( "GAME: ALL FACE DOWN, KEY0" );
            else            gui_status( "READY", GUI_LGREEN );
            continue;
        }
        if( xMsg.kind == REPORT_GAME_BOARD || xMsg.kind == REPORT_GAME_TURN || xMsg.kind == REPORT_GAME_NOTE )
        {
            prvGameReport( &xMsg );
            if( xMsg.sw & SW_PGM )                  /* the snapshot too, e.g. to check the backs */
            {
                prvSavePicture( &xScan, xScanWord );
            }
            xTaskNotifyGive( xVisionTask );
            continue;
        }
        if( xMsg.kind == REPORT_TURN )              /* SW0: the hint names the orientation */
        {
            gui_mode( xAutoScan, ORIENTS[ xMsg.shot ? ORIENT_TURNED : uOrientIdx ].name );
            continue;
        }
        if( xMsg.kind == REPORT_CLEAR )
        {
            gui_clear();                            /* a game: all face down again, same board */
            uGameN        = ( unsigned ) gui_game_n();
            ulGameMatched = 0;
            continue;
        }
        if( xMsg.kind == REPORT_CMD )
        {
            printf( "> %s\n", xMsg.cmd );
            if( ( xMsg.cmd[ 0 ] | 0x20 ) == 'm' && !xMsg.cmd[ 1 ] )
            {
                xAutoScan = !xAutoScan;
                printf( "  KEY0 now %s\n", xAutoScan ? "SCANS THE WHOLE GRID" : "reads the card at the green box" );
                gui_mode( xAutoScan, ORIENTS[ prvOrient() ].name );
            }
            else if( ( xMsg.cmd[ 0 ] | 0x20 ) == 'o' && !xMsg.cmd[ 1 ] )
            {
                uOrientIdx = ( uOrientIdx + 1u ) % N_ORIENTS;
                printf( "  camera orientation: %s (row 1 = the top row as the professor sees it)%s\n",
                        ORIENTS[ uOrientIdx ].name, ( read_switches() & SW_TURN ) ? " -- SW0 up overrides it" : "" );
                gui_mode( xAutoScan, ORIENTS[ prvOrient() ].name );
            }
            else if( !gui_command( xMsg.cmd ) )
            {
                prvPrintHelp();
            }
            continue;
        }
        if( xMsg.kind == REPORT_SCAN_FOUND )
        {
            gui_scan_begin( &xScan );
            printf( "\n[%3lu] grid scan: %lu card%s in %lu row%s x %lu column%s  (threshold %lu, %lu ms, %s)\n",
                    ( unsigned long ) xMsg.shot, ( unsigned long ) xScan.n, xScan.n == 1 ? "" : "s",
                    ( unsigned long ) xScan.nrows, xScan.nrows == 1 ? "" : "s",
                    ( unsigned long ) xScan.ncols, xScan.ncols == 1 ? "" : "s",
                    ( unsigned long ) xScan.threshold, ( unsigned long ) ( xMsg.aux / 1000UL ),
                    ORIENTS[ uScanOrient ].name );
            continue;
        }
        if( xMsg.kind == REPORT_SCAN_CARD )
        {
            const struct scan_card * c = &xScan.card[ xMsg.idx ];
            uint32_t                 r = xScanWord[ xMsg.idx ];

            gui_scan_card( c->row, c->col, r, xMsg.shot );
            printf( "  (%u,%u)  ", c->row, c->col );
            if( prvIdentity( r ) == 0xFFu ) printf( "no read" );
            else                            print_result_name( r );
            printf( "  logit %6d  %-5s  [%u reads, %u agree]%s\n", ( int ) RES_SCORE( r ), c->red ? "red" : "black",
                    xScanReads[ xMsg.idx ], xScanAgree[ xMsg.idx ], xMsg.aux ? "  (re-read: duplicate)" : "" );
            continue;
        }
        if( xMsg.kind == REPORT_SCAN_DONE || xMsg.kind == REPORT_SCAN_FAIL )
        {
            if( xMsg.kind == REPORT_SCAN_FAIL )
            {
                gui_status( xMsg.aux == 0xFFu ? "BLANK FRAME" : "CAMERA FAILED", GUI_LRED );
                printf( "\n[%3lu] grid scan failed: %s\n", ( unsigned long ) xMsg.shot,
                        xMsg.aux == 0xFFu ? "blank frame" : "no camera frame" );
            }
            else
            {
                gui_scan_done();
                prvPrintPairs();
                /* the frame the scan read, for sim_card_cnn.py -- the core is idle now */
                if( xMsg.sw & SW_PGM )
                {
                    prvSavePicture( &xScan, xScanWord );
                }
            }
            xTaskNotifyGive( xVisionTask );
            continue;
        }

        pxRes = xMsg.res;

        /* HDMI first: it takes milliseconds, the semihosted prints below far longer */
        gui_job_done( xMsg.shot, pxRes, !xMsg.auto_job, xMsg.on_core1 ? "CPU1" : "CPU0" );

        print_vision_result( xMsg.shot, pxRes, xMsg.on_core1 ? "CPU1" : "CPU0" );

        /* only once the core is done: until then the image read port is conv1's */
        if( ( xMsg.sw & SW_PGM ) && pxRes->status == VIS_OK && !RES_DDR_ERR( pxRes->result ) )
        {
            prvSavePicture( NULL, NULL );
        }

        ulResults++;
#if STATS_EVERY
        if( ( ulResults % STATS_EVERY ) == 0 )      /* SW2 is the face-down game now */
        {
            prvPrintStats();
        }
#endif

        xTaskNotifyGive( xVisionTask );
    }
}

/* ---- entry -------------------------------------------------------------- */

int rtos_main( void )
{
    printf( "\n=== DE10-Nano card CNN -- FreeRTOS on CPU0, vision worker on CPU1 ===\n" );

    /* main() has already started the global timer (the timebase for the AMP
     * handshake timeouts), loaded the DDR3 weights and checked the camera. */
    vInitPeriphClock();

    /* Exception vectors, VBAR and the exception-mode stacks; then the GIC.
     * None of this exists in the lab's bare-metal template. */
    vInstallVectorTable();
    vInitialiseGIC();
    printf( "  mpu_periph_clk %lu Hz\n", ( unsigned long ) ulGetPeriphClockHz() );

#if USE_CORE1
    printf( "  releasing CPU1 (trampoline at 0x0, entry via sysmgr.cpu1startaddr)...\n" );
    if( ampStartCore1() )
    {
        printf( "  CPU1 is up: vision worker (capture + CNN control), shared block at "
                "0x%08lX\n", ( unsigned long ) AMP_SHARED_BASE );
    }
    else
    {
        printf( "  ! CPU1 did not check in -- jobs run on CPU0 instead.\n" );
    }
#else
    printf( "  USE_CORE1 is 0: jobs run on CPU0.\n" );
#endif

    /* The HDMI dashboard, drawn once here while nothing else runs; after the
     * scheduler starts only Report touches it. */
    gui_init();
    gui_mode( xAutoScan, ORIENTS[ prvOrient() ].name );

    /* Report's queue also carries the display-only messages (job start, KEY1
     * undo/clear), which are dropped rather than waited for when it is full --
     * 8 leaves room for a burst of presses during a picture copy. */
    xJobs    = xQueueCreate( 1, sizeof( struct job_req ) );
    xReports = xQueueCreate( 8, sizeof( struct report_msg ) );

    if( ( xJobs == NULL ) || ( xReports == NULL ) )
    {
        printf( "  ! could not create queues\n" );
        return 1;
    }

    xTaskCreate( prvInputTask,  "Input",  INPUT_STACK,  NULL, INPUT_PRIO,  NULL );
    xTaskCreate( prvVisionTask, "Vision", VISION_STACK, NULL, VISION_PRIO, &xVisionTask );
    xTaskCreate( prvReportTask, "Report", REPORT_STACK, NULL, REPORT_PRIO, NULL );

    printf( "  tasks: Input(p%d) -> Vision(p%d) -> Report(p%d)\n",
            INPUT_PRIO, VISION_PRIO, REPORT_PRIO );
    printf( "%s", xAutoScan
            ? "KEY0 scans the whole grid: every card inside the side lines (UART m: one card\n"
              "at the green box instead).\n"
            : "grid mode -- the HDMI names the next cell: show that card at the green box\n"
              "and press KEY0. Starts as a face-up 3x3 scan.\n" );
    printf( "SW0 = camera turned 90 deg clockwise (4x4: row 1 at the right of the preview),\n"
            "SW1 = keep each picture in memory for Arm DS (pics.h), SW2 = face-down game\n"
            "(KEY0 once all face down, then 2 cards up per KEY0), SW3 = auto-capture every\n"
            "%u ms, shown but never placed.\n",
            ( unsigned ) AUTO_PERIOD_MS );
    prvPrintHelp();
    printf( "\n" );

    vTaskStartScheduler();

    /* Only reached if the kernel could not allocate the idle or timer task. */
    printf( "\n*** scheduler returned -- out of heap ***\n" );
    for( ;; ) { }
}
#endif /* RTOS_MODE */
