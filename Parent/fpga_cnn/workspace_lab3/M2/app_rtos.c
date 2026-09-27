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
 *     prio 1  Report  the only task that prints: result line, ASCII preview
 *                     (SW0), PGM dump (SW1), run-time statistics (SW2, and every
 *                     STATS_EVERY results). Lowest priority on purpose:
 *                     semihosted printf is slow and nothing else waits on it.
 *                     Also the only task that draws the HDMI dashboard
 *                     (hdmi_gui.c) -- a cell is a 3-access sequence that must
 *                     not interleave -- and it draws before it prints.
 *
 * FPGA ownership: the PGM dump and the camera diagnostics read the image RAM
 * from CPU0, which is only safe with no job in flight. So Vision takes the next
 * job only after Report has finished with the current one (task notification).
 *
 * Nothing here touches CPU1 except through amp.c's req/done handshake --
 * FreeRTOS is not SMP-safe and CPU1 must never see a kernel object.
 */
#include <stdio.h>
#include <stdint.h>

#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"

#include "card_pipeline.h"
#include "hdmi_gui.h"
#include "amp.h"

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
};

/* Only REPORT_RESULT is answered with the notification Vision waits on. */
enum { REPORT_RESULT, REPORT_BUSY, REPORT_JOB_START, REPORT_UNDO, REPORT_CLEAR };

struct report_msg
{
    uint8_t                      kind;
    uint8_t                      on_core1;
    uint8_t                      auto_job;  /* shown on HDMI, not tracked */
    uint32_t                     shot;
    unsigned                     sw;
    const struct vision_result * res;   /* stays valid until Report notifies Vision */
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

/* A display-only message for Report. Never blocks the sender: with the queue
 * full, the message is dropped rather than stalling input or vision. */
static void prvPost( uint8_t ucKind, uint32_t ulShot, uint8_t ucOnCore1 )
{
    struct report_msg xMsg = { ucKind, ucOnCore1, 0, ulShot, 0, NULL };

    xQueueSend( xReports, &xMsg, 0 );
}

/* ---- tasks -------------------------------------------------------------- */

static void prvInputTask( void * pvParameters )
{
    uint32_t   ulShot      = 0;
    TickType_t xLastIssued = xTaskGetTickCount();
    TickType_t xKey1Down   = 0;
    int        iKey1Timing = 0;

    ( void ) pvParameters;

    for( ;; )
    {
        /* Presses are latched in button_pio's edge-capture register, so one
         * made while CPU0 was halted in a semihosted printf is still here. */
        unsigned uPressed = key_presses();
        unsigned uSw      = read_switches();
        int      iAuto    = ( uSw & SW_AUTO ) && !xJobInFlight &&
                            ( xTaskGetTickCount() - xLastIssued ) >= pdMS_TO_TICKS( AUTO_PERIOD_MS );

        /* KEY1: released before KEY1_CLEAR_MS = undo, still down then = clear.
         * The level is only sampled here, so a press already over by the time
         * it is seen (e.g. made during a printf halt) counts as a tap. */
        if( uPressed & KEY_UNDO )
        {
            iKey1Timing = 1;
            xKey1Down   = xTaskGetTickCount();
        }
        if( iKey1Timing && !( keys_held() & KEY_UNDO ) )
        {
            prvPost( REPORT_UNDO, 0, 0 );
            iKey1Timing = 0;
        }
        else if( iKey1Timing && ( xTaskGetTickCount() - xKey1Down ) >= pdMS_TO_TICKS( KEY1_CLEAR_MS ) )
        {
            prvPost( REPORT_CLEAR, 0, 0 );
            iKey1Timing = 0;
        }
        uPressed &= KEY_CAPTURE;

        if( uPressed && xJobInFlight )
        {
            prvPost( REPORT_BUSY, ulShot - 1u, 0 );
        }
        else if( uPressed || iAuto )
        {
            struct job_req xReq;

            xReq.shot     = ulShot++;
            xReq.sw       = uSw;
            xReq.auto_job = !uPressed;
            xLastIssued   = xTaskGetTickCount();
            xJobInFlight  = 1;
            xQueueSend( xJobs, &xReq, portMAX_DELAY );
        }

        vTaskDelay( pdMS_TO_TICKS( KEY_POLL_MS ) );
    }
}

static void prvVisionTask( void * pvParameters )
{
    ( void ) pvParameters;

    for( ;; )
    {
        struct job_req    xReq;
        struct report_msg xMsg;

        if( xQueueReceive( xJobs, &xReq, portMAX_DELAY ) != pdPASS )
        {
            continue;
        }

        xMsg.kind     = REPORT_RESULT;
        xMsg.shot     = xReq.shot;
        xMsg.sw       = xReq.sw;
        xMsg.res      = NULL;
        xMsg.on_core1 = 0;
        xMsg.auto_job = xReq.auto_job;

        if( ampCore1Running() && !xCore1Lost )
        {
            prvPost( REPORT_JOB_START, xReq.shot, 1 );
            ampSubmit();
            if( ampWait( CORE1_JOB_TIMEOUT_US ) )
            {
                xMsg.res      = &AMP->job.res;
                xMsg.on_core1 = 1;
                ulJobsCore1++;
            }
            else
            {
                /* Longer than any job can take (capture timeouts + the 5 s
                 * inference timeout): CPU1 is wedged, so stop giving it work
                 * rather than risk two cores driving the FPGA at once. */
                xCore1Lost = 1;
            }
        }

        if( xMsg.res == NULL )
        {
            /* vision_run polls without blocking, so Report only gets to show
             * this after the job -- fallback mode only. */
            prvPost( REPORT_JOB_START, xReq.shot, 0 );
            vision_run( &xLocalResult );
            xMsg.res = &xLocalResult;
            ulJobsCore0++;
        }

        xQueueSend( xReports, &xMsg, portMAX_DELAY );

        /* Hold the next job until Report is done: its PGM dump / camera
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
            continue;
        }
        if( xMsg.kind == REPORT_CLEAR )
        {
            gui_clear();
            continue;
        }

        pxRes = xMsg.res;

        /* HDMI first: it takes milliseconds, the semihosted prints below far longer */
        gui_job_done( xMsg.shot, pxRes, !xMsg.auto_job, xMsg.on_core1 ? "CPU1" : "CPU0" );

        if( ( xMsg.sw & SW_PREVIEW ) &&
            ( pxRes->status == VIS_OK || pxRes->status == VIS_INFER_TIMEOUT ) )
        {
            print_preview( pxRes->art );
        }

        print_vision_result( xMsg.shot, pxRes, xMsg.on_core1 ? "CPU1" : "CPU0" );

        /* only once the core is done: until then the image read port is conv1's */
        if( ( xMsg.sw & SW_PGM ) && pxRes->status == VIS_OK && !RES_DDR_ERR( pxRes->result ) )
        {
            gui_status( "PGM DUMP (console)", GUI_YELLOW );
            dump_pgm();
            gui_status( "READY", GUI_LGREEN );
        }

        ulResults++;
        if( ( xMsg.sw & SW_STATS )
#if STATS_EVERY
            || ( ( ulResults % STATS_EVERY ) == 0 )
#endif
          )
        {
            prvPrintStats();
        }

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

    /* Report's queue also carries the display-only messages (job start, KEY1
     * undo/clear), which are dropped rather than waited for when it is full --
     * 8 leaves room for a burst of presses during a long PGM dump. */
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
    printf( "snapshot mode -- press KEY0 to capture and classify; the HDMI dashboard\n"
            "tracks each card. KEY1 undoes the newest one (hold %u ms: clear them all).\n"
            "SW0 = ASCII preview, SW1 = PGM dump, SW2 = scheduling stats,\n"
            "SW3 = auto-capture every %u ms, shown but not tracked (SW0+SW3 = ASCII viewfinder).\n\n",
            ( unsigned ) KEY1_CLEAR_MS, ( unsigned ) AUTO_PERIOD_MS );

    vTaskStartScheduler();

    /* Only reached if the kernel could not allocate the idle or timer task. */
    printf( "\n*** scheduler returned -- out of heap ***\n" );
    for( ;; ) { }
}
#endif /* RTOS_MODE */
