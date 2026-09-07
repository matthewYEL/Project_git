/*
 * The FreeRTOS application: the card pipeline as three tasks on CPU0, with the
 * preprocessing stage handed to CPU1.
 *
 *   prio 3  Capture   trigger the camera, read the raw cells, hand the
 *                     frame to CPU1. Highest priority because the camera is
 *                     the only part with a real deadline -- miss a frame and
 *                     it is gone.
 *   prio 2  Infer     collect the preprocessed frame from CPU1, upload it,
 *                     start the CNN, poll for the result.
 *   prio 1  Report    printf. Lowest priority on purpose: semihosting putchar
 *                     is slow enough to dominate the loop, and it is the one
 *                     stage nothing else waits on.
 *
 * Two slots circulate between Capture and Infer through a pair of queues, so
 * Capture can already be reading frame N+1 out of the FPGA while CPU1
 * preprocesses frame N and Infer uploads frame N-1. Without the second core
 * those stages serialise; with it they overlap, which is what the frame times
 * in the report line show.
 *
 * Nothing here touches CPU1 except through amp.c's req/done handshake --
 * FreeRTOS is not SMP-safe and CPU1 must never see a kernel object.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"

#include "app_config.h"
#include "card_pipeline.h"
#include "amp.h"

extern void     vInstallVectorTable( void );
extern void     vInitialiseGIC( void );
extern void     vInitPeriphClock( void );
extern void     vRunTimeStatsTimerInit( void );
extern uint32_t ulGlobalTimerNow( void );
extern uint32_t ulGlobalTimerToUs( uint32_t ulTicks );
extern uint32_t ulGetPeriphClockHz( void );

#define CAPTURE_PRIO    3
#define INFER_PRIO      2
#define REPORT_PRIO     1

#define CAPTURE_STACK   ( configMINIMAL_STACK_SIZE * 2 )
#define INFER_STACK     ( configMINIMAL_STACK_SIZE * 2 )
#define REPORT_STACK    ( configMINIMAL_STACK_SIZE * 8 )   /* printf lives here */

struct frame_msg
{
    uint8_t  slot;
    uint32_t frame;
    uint32_t cap_us;
};

struct result_msg
{
    uint32_t frame;
    unsigned result;
    unsigned red_count;
    unsigned minv, maxv;
    uint32_t cap_us, pre_us, inf_us;
    uint8_t  preproc_on_core1;
};

static QueueHandle_t xFreeSlots;    /* slot indices Capture may write */
static QueueHandle_t xReadyFrames;  /* slots handed on to Infer       */
static QueueHandle_t xResults;      /* results handed on to Report    */

static volatile uint32_t ulCore1Frames;   /* frames preprocessed on CPU1 */
static volatile uint32_t ulCore0Frames;   /* frames preprocessed on CPU0 */

/* ---- tasks -------------------------------------------------------------- */

static void prvCaptureTask( void * pvParameters )
{
    uint32_t ulFrame = 0;

    ( void ) pvParameters;

    for( ;; )
    {
        struct frame_msg xMsg;
        struct amp_slot *pxSlot;
        uint8_t          ucSlot;
        uint32_t         ulStart;
        unsigned         minv, maxv;

        if( xQueueReceive( xFreeSlots, &ucSlot, portMAX_DELAY ) != pdPASS )
        {
            continue;
        }

        pxSlot  = &AMP->slot[ ucSlot ];
        ulStart = ulGlobalTimerNow();

        if( !capture_once() )
        {
            /* Camera not streaming. Give the slot back and back off rather
             * than spinning on a dead link. */
            xQueueSend( xFreeSlots, &ucSlot, 0 );
            vTaskDelay( pdMS_TO_TICKS( 100 ) );
            continue;
        }

        read_snapshot( pxSlot->raw, &minv, &maxv );
        pxSlot->minv      = minv;
        pxSlot->maxv      = maxv;
        pxSlot->red_count = snap_read( SNAP_RED_COUNT );

        if( ampCore1Running() )
        {
            ampSubmit( pxSlot );
        }

        xMsg.slot   = ucSlot;
        xMsg.frame  = ulFrame++;
        xMsg.cap_us = ulGlobalTimerToUs( ulGlobalTimerNow() - ulStart );

        xQueueSend( xReadyFrames, &xMsg, portMAX_DELAY );
    }
}

static void prvInferTask( void * pvParameters )
{
    ( void ) pvParameters;

    for( ;; )
    {
        struct frame_msg  xMsg;
        struct result_msg xRes;
        struct amp_slot  *pxSlot;
        uint32_t          ulStart;

        if( xQueueReceive( xReadyFrames, &xMsg, portMAX_DELAY ) != pdPASS )
        {
            continue;
        }

        pxSlot = &AMP->slot[ xMsg.slot ];

        /* Collect the preprocessed frame from CPU1. If it never comes -- CPU1
         * was never released, or has wedged -- do it here instead. The demo
         * degrades to single-core rather than stopping. */
        ulStart = ulGlobalTimerNow();

        if( ampCore1Running() && ampWait( pxSlot, CORE1_FRAME_TIMEOUT_US ) )
        {
            xRes.preproc_on_core1 = 1;
            xRes.pre_us = pxSlot->us;
            ulCore1Frames++;
        }
        else
        {
            preprocess( pxSlot->raw, pxSlot->q );
            xRes.preproc_on_core1 = 0;
            xRes.pre_us = ulGlobalTimerToUs( ulGlobalTimerNow() - ulStart );
            ulCore0Frames++;
        }

        ulStart = ulGlobalTimerNow();
        xRes.result = upload_and_infer( pxSlot->q, pxSlot->red_count );
        xRes.inf_us = ulGlobalTimerToUs( ulGlobalTimerNow() - ulStart );

        xRes.frame     = xMsg.frame;
        xRes.cap_us    = xMsg.cap_us;
        xRes.red_count = pxSlot->red_count;
        xRes.minv      = pxSlot->minv;
        xRes.maxv      = pxSlot->maxv;

        /* The slot is free the moment its pixels have been uploaded. */
        xQueueSend( xFreeSlots, &xMsg.slot, portMAX_DELAY );
        xQueueSend( xResults, &xRes, portMAX_DELAY );
    }
}

#if STATS_EVERY
static char cStatsBuffer[ 640 ];

static void prvPrintRunTimeStats( void )
{
    printf( "\n  FreeRTOS run-time stats (CPU0)\n"
            "  Task            Abs time     %%\n" );
    vTaskGetRunTimeStats( cStatsBuffer );
    fputs( cStatsBuffer, stdout );

    printf( "  CPU1 worker: %lu frames, %lu us busy",
            ( unsigned long ) AMP->core1_frames,
            ( unsigned long ) AMP->core1_busy_us );
    if( AMP->core1_frames )
    {
        printf( " (%lu us/frame)",
                ( unsigned long ) ( AMP->core1_busy_us / AMP->core1_frames ) );
    }
    printf( "\n  preprocessing: %lu frames on CPU1, %lu on CPU0\n\n",
            ( unsigned long ) ulCore1Frames, ( unsigned long ) ulCore0Frames );
}
#endif /* STATS_EVERY */

static void prvReportTask( void * pvParameters )
{
    ( void ) pvParameters;

    for( ;; )
    {
        struct result_msg xRes;

        if( xQueueReceive( xResults, &xRes, portMAX_DELAY ) != pdPASS )
        {
            continue;
        }

        printf( "[%5lu] ", ( unsigned long ) xRes.frame );

        if( xRes.result == 0 )
        {
            printf( "inference timeout\n" );
        }
        else
        {
            print_result_name( xRes.result );
            printf( "  logit %5d  red %5u->%-5s  min %4u max %4u"
                    "  | cap %4lu us  pre %4lu us on CPU%d  inf %5lu us",
                    RES_SCORE( xRes.result ), xRes.red_count,
                    RES_COLOUR( xRes.result ) ? "red" : "black",
                    xRes.minv, xRes.maxv,
                    ( unsigned long ) xRes.cap_us,
                    ( unsigned long ) xRes.pre_us,
                    xRes.preproc_on_core1 ? 1 : 0,
                    ( unsigned long ) xRes.inf_us );

            if( xRes.maxv == 0 )
            {
                printf( "   <- BLANK FRAME" );
            }
            printf( "\n" );
        }

#if STATS_EVERY
        if( ( ( xRes.frame + 1 ) % STATS_EVERY ) == 0 )
        {
            prvPrintRunTimeStats();
        }
#endif
    }
}

/* ---- entry -------------------------------------------------------------- */

int rtos_main( void )
{
    uint8_t ucSlot;

    printf( "\n=== DE10-Nano card CNN -- FreeRTOS on CPU0, worker on CPU1 ===\n" );

    /* The global timer is the timebase for the AMP handshake timeouts, so it
     * has to be running before CPU1 is released -- not just from the point the
     * scheduler starts it for the run-time stats. */
    vInitPeriphClock();
    vRunTimeStatsTimerInit();

    /* Exception vectors, VBAR and the exception-mode stacks; then the GIC.
     * None of this exists in the lab's bare-metal template. */
    vInstallVectorTable();
    vInitialiseGIC();
    printf( "  mpu_periph_clk %lu Hz\n",
            ( unsigned long ) ulGetPeriphClockHz() );

    check_camera_alive();

#if USE_CORE1
    printf( "  releasing CPU1 (trampoline at 0x0, entry via "
            "sysmgr.cpu1startaddr)...\n" );
    if( ampStartCore1() )
    {
        printf( "  CPU1 is up: preprocessing runs there, shared block at "
                "0x%08lX\n", ( unsigned long ) AMP_SHARED_BASE );
    }
    else
    {
        printf( "  ! CPU1 did not check in -- continuing on CPU0 alone.\n"
                "    Everything still works, just without the overlap.\n" );
    }
#else
    printf( "  USE_CORE1 is 0: single core.\n" );
#endif

    xFreeSlots   = xQueueCreate( AMP_SLOTS, sizeof( uint8_t ) );
    xReadyFrames = xQueueCreate( AMP_SLOTS, sizeof( struct frame_msg ) );
    xResults     = xQueueCreate( 4, sizeof( struct result_msg ) );

    if( ( xFreeSlots == NULL ) || ( xReadyFrames == NULL ) || ( xResults == NULL ) )
    {
        printf( "  ! could not create queues\n" );
        return 1;
    }

    for( ucSlot = 0; ucSlot < AMP_SLOTS; ucSlot++ )
    {
        xQueueSend( xFreeSlots, &ucSlot, 0 );
    }

    xTaskCreate( prvCaptureTask, "Capture", CAPTURE_STACK, NULL, CAPTURE_PRIO, NULL );
    xTaskCreate( prvInferTask,   "Infer",   INFER_STACK,   NULL, INFER_PRIO,   NULL );
    xTaskCreate( prvReportTask,  "Report",  REPORT_STACK,  NULL, REPORT_PRIO,  NULL );

    printf( "  tasks: Capture(p%d) -> Infer(p%d) -> Report(p%d), %d frame slots\n",
            CAPTURE_PRIO, INFER_PRIO, REPORT_PRIO, AMP_SLOTS );
    printf( "  starting scheduler\n\n" );

    vTaskStartScheduler();

    /* Only reached if the kernel could not allocate the idle or timer task. */
    printf( "\n*** scheduler returned -- out of heap ***\n" );
    for( ;; ) { }
}
