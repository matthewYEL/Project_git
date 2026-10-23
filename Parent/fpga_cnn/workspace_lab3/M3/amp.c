/* Compiled only in the FreeRTOS build (RTOS_MODE in app_config.h). */
#include "app_config.h"
#if RTOS_MODE
/*
 * Releasing and driving the second Cortex-A9. See amp.h for the shape of it.
 *
 * How CPU1 actually starts, which is not obvious and is easy to get wrong:
 *
 *   1. CPU1 is held in reset by rstmgr.mpumodrst bit 1 after the preloader
 *      runs. Nothing else has touched it.
 *   2. When that bit is cleared, CPU1 does NOT begin at the boot ROM and does
 *      not read any "start address" register on its own. It fetches from
 *      physical address 0. This is the part people trip over.
 *   3. So a trampoline is written to address 0 first. It reads the entry point
 *      out of sysmgr.romcodegrp.cpu1startaddr and branches to it -- the same
 *      two-step Linux uses in arch/arm/mach-socfpga/headsmp.S.
 *
 * Writing over address 0 is safe here: it holds u-boot's SPL vector table, and
 * SPL has finished by the time the application runs. The application's own
 * exception vectors are found through VBAR (a9_startup.c), not through address
 * 0, so nothing else reads what is being overwritten.
 */
#include <stdio.h>
#include <stdint.h>

#include "FreeRTOS.h"
#include "task.h"

#include "amp.h"

#define REG32( a ) ( *( volatile uint32_t * ) ( uintptr_t ) ( a ) )

#define GICD_BASE           0xFFFED000UL
#define GICD_SGIR           REG32( GICD_BASE + 0xF00 )
#define GICD_IPRIORITY8     ( ( volatile uint8_t * ) ( GICD_BASE + 0x400 ) )

extern uint32_t ulGlobalTimerNow( void );
extern uint32_t ulGlobalTimerToUs( uint32_t ulTicks );

/* AMP_CORE1_STACK_TOP has to appear as a literal inside a naked function. */
#define AMP_STR2( x ) #x
#define AMP_STR( x )  AMP_STR2( x )

/* The shared block and CPU1's stack are both placed by hand at absolute
 * addresses, so nothing but this check stands between a grown struct and CPU1
 * quietly scribbling over the frame buffers it is supposed to be reading. */
typedef char amp_shared_fits[ ( AMP_SHARED_BASE + ( int ) sizeof( struct amp_shared )
                                <= AMP_CORE1_STACK_TOP - 0x8000 ) ? 1 : -1 ];

static inline void dmb( void ) { __asm__ volatile ( "dmb" ::: "memory" ); }
static inline void dsb( void ) { __asm__ volatile ( "dsb" ::: "memory" ); }

/* ---- cache maintenance -------------------------------------------------
 * Clean/invalidate by MVA to the point of coherency, 32-byte lines. If the
 * MMU is off these are architecturally harmless -- data accesses are strongly
 * ordered and never cached -- so this costs a few cycles and removes the need
 * to know which regime the preloader left behind. */
void ampCacheClean( const void * pvAddr, uint32_t ulBytes )
{
    uint32_t ulAddr = ( uint32_t ) ( uintptr_t ) pvAddr & ~31UL;
    uint32_t ulEnd  = ( ( uint32_t ) ( uintptr_t ) pvAddr + ulBytes + 31UL ) & ~31UL;

    for( ; ulAddr < ulEnd; ulAddr += 32UL )
    {
        __asm__ volatile ( "mcr p15, 0, %0, c7, c10, 1" :: "r" ( ulAddr ) : "memory" );
    }
    dsb();
}

void ampCacheInvalidate( const void * pvAddr, uint32_t ulBytes )
{
    uint32_t ulAddr = ( uint32_t ) ( uintptr_t ) pvAddr & ~31UL;
    uint32_t ulEnd  = ( ( uint32_t ) ( uintptr_t ) pvAddr + ulBytes + 31UL ) & ~31UL;

    /* Clean *and* invalidate: a plain invalidate would discard anything this
     * core had legitimately dirtied in the same line. */
    for( ; ulAddr < ulEnd; ulAddr += 32UL )
    {
        __asm__ volatile ( "mcr p15, 0, %0, c7, c14, 1" :: "r" ( ulAddr ) : "memory" );
    }
    dsb();
}

/* ---- CPU1 side ---------------------------------------------------------- */

/* The worker. No RTOS, no interrupts, no printf. While a job is in flight it
 * owns the camera/CNN PIOs; CPU0 touches them only between jobs (app_rtos.c's
 * Vision task holds the next job until the Report task is done with this one).
 * The only other thing it shares with CPU0 is the req/done handshake. */
static void ampCore1Main( void )
{
    struct amp_shared * pxShared = AMP;
    struct amp_job    * pxJob    = &pxShared->job;

    pxShared->core1_jobs    = 0;
    pxShared->core1_busy_us = 0;
    dmb();
    pxShared->core1_alive = AMP_MAGIC;
    dsb();

    while( pxShared->stop == 0UL )
    {
        uint32_t ulReq;
        uint32_t ulStart;
        uint32_t ulUs;

        ampCacheInvalidate( ( const void * ) pxJob, AMP_JOB_HEAD_BYTES );
        ulReq = pxJob->req;

        if( ulReq == pxJob->done )
        {
            /* Nothing to do: sleep until CPU0's ampSubmit (or ampStopCore1)
             * executes SEV. A SEV that lands between the check above and this
             * WFE is latched in the event register, so no wake-up is lost --
             * and CPU1 is not hammering uncached DDR3 while it waits. */
            __asm__ volatile ( "wfe" );
            continue;
        }

        /* The real work: exactly the code the single-core build runs. The
         * command CPU0 wrote before bumping req: a guided read (VJOB_FULL)
         * or one step of a grid scan. */
        ampCacheInvalidate( &pxJob->cmd, sizeof( pxJob->cmd ) );
        ulStart = ulGlobalTimerNow();
        vision_do( &pxJob->cmd, &pxJob->res );
        ulUs = ulGlobalTimerToUs( ulGlobalTimerNow() - ulStart );

        pxShared->core1_busy_us += ulUs;
        pxShared->core1_jobs++;

        ampCacheClean( &pxJob->res, sizeof( pxJob->res ) );
        ampCacheClean( ( const void * ) pxShared, 32 );
        dmb();

        pxJob->done = ulReq;
        ampCacheClean( ( const void * ) pxJob, AMP_JOB_HEAD_BYTES );
        dsb();

        /* Doorbell: SGI AMP_SGI_DONE, target list = CPU0. */
        GICD_SGIR = ( 1UL << 16 ) | AMP_SGI_DONE;
    }

    pxShared->core1_alive = 0;
    dsb();
    for( ;; )
    {
        __asm__ volatile ( "wfe" );
    }
}

/*
 * Where CPU1 lands. Naked because there is no stack yet -- setting one up is
 * the first thing it does. It also enables the FPU/NEON exactly as hwlib's
 * lowlevel_init does for CPU0: that runs only on CPU0, and a compiler-emitted
 * NEON instruction on a core with the FPU disabled would take an undefined
 * instruction exception into a vector table CPU1 does not have.
 */
__attribute__( ( naked, noreturn ) ) void ampCore1Entry( void )
{
    __asm__ volatile (
        "cpsid  if                              \n" /* no interrupts on CPU1  */
        "ldr    sp, =" AMP_STR( AMP_CORE1_STACK_TOP ) "\n"
        "mrc    p15, 0, r0, c1, c1, 2           \n" /* NSACR: cp10/cp11       */
        "orr    r0, r0, #(0x3 << 20)            \n"
        "mcr    p15, 0, r0, c1, c1, 2           \n"
        "ldr    r0, =(0xf << 20)                \n" /* CPACR: full access     */
        "mcr    p15, 0, r0, c1, c0, 2           \n"
        "isb                                    \n"
        "mov    r3, #0x40000000                 \n" /* FPEXC.EN               */
        "vmsr   fpexc, r3                       \n"
        "bl     ampCore1MainTrampoline          \n"
        "b      .                               \n"
    );
}

/* Separate symbol so the naked function above contains nothing but asm. */
void ampCore1MainTrampoline( void );
void ampCore1MainTrampoline( void )
{
    ampCore1Main();
}

/* ---- CPU0 side ---------------------------------------------------------- */

static int prvCore1Started;

int ampCore1Running( void )
{
    return prvCore1Started && ( AMP->core1_alive == AMP_MAGIC );
}

int ampStartCore1( void )
{
    struct amp_shared * pxShared = AMP;
    volatile uint32_t * pulTramp = ( volatile uint32_t * ) AMP_TRAMPOLINE_ADDR;
    uint32_t ulDeadline;

    /* Clear the control fields; the result does not need zeroing. */
    pxShared->magic         = AMP_MAGIC;
    pxShared->core1_alive   = 0;
    pxShared->core1_jobs    = 0;
    pxShared->core1_busy_us = 0;
    pxShared->stop          = 0;
    pxShared->job.req = pxShared->job.done = 0;
    ampCacheClean( ( const void * ) pxShared, sizeof( *pxShared ) );

    /* Hold CPU1 in reset while the trampoline and the start address are set
     * up, in case something already released it. */
    REG32( RSTMGR_MPUMODRST ) |= RSTMGR_MPUMODRST_CPU1;
    dsb();

    /*
     *   0x00  LDR r0, [PC, #8]   ; r0 = SYSMGR_CPU1STARTADDR (literal at 0x10)
     *   0x04  LDR r1, [r0]       ; r1 = entry point
     *   0x08  BX  r1
     *   0x0C  NOP
     *   0x10  .word 0xFFD080C4
     */
    pulTramp[ 0 ] = 0xE59F0008UL;
    pulTramp[ 1 ] = 0xE5901000UL;
    pulTramp[ 2 ] = 0xE12FFF11UL;
    pulTramp[ 3 ] = 0xE320F000UL;
    pulTramp[ 4 ] = SYSMGR_CPU1STARTADDR;

    REG32( SYSMGR_CPU1STARTADDR ) = ( uint32_t ) ( uintptr_t ) ampCore1Entry;

    /* CPU1 fetches this as instructions, so it has to be out of this core's
     * data cache and visible at the point of coherency before reset drops. */
    ampCacheClean( ( const void * ) pulTramp, 32 );
    dsb();

    /* The doorbell SGI must be no more urgent than the FreeRTOS API ceiling,
     * or the kernel will assert the first time the handler touches it. */
    GICD_IPRIORITY8[ AMP_SGI_DONE ] = ( uint8_t ) ( 20UL << 3 );

    REG32( RSTMGR_MPUMODRST ) &= ~RSTMGR_MPUMODRST_CPU1;
    dsb();

    /* Give it ~200 ms of global-timer time to check in. */
    ulDeadline = ulGlobalTimerNow();
    for( ;; )
    {
        ampCacheInvalidate( ( const void * ) pxShared, 32 );
        if( pxShared->core1_alive == AMP_MAGIC )
        {
            prvCore1Started = 1;
            return 1;
        }
        if( ulGlobalTimerToUs( ulGlobalTimerNow() - ulDeadline ) > 200000UL )
        {
            break;
        }
    }

    prvCore1Started = 0;
    return 0;
}

void ampStopCore1( void )
{
    if( prvCore1Started )
    {
        AMP->stop = 1;
        ampCacheClean( ( const void * ) AMP, 32 );
        dsb();
        __asm__ volatile ( "sev" );     /* CPU1 may be asleep in WFE */
    }
}

void ampSubmit( void )
{
    struct amp_job * pxJob = &AMP->job;

    dmb();
    pxJob->req++;
    ampCacheClean( ( const void * ) pxJob, AMP_JOB_HEAD_BYTES );
    dsb();
    __asm__ volatile ( "sev" );
}

int ampWait( uint32_t ulTimeoutUs )
{
    struct amp_job * pxJob   = &AMP->job;
    uint32_t         ulStart = ulGlobalTimerNow();

    for( ;; )
    {
        ampCacheInvalidate( ( const void * ) pxJob, AMP_JOB_HEAD_BYTES );
        if( pxJob->done == pxJob->req )
        {
            ampCacheInvalidate( &pxJob->res, sizeof( pxJob->res ) );
            return 1;
        }
        if( ulGlobalTimerToUs( ulGlobalTimerNow() - ulStart ) > ulTimeoutUs )
        {
            return 0;
        }

        /* Sleep rather than spin. taskYIELD() would only hand over to tasks
         * of equal or higher priority, so the report task -- deliberately the
         * lowest -- would never run while a frame was in flight on CPU1. One
         * tick of latency against a frame period two orders of magnitude
         * larger is not worth optimising away. */
        if( xTaskGetSchedulerState() == taskSCHEDULER_RUNNING )
        {
            vTaskDelay( 1 );
        }
    }
}
#endif /* RTOS_MODE */
