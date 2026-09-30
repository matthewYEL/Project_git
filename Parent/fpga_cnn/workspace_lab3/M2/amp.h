/*
 * Asymmetric multiprocessing across the two Cortex-A9 cores of the Cyclone V
 * HPS, bare metal.
 *
 * CPU0 runs FreeRTOS (input, job control, reporting). CPU1 runs no operating
 * system at all: it is released from reset into a plain worker loop that runs
 * the vision job -- capture, sample, CNN, decode (vision_run in atlas_main.c)
 * -- and owns the camera/CNN PIOs while a job is in flight. Work is handed over
 * through a shared DDR3 block, not through the RTOS, because FreeRTOS is not
 * SMP-safe and CPU1 must never touch a kernel object. CPU1 never printfs:
 * semihosting traps to the debugger, which is attached to CPU0 only.
 *
 * The 384x384 snapshot build changed what crosses the cores: the fabric now
 * writes the CNN input itself, so the block carries a job request and its
 * result (~2.4 KB), not two frames of pixels.
 */
#ifndef AMP_H
#define AMP_H

#include <stdint.h>

#include "card_pipeline.h"

/*
 * Shared block, placed high in DDR3 and well clear of everything else:
 * u-boot's SPL sits at 0x0, the application links at 0x00100040 and its stack
 * is at 0x80000, fc_shared's weights start at 0x10000000 (card_pipeline.h), so
 * 48 MB up is untouched by all of them.
 *
 * The numeric literals are repeated in the asm in amp.c because a naked
 * function cannot take operands. amp.c asserts the block stays below CPU1's
 * 32 KB stack.
 */
#define AMP_SHARED_BASE     0x03000000
#define AMP_CORE1_STACK_TOP 0x03030000

#define AMP_MAGIC           0x414D5032UL   /* "AMP2": the snapshot-job layout */

/* Hardware-fixed: where CPU1 fetches from when it leaves reset, the system
 * manager register the trampoline reads the real entry point out of, and the
 * reset manager bit that holds CPU1. */
#define AMP_TRAMPOLINE_ADDR 0x00000000UL
#define SYSMGR_CPU1STARTADDR 0xFFD080C4UL   /* sysmgr.romcodegrp.cpu1startaddr */
#define RSTMGR_MPUMODRST     0xFFD05010UL   /* rstmgr.mpumodrst                */
#define RSTMGR_MPUMODRST_CPU1 ( 1UL << 1 )

/*
 * The one job in flight. req/done are sequence numbers rather than flags so a
 * missed wake-up cannot be mistaken for "already finished". The control words
 * come first, in their own 32-byte line, so the handshake can clean/invalidate
 * just that line (a no-op with the MMU off, correct if it is ever turned on).
 */
struct amp_job
{
    volatile uint32_t    req;           /* bumped by CPU0 to hand CPU1 a job   */
    volatile uint32_t    done;          /* set to req by CPU1 when res is valid */
    uint32_t             reserved[ 6 ]; /* pad the head to one 32-byte line     */

    struct vision_job    cmd __attribute__( ( aligned( 32 ) ) );   /* CPU0 writes before req */
    struct vision_result res __attribute__( ( aligned( 32 ) ) );   /* CPU1 writes */
};

#define AMP_JOB_HEAD_BYTES  32u

struct amp_shared
{
    volatile uint32_t magic;
    volatile uint32_t core1_alive;      /* CPU1 sets this to AMP_MAGIC        */
    volatile uint32_t core1_jobs;       /* jobs CPU1 has completed            */
    volatile uint32_t core1_busy_us;    /* cumulative CPU1 job time, us       */
    volatile uint32_t stop;             /* CPU0 asks CPU1 to leave the loop   */
    volatile uint32_t pad[ 3 ];
    struct amp_job    job __attribute__( ( aligned( 32 ) ) );
};

#define AMP ( ( struct amp_shared * ) AMP_SHARED_BASE )

/* SGI 0, CPU1 -> CPU0, raised when a job is finished. Purely an optimisation:
 * the waiting task also polls, so a lost SGI costs latency, not correctness. */
#define AMP_SGI_DONE        0UL

/* Returns 1 if CPU1 checked in before the timeout, 0 if it never did. The
 * caller falls back to running the job on CPU0 -- releasing a second core is
 * the feature most likely to fail on a given board, and it must not be able to
 * take the demo down with it. */
int  ampStartCore1( void );

int  ampCore1Running( void );
void ampStopCore1( void );

/* Hand the job to CPU1 / collect it. ampSubmit bumps req; ampWait sleeps (one
 * RTOS tick at a time) until done == req, returning 0 on timeout. */
void ampSubmit( void );
int  ampWait( uint32_t ulTimeoutUs );

/* Cache maintenance over the shared block. No-ops when the MMU is off (all
 * accesses are then strongly ordered), correct when it is on. */
void ampCacheClean( const void * pvAddr, uint32_t ulBytes );
void ampCacheInvalidate( const void * pvAddr, uint32_t ulBytes );

/* Entry point CPU1 lands on. Not called from C. */
void ampCore1Entry( void );

#endif /* AMP_H */
