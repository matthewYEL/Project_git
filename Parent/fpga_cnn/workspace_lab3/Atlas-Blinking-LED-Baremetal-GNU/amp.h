/*
 * Asymmetric multiprocessing across the two Cortex-A9 cores of the Cyclone V
 * HPS, bare metal.
 *
 * CPU0 runs FreeRTOS and owns the FPGA. CPU1 runs no operating system at all:
 * it is released from reset into a plain worker loop that preprocesses frames
 * out of a shared DDR3 block. Work is handed over through that block, not
 * through the RTOS, because FreeRTOS is not SMP-safe and CPU1 must never touch
 * a kernel object.
 *
 * Frame N is preprocessed on CPU1 while CPU0 is already capturing frame N+1,
 * which is the point of the exercise: the two halves overlap rather than
 * taking turns.
 */
#ifndef AMP_H
#define AMP_H

#include <stdint.h>

#include "card_pipeline.h"

/*
 * Shared block, placed high in DDR3 and well clear of everything else:
 * u-boot's SPL sits at 0x0, the application links at 0x00100040 and its stack
 * is at 0x80000, so 48 MB up is untouched by all of them.
 *
 * The numeric literals are repeated in the asm below because a naked function
 * cannot take operands; AMP_STACK_TOP_STR keeps the two spellings tied
 * together.
 */
/* The window has to hold AMP_SLOTS frame slots plus 32 KB of CPU1 stack below
 * the top -- see the amp_shared_fits assertion in amp.c. At 96x96 a slot is
 * 36,896 bytes (raw + q + head), so two slots need 73,824 and the 96 KB window
 * the 48x48 build used no longer fits. 192 KB, of which the assertion lets the
 * block use 160 KB. */
#define AMP_SHARED_BASE     0x03000000
#define AMP_CORE1_STACK_TOP 0x03030000

#define AMP_SLOTS           2
#define AMP_MAGIC           0x414D5031UL   /* "AMP1" */

/* The address CPU1 fetches from when it leaves reset, and the system manager
 * register the trampoline reads the real entry point out of. Both are fixed by
 * the Cyclone V hardware. */
#define AMP_TRAMPOLINE_ADDR 0x00000000UL
#define SYSMGR_CPU1STARTADDR 0xFFD080C4UL   /* sysmgr.romcodegrp.cpu1startaddr */
#define RSTMGR_MPUMODRST     0xFFD05010UL   /* rstmgr.mpumodrst                */
#define RSTMGR_MPUMODRST_CPU1 ( 1UL << 1 )

/*
 * One frame in flight. req/done are sequence numbers rather than flags so a
 * missed wake-up cannot be mistaken for "already finished".
 *
 * The control fields come FIRST and the bulk arrays after, deliberately: the
 * handshake is synchronised by cleaning/invalidating just the head of the
 * struct, which only works if the flags live there. With the arrays in front,
 * a 64-byte maintenance op would land in the middle of raw[] and never touch
 * req/done at all.
 *
 * Each array is cache-line aligned so that cleaning one cannot write back a
 * stale line belonging to the other.
 */
struct amp_slot
{
    volatile uint32_t req;                 /* bumped by CPU0 when raw is ready */
    volatile uint32_t done;                /* set to req by CPU1 when q is ready */
    volatile uint32_t us;                  /* CPU1 time for this slot, in us */
    uint32_t          red_count;
    uint32_t          minv, maxv;
    uint32_t          reserved[ 2 ];       /* pad the head to one 32-byte line */

    uint16_t          raw[ IMG_PIXELS ] __attribute__( ( aligned( 32 ) ) );  /* CPU0 writes */
    uint16_t          q[ IMG_PIXELS ]   __attribute__( ( aligned( 32 ) ) );  /* CPU1 writes */
};

/* How much of a slot the handshake has to push around: the control head only. */
#define AMP_SLOT_HEAD_BYTES  32u

struct amp_shared
{
    volatile uint32_t magic;
    volatile uint32_t core1_alive;         /* CPU1 sets this to AMP_MAGIC */
    volatile uint32_t core1_frames;        /* frames CPU1 has preprocessed */
    volatile uint32_t core1_busy_us;       /* cumulative CPU1 work, microseconds */
    volatile uint32_t stop;                /* CPU0 asks CPU1 to leave the loop */
    volatile uint32_t pad[ 3 ];
    struct amp_slot   slot[ AMP_SLOTS ];
};

#define AMP ( ( struct amp_shared * ) AMP_SHARED_BASE )

/* SGI 0, CPU1 -> CPU0, raised when a slot is finished. Purely an optimisation:
 * the waiting task also times out and rechecks the slot, so a lost or
 * misconfigured SGI costs latency, not correctness. */
#define AMP_SGI_DONE        0UL

/* Returns 1 if CPU1 checked in before the timeout, 0 if it never did. The
 * caller is expected to fall back to doing the work itself -- releasing a
 * second core is the feature most likely to fail on a given board, and it must
 * not be able to take the demo down with it. */
int  ampStartCore1( void );

int  ampCore1Running( void );
void ampStopCore1( void );

/* Hand a slot to CPU1 / collect it. ampSubmit bumps req; ampWait spins for
 * done == req with a bounded timeout, returning 0 on timeout. */
void ampSubmit( struct amp_slot * pxSlot );
int  ampWait( struct amp_slot * pxSlot, uint32_t ulTimeoutUs );

/* Cache maintenance over the shared block. No-ops when the MMU is off (all
 * accesses are then strongly ordered), correct when it is on. */
void ampCacheClean( const void * pvAddr, uint32_t ulBytes );
void ampCacheInvalidate( const void * pvAddr, uint32_t ulBytes );

/* Entry point CPU1 lands on. Not called from C. */
void ampCore1Entry( void );

#endif /* AMP_H */
