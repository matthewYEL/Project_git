/*
 * Cortex-A9 exception, GIC and timer bring-up for FreeRTOS on the Cyclone V HPS.
 *
 * The lab's bare-metal template does none of this: hwlib's alt_base.c enables
 * NEON and jumps straight to _mainCRTStartup, so there is no vector table, no
 * VBAR, no interrupt controller and no tick. main() runs in SVC mode with
 * interrupts masked, which is fine for polling PIOs and useless for an RTOS.
 * Everything the FreeRTOS ARM_CA9 port expects to already exist is set up here.
 *
 * A9 MPCore private peripherals sit at PERIPHBASE 0xFFFEC000 on Cyclone V:
 *
 *   0xFFFEC100  GIC CPU interface (GICC)
 *   0xFFFEC200  global timer      -- free-running, drives the run-time stats
 *   0xFFFEC600  private timer     -- the RTOS tick, interrupt ID 29 (a PPI)
 *   0xFFFED000  GIC distributor   (GICD)
 *
 * The tick rate is derived from the real mpu_periph_clk read out of the clock
 * manager rather than assumed, because that clock is board- and preloader-
 * dependent and a wrong guess yields a scheduler that runs at the wrong speed
 * without ever looking broken.
 */
#include <stdio.h>
#include <stdint.h>

#include "FreeRTOS.h"
#include "task.h"

#include "alt_clock_manager.h"

/* Provided by the FreeRTOS ARM_CA9 port (portASM.S / port.c). */
extern void FreeRTOS_IRQ_Handler( void );
extern void FreeRTOS_SWI_Handler( void );
extern void FreeRTOS_Tick_Handler( void );

/* ---- register map ------------------------------------------------------ */
#define PERIPHBASE          0xFFFEC000UL
#define GICC_BASE           ( PERIPHBASE + 0x100UL )
#define GLOBALTMR_BASE      ( PERIPHBASE + 0x200UL )
#define PRIVTMR_BASE        ( PERIPHBASE + 0x600UL )
#define GICD_BASE           0xFFFED000UL

#define REG32( a )          ( *( volatile uint32_t * ) ( a ) )

/* GIC CPU interface */
#define GICC_CTLR           REG32( GICC_BASE + 0x00 )
#define GICC_PMR            REG32( GICC_BASE + 0x04 )
#define GICC_BPR            REG32( GICC_BASE + 0x08 )

/* GIC distributor */
#define GICD_CTLR           REG32( GICD_BASE + 0x000 )
#define GICD_ISENABLER( n ) REG32( GICD_BASE + 0x100 + 4UL * ( n ) )
#define GICD_ICENABLER( n ) REG32( GICD_BASE + 0x180 + 4UL * ( n ) )
#define GICD_IPRIORITY8     ( ( volatile uint8_t * ) ( GICD_BASE + 0x400 ) )
#define GICD_SGIR           REG32( GICD_BASE + 0xF00 )

/* A9 private timer */
#define PRIV_LOAD           REG32( PRIVTMR_BASE + 0x00 )
#define PRIV_COUNTER        REG32( PRIVTMR_BASE + 0x04 )
#define PRIV_CONTROL        REG32( PRIVTMR_BASE + 0x08 )
#define PRIV_INTSTAT        REG32( PRIVTMR_BASE + 0x0C )
#define PRIV_CTRL_ENABLE    ( 1UL << 0 )
#define PRIV_CTRL_AUTORELD  ( 1UL << 1 )
#define PRIV_CTRL_IRQEN     ( 1UL << 2 )

/* A9 global timer */
#define GT_COUNTER_LO       REG32( GLOBALTMR_BASE + 0x00 )
#define GT_COUNTER_HI       REG32( GLOBALTMR_BASE + 0x04 )
#define GT_CONTROL          REG32( GLOBALTMR_BASE + 0x08 )

#define PRIVATE_TIMER_IRQ_ID    29UL

/* The tick ISR calls xTaskIncrementTick, so its GIC priority must be
 * numerically at or below configMAX_API_CALL_INTERRUPT_PRIORITY in logical
 * terms -- that is, numerically >= it. The A9 GIC implements the top 5 bits of
 * the 8-bit priority field, hence the shift of 3. */
#define TICK_PRIORITY           20UL
#define GIC_PRIORITY_SHIFT      3

/* Fallback if the clock manager cannot be read. mpu_clk on a DE10-Nano is
 * 800 MHz and mpu_periph_clk is mpu_clk/4. Only used if alt_clk_freq_get
 * fails, and it says so when it does. */
#define MPU_PERIPH_CLK_FALLBACK 200000000UL

static uint32_t ulPeriphClockHz;

/* ---- exception vectors -------------------------------------------------
 * ARM vector table: 8 entries, 4 bytes each. SVC and IRQ go to the FreeRTOS
 * port; the faults land in reporting stubs rather than a silent lock-up,
 * because a data abort on this platform is nearly always a bad PIO address
 * and you want to be told that. */
__asm__(
    ".section .text.vectors, \"ax\"                  \n"
    ".align  5                                       \n"
    ".global _freertos_vector_table                   \n"
    "_freertos_vector_table:                          \n"
    "    B       .                                    \n" /* 0x00 reset      */
    "    B       vUndefinedInstructionHandler         \n" /* 0x04 undefined  */
    "    LDR     PC, _vt_swi_addr                     \n" /* 0x08 SVC        */
    "    B       vPrefetchAbortHandler                \n" /* 0x0C prefetch   */
    "    B       vDataAbortHandler                    \n" /* 0x10 data abort */
    "    NOP                                          \n" /* 0x14 reserved   */
    "    LDR     PC, _vt_irq_addr                     \n" /* 0x18 IRQ        */
    "    B       .                                    \n" /* 0x1C FIQ        */
    "_vt_swi_addr:  .word FreeRTOS_SWI_Handler        \n"
    "_vt_irq_addr:  .word FreeRTOS_IRQ_Handler        \n"
    ".text                                            \n"
);

extern uint32_t _freertos_vector_table;

/* Separate stacks for the exception modes. Only the IRQ one is used in anger:
 * FreeRTOS_IRQ_Handler runs a few instructions on it before switching to
 * system mode, so it does not need to be large. */
#define IRQ_STACK_WORDS     512
#define ABT_STACK_WORDS     128
#define UND_STACK_WORDS     128

static uint32_t ulIRQStack[ IRQ_STACK_WORDS ] __attribute__( ( aligned( 8 ) ) );
static uint32_t ulABTStack[ ABT_STACK_WORDS ] __attribute__( ( aligned( 8 ) ) );
static uint32_t ulUNDStack[ UND_STACK_WORDS ] __attribute__( ( aligned( 8 ) ) );

void vUndefinedInstructionHandler( void )
{
    printf( "\n*** undefined instruction ***\n" );
    for( ;; ) { }
}

void vPrefetchAbortHandler( void )
{
    printf( "\n*** prefetch abort ***\n" );
    for( ;; ) { }
}

void vDataAbortHandler( void )
{
    uint32_t ulDFAR = 0, ulDFSR = 0;

    __asm__ volatile ( "mrc p15, 0, %0, c6, c0, 0" : "=r" ( ulDFAR ) );
    __asm__ volatile ( "mrc p15, 0, %0, c5, c0, 0" : "=r" ( ulDFSR ) );
    printf( "\n*** data abort at 0x%08lx (DFSR 0x%08lx) ***\n",
            ( unsigned long ) ulDFAR, ( unsigned long ) ulDFSR );
    for( ;; ) { }
}

/* Give each exception mode a stack, then point VBAR at our table and make sure
 * SCTLR.V is clear so VBAR is actually consulted. */
static void prvSetupExceptionModes( void )
{
    uint32_t ulIRQTop = ( uint32_t ) ( ulIRQStack + IRQ_STACK_WORDS );
    uint32_t ulABTTop = ( uint32_t ) ( ulABTStack + ABT_STACK_WORDS );
    uint32_t ulUNDTop = ( uint32_t ) ( ulUNDStack + UND_STACK_WORDS );

    __asm__ volatile (
        "mrs  r3, cpsr              \n"
        "bic  r2, r3, #0x1f         \n"
        "orr  r2, r2, #0x12         \n"     /* IRQ mode       */
        "msr  cpsr_c, r2            \n"
        "mov  sp, %0                \n"
        "bic  r2, r3, #0x1f         \n"
        "orr  r2, r2, #0x17         \n"     /* Abort mode     */
        "msr  cpsr_c, r2            \n"
        "mov  sp, %1                \n"
        "bic  r2, r3, #0x1f         \n"
        "orr  r2, r2, #0x1b         \n"     /* Undefined mode */
        "msr  cpsr_c, r2            \n"
        "mov  sp, %2                \n"
        "msr  cpsr_c, r3            \n"     /* back to where we started */
        :
        : "r" ( ulIRQTop ), "r" ( ulABTTop ), "r" ( ulUNDTop )
        : "r2", "r3", "memory"
    );
}

void vInstallVectorTable( void )
{
    uint32_t ulSCTLR;

    prvSetupExceptionModes();

    /* SCTLR.V (bit 13) selects the legacy high vectors at 0xFFFF0000; VBAR is
     * only used when it is clear. */
    __asm__ volatile ( "mrc p15, 0, %0, c1, c0, 0" : "=r" ( ulSCTLR ) );
    ulSCTLR &= ~( 1UL << 13 );
    __asm__ volatile ( "mcr p15, 0, %0, c1, c0, 0" :: "r" ( ulSCTLR ) );

    __asm__ volatile ( "mcr p15, 0, %0, c12, c0, 0"
                       :: "r" ( &_freertos_vector_table ) );
    __asm__ volatile ( "dsb" ::: "memory" );
    __asm__ volatile ( "isb" ::: "memory" );
}

/* ---- interrupt controller ---------------------------------------------- */

void vInitialiseGIC( void )
{
    /* Take the distributor down while it is reconfigured -- the preloader may
     * have left it in any state. */
    GICD_CTLR = 0;

    /* Nothing is wanted from IDs 0..31 except the private timer; SGI 0 is left
     * enabled because amp.c uses it for the CPU1 doorbell. */
    GICD_ICENABLER( 0 ) = 0xFFFFFFFFUL;

    GICD_CTLR = 1;

    /* PMR 0xFF: do not mask anything at the CPU interface -- FreeRTOS drives
     * masking itself through this register. BPR 0: no sub-priority grouping. */
    GICC_PMR  = 0xFF;
    GICC_BPR  = 0;
    GICC_CTLR = 1;
}

uint32_t ulGetPeriphClockHz( void )
{
    return ulPeriphClockHz;
}

/* Idempotent: the AMP handshake needs a timebase before the scheduler starts,
 * and the port asks again when it sets up the tick. */
void vInitPeriphClock( void )
{
    alt_freq_t xFreq = 0;

    if( ulPeriphClockHz != 0UL )
    {
        return;
    }

    if( ( alt_clk_freq_get( ALT_CLK_MPU_PERIPH, &xFreq ) == ALT_E_SUCCESS ) &&
        ( xFreq > 1000000u ) )
    {
        ulPeriphClockHz = ( uint32_t ) xFreq;
    }
    else
    {
        ulPeriphClockHz = MPU_PERIPH_CLK_FALLBACK;
        printf( "  ! mpu_periph_clk unreadable, assuming %lu Hz -- the tick "
                "rate will be wrong if that is not right\n",
                ( unsigned long ) ulPeriphClockHz );
    }
}

void vConfigureTickInterrupt( void )
{
    uint32_t ulLoad;

    vInitPeriphClock();

    ulLoad = ( ulPeriphClockHz / configTICK_RATE_HZ ) - 1UL;

    PRIV_CONTROL = 0;                       /* stop before reprogramming */
    PRIV_LOAD    = ulLoad;
    PRIV_INTSTAT = 1;                       /* discard anything pending */

    GICD_IPRIORITY8[ PRIVATE_TIMER_IRQ_ID ] =
        ( uint8_t ) ( TICK_PRIORITY << GIC_PRIORITY_SHIFT );
    GICD_ISENABLER( 0 ) = 1UL << PRIVATE_TIMER_IRQ_ID;

    PRIV_CONTROL = PRIV_CTRL_ENABLE | PRIV_CTRL_AUTORELD | PRIV_CTRL_IRQEN;

    printf( "  tick: private timer, mpu_periph_clk %lu Hz, load %lu "
            "-> %u Hz\n",
            ( unsigned long ) ulPeriphClockHz, ( unsigned long ) ulLoad,
            ( unsigned ) configTICK_RATE_HZ );
}

void vClearTickInterrupt( void )
{
    PRIV_INTSTAT = 1;
}

/* ---- run-time statistics ----------------------------------------------
 * The global timer is 64-bit and free-running at mpu_periph_clk. FreeRTOS
 * wants a counter running well above the tick rate but not so fast that a
 * 32-bit value wraps during a demo: >>8 gives ~780 kHz from 200 MHz, which is
 * 780x the 1 kHz tick and wraps in about 90 minutes. */
void vRunTimeStatsTimerInit( void )
{
    GT_CONTROL    = 0;
    GT_COUNTER_LO = 0;
    GT_COUNTER_HI = 0;
    GT_CONTROL    = 1;
}

uint32_t ulRunTimeStatsCounter( void )
{
    return GT_COUNTER_LO >> 8;
}

/* Free-running microsecond-ish timebase for the "who did what" accounting in
 * the tasks. Same source, unscaled. */
uint32_t ulGlobalTimerNow( void )
{
    return GT_COUNTER_LO;
}

uint32_t ulGlobalTimerToUs( uint32_t ulTicks )
{
    uint32_t ulMHz = ulPeriphClockHz ? ( ulPeriphClockHz / 1000000UL ) : 200UL;

    return ulTicks / ulMHz;
}

/* ---- interrupt dispatch ------------------------------------------------
 * FreeRTOS_IRQ_Handler has already read the ICCIAR and will write the ICCEOIR
 * when this returns; all that is needed here is the demultiplex. */
void vApplicationIRQHandler( uint32_t ulICCIAR )
{
    uint32_t ulID = ulICCIAR & 0x3FFUL;

    if( ulID == PRIVATE_TIMER_IRQ_ID )
    {
        FreeRTOS_Tick_Handler();
    }

    /* Any other ID is spurious: nothing else is enabled in the distributor. */
}

/* ---- FreeRTOS hooks ---------------------------------------------------- */

void vApplicationMallocFailedHook( void )
{
    taskDISABLE_INTERRUPTS();
    printf( "\n*** FreeRTOS heap exhausted (configTOTAL_HEAP_SIZE %u) ***\n",
            ( unsigned ) configTOTAL_HEAP_SIZE );
    for( ;; ) { }
}

void vApplicationStackOverflowHook( TaskHandle_t xTask, char * pcTaskName )
{
    ( void ) xTask;
    taskDISABLE_INTERRUPTS();
    printf( "\n*** stack overflow in task \"%s\" ***\n", pcTaskName );
    for( ;; ) { }
}

void vAssertCalled( const char * pcFile, unsigned long ulLine )
{
    taskDISABLE_INTERRUPTS();
    printf( "\n*** configASSERT failed: %s:%lu ***\n", pcFile, ulLine );
    for( ;; ) { }
}
