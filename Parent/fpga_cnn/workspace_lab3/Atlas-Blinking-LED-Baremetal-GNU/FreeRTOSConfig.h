/*
 * FreeRTOS configuration for the Cyclone V HPS (Cortex-A9 MPCore) on the
 * DE10-Nano, running under the Arm DS bare-metal flow.
 *
 * The kernel runs on CPU0 only. CPU1 is released separately as a bare-metal
 * worker (amp.c) -- this is AMP, not SMP, so nothing here is shared with it.
 *
 * Interrupt controller and tick come from the A9 MPCore private peripherals,
 * whose base (PERIPHBASE) is 0xFFFEC000 on Cyclone V:
 *   GIC CPU interface (GICC)  0xFFFEC100
 *   private timer             0xFFFEC600   (interrupt ID 29, a PPI)
 *   global timer              0xFFFEC200   (free-running, run-time stats)
 *   GIC distributor  (GICD)   0xFFFED000
 * FreeRTOS wants the DISTRIBUTOR as the base and reaches the CPU interface by
 * adding a negative offset, exactly as the Zynq port does.
 */
#ifndef FREERTOS_CONFIG_H
#define FREERTOS_CONFIG_H

#ifndef __ASSEMBLER__
    #include <stdint.h>
    void vConfigureTickInterrupt( void );
    void vClearTickInterrupt( void );
    void vRunTimeStatsTimerInit( void );
    uint32_t ulRunTimeStatsCounter( void );
    void vAssertCalled( const char *pcFile, unsigned long ulLine );
#endif

/* ---- scheduler behaviour ---------------------------------------------- */
#define configUSE_PREEMPTION                    1
#define configUSE_TIME_SLICING                  1
#define configUSE_PORT_OPTIMISED_TASK_SELECTION 0
#define configUSE_IDLE_HOOK                     0
#define configUSE_TICK_HOOK                     0
#define configTICK_RATE_HZ                      ( ( TickType_t ) 1000 )
#define configMAX_PRIORITIES                    7
#define configMINIMAL_STACK_SIZE                ( ( unsigned short ) 256 )
#define configTOTAL_HEAP_SIZE                   ( ( size_t ) ( 96 * 1024 ) )
#define configMAX_TASK_NAME_LEN                 16
#define configUSE_16_BIT_TICKS                  0
#define configIDLE_SHOULD_YIELD                 1
#define configUSE_MUTEXES                       1
#define configUSE_COUNTING_SEMAPHORES           1
#define configQUEUE_REGISTRY_SIZE               8
#define configSUPPORT_DYNAMIC_ALLOCATION        1
#define configSUPPORT_STATIC_ALLOCATION         0
#define configCHECK_FOR_STACK_OVERFLOW          2
#define configUSE_MALLOC_FAILED_HOOK            1

/* printf() is newlib's and is not reentrant. Rather than pay for
 * configUSE_NEWLIB_REENTRANT, every task funnels its output through the
 * report task -- see app_rtos.c. Keep it that way. */
#define configUSE_NEWLIB_REENTRANT              0

/* ---- run-time statistics ----------------------------------------------
 * This is the scheduling evidence: vTaskGetRunTimeStats() prints how long
 * each task actually held the CPU. Clocked off the A9 global timer. */
#define configGENERATE_RUN_TIME_STATS           1
#define configUSE_TRACE_FACILITY                1
#define configUSE_STATS_FORMATTING_FUNCTIONS    1
#define portCONFIGURE_TIMER_FOR_RUN_TIME_STATS()  vRunTimeStatsTimerInit()
#define portGET_RUN_TIME_COUNTER_VALUE()          ulRunTimeStatsCounter()

/* ---- software timers / co-routines ------------------------------------ */
#define configUSE_TIMERS                        1
#define configTIMER_TASK_PRIORITY               ( configMAX_PRIORITIES - 1 )
#define configTIMER_QUEUE_LENGTH                8
#define configTIMER_TASK_STACK_DEPTH            ( configMINIMAL_STACK_SIZE * 2 )
#define configUSE_CO_ROUTINES                   0
#define configMAX_CO_ROUTINE_PRIORITIES         2

/* ---- API inclusion ----------------------------------------------------- */
#define INCLUDE_vTaskPrioritySet                1
#define INCLUDE_uxTaskPriorityGet               1
#define INCLUDE_vTaskDelete                     1
#define INCLUDE_vTaskSuspend                    1
#define INCLUDE_vTaskDelayUntil                 1
#define INCLUDE_vTaskDelay                      1
#define INCLUDE_xTaskGetSchedulerState          1
#define INCLUDE_uxTaskGetStackHighWaterMark     1
#define INCLUDE_xTaskGetIdleTaskHandle          1
#define INCLUDE_eTaskGetState                   1

/* ---- Cortex-A9 port ----------------------------------------------------
 * Every task is given an FPU context. The build passes -mfpu=neon, so a
 * compiler-generated NEON access in any task would otherwise corrupt state. */
#define configUSE_TASK_FPU_SUPPORT              2

/* GIC distributor, and the offset from it back to the CPU interface. */
#define configINTERRUPT_CONTROLLER_BASE_ADDRESS         ( 0xFFFED000UL )
#define configINTERRUPT_CONTROLLER_CPU_INTERFACE_OFFSET ( -0xF00 )

/* The A9 GIC implements 5 priority bits -> 32 distinct levels. Numerically
 * lower means logically higher, so 18 leaves 0..17 for interrupts that must
 * never call a FreeRTOS API. */
#define configUNIQUE_INTERRUPT_PRIORITIES               32
#define configMAX_API_CALL_INTERRUPT_PRIORITY           18

#define configSETUP_TICK_INTERRUPT()            vConfigureTickInterrupt()
#define configCLEAR_TICK_INTERRUPT()            vClearTickInterrupt()

#define configASSERT( x )   if( ( x ) == 0 ) vAssertCalled( __FILE__, __LINE__ )

#endif /* FREERTOS_CONFIG_H */
