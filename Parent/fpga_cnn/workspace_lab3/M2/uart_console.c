/*
 * Console on HPS UART0 -- the board's mini-USB "UART" port, 115200 8N1.
 * Built when SEMIHOSTED=0, the default in this project's Makefile.
 *
 * Why not semihosting: with semihosting on, Arm DS traps every SVC to see
 * whether it is a console request, and the FreeRTOS port switches tasks with
 * SVC. Every context switch then froze CPU0 for ~0.5 s over the USB-Blaster:
 * on 26 Sep the Input task was charged 93% of CPU0 and each result took about
 * 1.8 min, although CPU1 finished every job in 1.7 s.
 *
 * newlib's printf / fputs / putchar all end in _write(), so retargeting it here
 * moves every existing console line to the UART unchanged. Only CPU0 prints --
 * main() before the scheduler, then the Report task alone; CPU1 never does --
 * so no locking is needed.
 */
#include <stddef.h>
#include <stdio.h>

#include "alt_16550_uart.h"
#include "card_pipeline.h"

#ifndef PRINTF_HOST

static ALT_16550_HANDLE_t uart0;
static int                uart0_ok;

void uart_console_init(void)
{
    /* The sequence hwlib's alt_p2uart.c uses. For UART0, alt_16550_init takes
     * the L4_SP clock itself and sets PTIME, so fifo_write_safe below waits
     * only while the TX FIFO is full; the location/frequency arguments are
     * unused for this device. */
    uart0_ok = alt_16550_init(ALT_16550_DEVICE_SOCFPGA_UART0, NULL, 0, &uart0) == ALT_E_SUCCESS
            && alt_16550_baudrate_set(&uart0, 115200) == ALT_E_SUCCESS
            && alt_16550_line_config_set(&uart0, ALT_16550_DATABITS_8,
                                         ALT_16550_PARITY_DISABLE,
                                         ALT_16550_STOPBITS_1) == ALT_E_SUCCESS
            && alt_16550_fifo_enable(&uart0) == ALT_E_SUCCESS
            && alt_16550_enable(&uart0) == ALT_E_SUCCESS;

    /* nosys has no working isatty, so newlib would fully buffer stdout and hold
     * lines back until 1 KB had piled up */
    setvbuf(stdout, NULL, _IONBF, 0);
}

/* newlib's hook for every write to a stream (sys/unistd.h: int, const void *,
 * size_t). Without a working UART the text is dropped rather than blocking. */
int _write(int fd, const void *buf, size_t len)
{
    const char *p = buf;
    size_t      i;

    (void)fd;
    if (!uart0_ok) return (int)len;
    for (i = 0; i < len; i++) {
        if (p[i] == '\n')
            alt_16550_fifo_write_safe(&uart0, "\r", 1, true);
        alt_16550_fifo_write_safe(&uart0, &p[i], 1, true);
    }
    return (int)len;
}

/* One received character, or -1 if none is waiting -- never blocks. The Input
 * task polls it for the operator's command lines; only it reads the RX side. */
int uart_getc(void)
{
    uint32_t level = 0;
    char     c;

    if (!uart0_ok || alt_16550_fifo_level_get_rx(&uart0, &level) != ALT_E_SUCCESS || level == 0)
        return -1;
    if (alt_16550_fifo_read(&uart0, &c, 1) != ALT_E_SUCCESS)
        return -1;
    return (unsigned char)c;
}

#else /* PRINTF_HOST: the console is the debugger's, which has no input here */

int uart_getc(void)
{
    return -1;
}

#endif /* !PRINTF_HOST */
