/* ---------------------------------------------------------------------------
 * bridge_test.c  --  HPS-to-FPGA bridge smoke test
 *
 * The Linux equivalent of LAB 5's bare-metal blinky. Its only job is to prove
 * that software on the HPS can read and write registers in the FPGA fabric.
 *
 * WHY BOTHER: if card_cnn.c misbehaves, the fault could be the accelerator,
 * the base address, the bridge configuration, or the FPGA not being programmed
 * at all. This isolates the last three. Get the LEDs blinking first and every
 * later failure is unambiguously the accelerator.
 *
 * The lab runs bare-metal through DS-5; this project runs embedded Linux (also
 * what satisfies the Milestone 1 "RTOS/LINUX on at least one processing
 * context" criterion), so the fabric is reached by mapping /dev/mem instead.
 *
 * Build (SoC EDS embedded command shell):
 *     arm-linux-gnueabihf-gcc -O2 -o bridge_test bridge_test.c
 *
 * Run on the board as root:
 *     ./bridge_test                 # uses the default LED PIO offset
 *     ./bridge_test 0x3000          # or pass the offset from Platform Designer
 *
 * FIND THE OFFSET: open soc_system.qsys, look at the Address Map tab, and read
 * the address the LED PIO is assigned on the h2f_lw_axi_master. Subtract the
 * bridge base (0xFF200000) if the tool shows the full address. Guessing is a
 * waste of time -- the GHRD's PIO offsets differ between board revisions and
 * between projects.
 * ------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define LW_BRIDGE_BASE   0xFF200000
#define LW_BRIDGE_SPAN   0x00200000

/* Common offset in the DE10-Nano GHRD -- verify against YOUR Platform
 * Designer address map rather than trusting this. */
#define DEFAULT_LED_OFFSET 0x00003000

int main(int argc, char **argv) {
    uint32_t led_offset = DEFAULT_LED_OFFSET;
    if (argc > 1) led_offset = (uint32_t)strtoul(argv[1], NULL, 0);

    printf("mapping lightweight bridge at 0x%08X, LED PIO at +0x%04X\n",
           LW_BRIDGE_BASE, led_offset);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem (run as root)");
        return 1;
    }

    volatile uint8_t *base = mmap(NULL, LW_BRIDGE_SPAN, PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd, LW_BRIDGE_BASE);
    if (base == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    volatile uint32_t *leds = (volatile uint32_t *)(base + led_offset);

    /* An Altera PIO configured as BIDIRECTIONAL has a direction register at
     * offset 0x4, and it resets to all-inputs. Until it is set, writes land in
     * the data register (so a read-back appears to work) but never reach the
     * pins -- the LEDs stay dark while everything looks fine.
     *
     * ghrd_top.v connects BOTH led_pio_external_connection_in_port and
     * _out_port, which is what a bidir PIO looks like. Setting all 8 bits to
     * output costs nothing if the PIO turns out to be output-only: on those,
     * offset 0x4 is the interrupt mask, and enabling interrupts on a PIO with
     * no IRQ connected has no effect. */
    volatile uint32_t *led_dir = (volatile uint32_t *)(base + led_offset + 4);
    *led_dir = 0xFF;
    printf("direction register (+0x4) set to 0xFF for output\n");

    /* Read-back check. A peripheral that reads as 0xFFFFFFFF usually means
     * nothing is mapped at that offset -- wrong address, or the FPGA is not
     * programmed. */
    uint32_t initial = *leds;
    printf("initial read: 0x%08X%s\n", initial,
           (initial == 0xFFFFFFFF) ? "   <-- suspicious: check the offset and "
                                     "that the .rbf/.sof is loaded" : "");

    printf("walking a single lit LED across the row, 10 passes...\n");
    for (int pass = 0; pass < 10; pass++) {
        for (int i = 0; i < 8; i++) {
            *leds = (1u << i);
            usleep(80000);
        }
    }

    /* Write-then-read confirms the bridge works in both directions, which a
     * blinking LED alone does not. */
    *leds = 0xA5;
    uint32_t readback = *leds & 0xFF;
    printf("wrote 0xA5, read back 0x%02X -- %s\n", readback,
           (readback == 0xA5) ? "PASS, the bridge is working both ways"
                              : "MISMATCH (some PIOs are write-only, so this "
                                "is not necessarily a fault)");

    *leds = 0;
    munmap((void *)base, LW_BRIDGE_SPAN);
    close(fd);
    return 0;
}
