/* ---------------------------------------------------------------------------
 * probe_cnn.c  --  is the accelerator actually at this address?
 *
 * bridge_test only reads and writes ONE offset, which cannot tell an
 * accelerator apart from an aliased PIO. If writes to an unmapped address get
 * decoded onto some other peripheral, a write-then-read appears to succeed and
 * the LEDs even flash -- which looks like everything is fine.
 *
 * This reads all six accelerator registers and checks whether they behave the
 * way card_cnn_avalon.v defines them:
 *
 *   +0x00 CONTROL  W   reads back 0 (the read mux has no case for it)
 *   +0x04 STATUS   R   bit0 done, bit1 busy -- so 0, 1, 2 or 3, never large
 *   +0x08 RANK     R   0..12
 *   +0x0C SUIT     R   0..3
 *   +0x10 JOKER    R   0 or 1
 *   +0x14 SCORE    R   signed Q6.10
 *
 * Tell-tales:
 *   all six identical            -> aliased onto one register, wrong address
 *   all 0xFFFFFFFF               -> nothing mapped, or FPGA not programmed
 *   CONTROL reads back what you  -> a PIO, not this accelerator
 *     wrote
 *   RANK > 12, SUIT > 3          -> not this accelerator
 *
 * Build:  arm-linux-gnueabihf-gcc -O2 -static -o probe_cnn probe_cnn.c
 * Run:    ./probe_cnn 0x20000
 * ------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define LW_BASE 0xFF200000
#define LW_SPAN 0x00200000

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <base offset, e.g. 0x20000>\n", argv[0]);
        return 1;
    }
    uint32_t base = (uint32_t)strtoul(argv[1], NULL, 0);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); return 1; }

    volatile uint8_t *lw = mmap(NULL, LW_SPAN, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, LW_BASE);
    if (lw == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    volatile uint32_t *r = (volatile uint32_t *)(lw + base);
    const char *names[7] = { "CONTROL", "STATUS", "RANK", "SUIT", "JOKER", "SCORE", "ID" };

    printf("probing accelerator at +0x%05X\n\n", base);
    uint32_t v[7];
    for (int i = 0; i < 7; i++) {
        v[i] = r[i];
        printf("  +0x%02X  %-8s = 0x%08X  (%d)\n", i*4, names[i], v[i], (int)v[i]);
    }

    /* The ID register is the read-path test: it returns a constant, so if it
     * does not come back correct, nothing else read here can be trusted. */
    printf("\n");
    if (v[6] == 0xCA5D0001) {
        printf("READ PATH OK -- ID register returned its magic value.\n\n");
    } else {
        printf("*** READ PATH BROKEN ***\n");
        printf("    ID at +0x18 returned 0x%08X, expected 0xCA5D0001.\n", v[6]);
        printf("    avs_read is not reaching the component, or avs_readdata is\n");
        printf("    not wired back. Check the Component Editor Signals tab:\n");
        printf("      avs_read     -> signal type 'read'\n");
        printf("      avs_readdata -> signal type 'readdata'\n");
        printf("      both on interface avalon_slave_0\n");
        printf("    Every register value above is meaningless until this is\n");
        printf("    fixed.\n\n");
    }

    int all_ones = 1, all_zero = 1;
    for (int i = 0; i < 6; i++) {
        if (v[i] != 0xFFFFFFFF) all_ones = 0;
        if (v[i] != 0)          all_zero = 0;
    }

    printf("\n");
    if (all_ones) {
        printf("VERDICT: all 0xFFFFFFFF -- nothing mapped here, or the FPGA is\n"
               "         not programmed with your bitstream.\n");
        munmap((void *)lw, LW_SPAN); close(fd); return 0;
    }

    /* All zeros is NOT evidence of a problem: CONTROL is write-only so it
     * reads 0, STATUS is 0 before anything runs, and the result registers
     * reset to 0. That is precisely what a freshly reset accelerator looks
     * like. Unmapped Avalon addresses also read 0, so the two cases cannot be
     * told apart by reading -- only the start/done handshake distinguishes
     * them. */
    if (all_zero)
        printf("all zeros -- consistent with either a reset accelerator or an\n"
               "unmapped address. Only the handshake can tell them apart.\n\n");
    else if (v[2] > 12 || v[3] > 3 || v[1] > 3)
        printf("WARNING: RANK/SUIT/STATUS are outside their legal ranges\n"
               "         (RANK 0-12, SUIT 0-3, STATUS 0-3). Probably not the\n"
               "         accelerator, but testing the handshake anyway.\n\n");

    printf("writing CONTROL = start | colour_override...\n");
    r[0] = 0x1 | 0x2;

    /* BUSY is set by the Avalon write itself, in the same always block as
     * start_pulse, with no dependence on the core. So it separates two very
     * different failures:
     *
     *   busy never goes high -> the write is not decoding. The address is
     *                            wrong, or the component is not in the build.
     *   busy goes high, done -> the write works and start fired, but the core
     *     never follows        is stalled. Suspect clock or reset. */
    /* Read STATUS TWICE. If avs_readdata is registered but the interface is
     * configured for zero read latency, every read returns the PREVIOUS
     * read's data -- so the first read here returns CONTROL's value and the
     * second returns the real STATUS. Two different answers to the same
     * address is proof of that off-by-one. */
    uint32_t st1 = r[1];
    uint32_t st2 = r[1];
    printf("  STATUS read twice: 0x%08X then 0x%08X\n", st1, st2);
    if (st1 != st2) {
        printf("\n  *** Two reads of the SAME address returned different\n");
        printf("  *** values. avs_readdata is registered but the interface\n");
        printf("  *** expects it combinationally, so every read returns the\n");
        printf("  *** previous one's data.\n");
        printf("  *** FIX: Component Editor -> Interfaces tab -> the Avalon\n");
        printf("  *** slave -> set Read Wait = 1. Then regenerate and\n");
        printf("  *** recompile.\n\n");
    }
    uint32_t st = st2;
    printf("  STATUS (settled): 0x%08X  (busy=%d done=%d)\n",
           st, (st >> 1) & 1, st & 1);
    if (!((st >> 1) & 1) && !(st & 1)) {
        printf("\n  busy did NOT assert. The CONTROL write is not reaching the\n");
        printf("  component -- this is an addressing or build problem, not a\n");
        printf("  core problem. Check the Address Map, and grep the fitter\n");
        printf("  report for card_cnn_avalon.\n");
    } else {
        printf("  busy asserted -- the write decoded and start fired.\n");
    }
    printf("\npolling STATUS for done...\n");

    int done = 0, ms;
    for (ms = 0; ms < 500; ms++) {
        if (r[1] & 1) { done = 1; break; }
        usleep(1000);
    }

    if (done) {
        printf("\nVERDICT: done asserted after ~%d ms.\n", ms);
        printf("         The accelerator IS here and running.\n");
        printf("         (expect roughly 50 ms at 50 MHz)\n\n");
        printf("  RANK  = %d\n  SUIT  = %d\n  JOKER = %d\n  SCORE = %d\n",
               r[2] & 0xF, r[3] & 0x3, r[4] & 0x1, (int32_t)r[5]);
        printf("\nSet CNN_BASE to 0x%05X in card_cnn.c.\n", base);
    } else {
        printf("\nVERDICT: done never asserted after 500 ms.\n");
        printf("         (a full inference is ~50 ms, so this is a real stall)\n");
        printf("         Either nothing is mapped at 0x%05X, or the core is\n", base);
        printf("         present but stalled. Things to check, in order:\n");
        printf("           1. the Address Map tab -- what base did Platform\n");
        printf("              Designer actually assign to card_cnn_avalon_0?\n");
        printf("           2. the fitter report -- does it list card_cnn_avalon?\n");
        printf("              If the component dropped out of the build, no\n");
        printf("              address will work.\n");
        printf("           3. clock and reset connected to the component in\n");
        printf("              Platform Designer. Without a clock the FSM never\n");
        printf("              leaves idle and done stays low forever.\n");
    }

    munmap((void *)lw, LW_SPAN);
    close(fd);
    return 0;
}
