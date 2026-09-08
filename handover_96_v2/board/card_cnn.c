/* ---------------------------------------------------------------------------
 * card_cnn.c  --  HPS-side driver for the FPGA card recognition accelerator
 *
 * LAB 5 runs its inference from a bare-metal application. This project runs
 * embedded Linux on the HPS, so the accelerator is reached by mmap-ing
 * /dev/mem over the lightweight HPS-to-FPGA bridge.
 *
 * Build (in the SoC EDS embedded command shell):
 *     arm-linux-gnueabihf-gcc -O2 -o card_cnn card_cnn.c
 *
 * Run on the board as root (mmap of /dev/mem requires it):
 *     ./card_cnn image.pgm
 *
 * FIXED POINT: the fabric works in Q6.10, so a pixel value v in [0,1] is
 * sent as round(v * 1024). Q4.12 was tried first and saturated fc_shared,
 * costing 6.6% accuracy -- see fpga/README.md. Feeding raw 0..255 bytes instead would overdrive
 * every activation by 255x and saturate the network -- the images must be
 * normalised the same way the PyTorch transform normalises them.
 * ------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

/* Cyclone V lightweight HPS-to-FPGA bridge */
#define LW_BRIDGE_BASE   0xFF200000
#define LW_BRIDGE_SPAN   0x00200000

/* Base offset of the accelerator within the bridge.
 *
 * THIS MUST MATCH the address Platform Designer assigned to
 * card_cnn_avalon_0 -- read it from the Address Map tab. A wrong value does
 * not error: the program reads zeros from empty address space, STATUS never
 * shows done, and it reports a timeout that looks exactly like an
 * unprogrammed FPGA.
 *
 * 0x00020000 is what the component was moved to after it overlapped the PIOs
 * at 0x0 and the LED PIO at 0x10040. Verify against your own address map. */
#define CNN_BASE         0x00020000

#define REG_CONTROL      0x0000
#define REG_STATUS       0x0004
#define REG_RANK         0x0008
#define REG_SUIT         0x000C
#define REG_JOKER        0x0010
#define REG_SCORE        0x0014
/* BYTE offset of the image window.
 *
 * card_cnn_avalon.v selects it with avs_address[12], and the slave is
 * configured in WORD address units -- so the window begins at word 0x1000,
 * which is BYTE 0x4000, not 0x2000. Writing at 0x2000 puts image data in the
 * register space where it is silently discarded. */
#define IMAGE_BASE       0x4000

#define CTRL_START       (1u << 0)
#define CTRL_COL_OVR_EN  (1u << 1)
#define CTRL_COL_OVR_VAL (1u << 2)

#define STATUS_DONE      (1u << 0)
#define STATUS_BUSY      (1u << 1)

#define IMG_DIM          48
#define IMG_PIXELS       (IMG_DIM * IMG_DIM)
#define Q_ONE            1024   /* 1.0 in Q6.10 */

static const char *RANK_NAMES[13] = {
    "2","3","4","5","6","7","8","9","10","J","Q","K","A"
};
static const char *SUIT_NAMES[4] = { "Spades", "Clubs", "Hearts", "Diamonds" };

static volatile uint8_t *lw_base;

static inline void reg_wr(uint32_t off, uint32_t val) {
    *(volatile uint32_t *)(lw_base + CNN_BASE + off) = val;
}
static inline uint32_t reg_rd(uint32_t off) {
    return *(volatile uint32_t *)(lw_base + CNN_BASE + off);
}

/* Upload a 48x48 image. `pixels` holds 0..255 grayscale; it is converted to
 * Q4.12 in [0,1] here, matching transforms.ToTensor() in the training
 * pipeline. */
static void upload_image(const uint8_t *pixels) {
    for (int i = 0; i < IMG_PIXELS; i++) {
        int16_t q = (int16_t)((pixels[i] * Q_ONE) / 255);
        *(volatile uint32_t *)(lw_base + CNN_BASE + IMAGE_BASE + i * 4) = (uint32_t)q;
    }
}

/* Returns 0 on success, -1 on timeout. The full chain is roughly 46 ms at
 * 50 MHz, so a 500 ms ceiling is generous without hanging forever if the
 * accelerator never asserts done. */
static int run_inference(int colour_is_red) {
    uint32_t ctrl = CTRL_START | CTRL_COL_OVR_EN;
    if (colour_is_red) ctrl |= CTRL_COL_OVR_VAL;
    reg_wr(REG_CONTROL, ctrl);

    for (int ms = 0; ms < 500; ms++) {
        if (reg_rd(REG_STATUS) & STATUS_DONE) return 0;
        usleep(1000);
    }
    return -1;
}

/* Minimal binary PGM (P5) reader, 48x48 expected. */
static int load_pgm(const char *path, uint8_t *out) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return -1; }

    char magic[3] = {0};
    int w = 0, h = 0, maxv = 0;
    if (fscanf(f, "%2s %d %d %d", magic, &w, &h, &maxv) != 4 ||
        strcmp(magic, "P5") != 0) {
        fprintf(stderr, "not a binary PGM (P5)\n");
        fclose(f); return -1;
    }
    if (w != IMG_DIM || h != IMG_DIM) {
        fprintf(stderr, "image is %dx%d, expected %dx%d -- the accelerator's "
                        "dimensions are fixed at synthesis time\n",
                w, h, IMG_DIM, IMG_DIM);
        fclose(f); return -1;
    }
    fgetc(f);                                   /* single whitespace byte */
    size_t n = fread(out, 1, IMG_PIXELS, f);
    fclose(f);
    if (n != IMG_PIXELS) { fprintf(stderr, "short read\n"); return -1; }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <48x48.pgm> [--red|--black]\n", argv[0]);
        return 1;
    }
    int colour_is_red = 0;
    for (int i = 2; i < argc; i++)
        if (strcmp(argv[i], "--red") == 0) colour_is_red = 1;

    uint8_t pixels[IMG_PIXELS];
    if (load_pgm(argv[1], pixels) != 0) return 1;

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (run as root)"); return 1; }

    lw_base = mmap(NULL, LW_BRIDGE_SPAN, PROT_READ | PROT_WRITE,
                   MAP_SHARED, fd, LW_BRIDGE_BASE);
    if (lw_base == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    upload_image(pixels);

    if (run_inference(colour_is_red) != 0) {
        fprintf(stderr, "timeout waiting for done -- check that the FPGA is "
                        "programmed and the base address matches Platform Designer\n");
        munmap((void *)lw_base, LW_BRIDGE_SPAN); close(fd); return 1;
    }

    uint32_t rank  = reg_rd(REG_RANK)  & 0xF;
    uint32_t suit  = reg_rd(REG_SUIT)  & 0x3;
    uint32_t joker = reg_rd(REG_JOKER) & 0x1;
    int32_t  score = (int32_t)reg_rd(REG_SCORE);

    if (joker) {
        printf("Joker\n");
    } else if (rank < 13) {
        printf("%s of %s\n", RANK_NAMES[rank], SUIT_NAMES[suit]);
        printf("  rank logit: %.4f (Q6.10 raw %d)\n", score / 1024.0, score);
    } else {
        printf("invalid rank index %u\n", rank);
    }

    munmap((void *)lw_base, LW_BRIDGE_SPAN);
    close(fd);
    return 0;
}
