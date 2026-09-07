/*
 * Card-recognition version of the lab's atlas_main.c.
 *
 * Same hardware flow as the original (capture -> read 784 px -> normalize ->
 * load CNN -> poll result). Only difference: the CNN now has 14 card classes
 * instead of 10 digits, so the result index is looked up in CARD_LABELS.
 *
 * Drop this in place of atlas_main.c in the Arm DS project and rebuild.
 * Requires the FPGA to be programmed with the 14-class bitstream
 * (fc_layer.v / argmax.v / cnn_core.v widened + card .hex weights).
 */
#include <stdio.h>
#include "socal.h"
#include "hps_0_arm_a9_0.h"
#include <stdlib.h>

int __auto_semihosting;

#define IMG_WR_CTRL_PIO_BASE      (0xFF200000)
#define IMG_WR_DATA_PIO_BASE      (0xFF200010)
#define IMG_WR_ADDR_PIO_BASE      (0xFF200020)
#define CAMERA_TRIGGER_PIO_BASE   (0xFF200030)
#define SNAPSHOT_DATA_PIO_BASE    (0xFF200040)
#define SNAPSHOT_ADDR_PIO_BASE    (0xFF200050)
#define CNN_START_PIO_BASE        (0xFF200060)
#define CNN_RESULT_PIO_BASE       (0xFF200070)

/* Class index -> card rank. Order is torchvision ImageFolder (alphabetical),
 * matching class_labels.txt produced by train_cards.py. */
static const char *const CARD_LABELS[14] = {
    "10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "J", "Joker", "K", "Q"
};

/* The original contrast stretch was tuned for pen-on-paper digits. Set to 0 if
 * it hurts recognition on printed cards. */
#define USE_CONTRAST_ENHANCE 1

int main(int argc, char** argv)
{
    int i, j;
    unsigned int val;
    short pixels[784];

    printf("Triggering camera capture...\n");
    alt_write_word(CAMERA_TRIGGER_PIO_BASE, 1);
    alt_write_word(CAMERA_TRIGGER_PIO_BASE, 0);
    for (volatile int d = 0; d < 5000000; d++);

    printf("Reading snapshot...\n");
    for (i = 0; i < 784; i++) {
        alt_write_word(SNAPSHOT_ADDR_PIO_BASE, i);
        for (volatile int d = 0; d < 10; d++);
        val = alt_read_word(SNAPSHOT_DATA_PIO_BASE);
        pixels[i] = (short)(val & 0xFFFF);
    }

    printf("\nCaptured 28x28 image (raw):\n");
    for (i = 0; i < 28; i++) {
        for (j = 0; j < 28; j++) {
            int p = pixels[i*28 + j];
            char c;
            if (p < 500)       c = '.';
            else if (p < 1500) c = '-';
            else if (p < 2500) c = '+';
            else if (p < 3500) c = '#';
            else               c = '@';
            printf("%c", c);
        }
        printf("\n");
    }
    printf("\n");

    int minv = pixels[0], maxv = pixels[0];
    for (i = 0; i < 784; i++) {
        if (pixels[i] < minv) minv = pixels[i];
        if (pixels[i] > maxv) maxv = pixels[i];
    }
    printf("Min value: %d, Max value: %d\n", minv, maxv);

    int range = maxv - minv;
    if (range < 1) range = 1;

    short normalized[784];
    for (i = 0; i < 784; i++) {
        int v = ((pixels[i] - minv) * 4096) / range;
        if (v < 0) v = 0;
        if (v > 4095) v = 4095;
        normalized[i] = (short)v;
    }

    short enhanced[784];
    for (i = 0; i < 784; i++) {
#if USE_CONTRAST_ENHANCE
        int v = normalized[i];
        if (v > 2048) {
            v = 2048 + (v - 2048) * 2;
            if (v > 4095) v = 4095;
        } else {
            v = v / 3;
        }
        enhanced[i] = (short)v;
#else
        enhanced[i] = normalized[i];
#endif
    }

    printf("\nEnhanced 28x28 image:\n");
    for (i = 0; i < 28; i++) {
        for (j = 0; j < 28; j++) {
            int p = enhanced[i*28 + j];
            char c;
            if (p < 500)       c = '.';
            else if (p < 1500) c = '-';
            else if (p < 2500) c = '+';
            else if (p < 3500) c = '#';
            else               c = '@';
            printf("%c", c);
        }
        printf("\n");
    }
    printf("\n");

    printf("Loading image into CNN...\n");
    for (i = 0; i < 784; i++) {
        alt_write_word(IMG_WR_ADDR_PIO_BASE, i);
        alt_write_word(IMG_WR_DATA_PIO_BASE, (unsigned int)(unsigned short)enhanced[i]);
        alt_write_word(IMG_WR_CTRL_PIO_BASE, 0x1);
        for (volatile int d = 0; d < 2000; d++);
        alt_write_word(IMG_WR_CTRL_PIO_BASE, 0x0);
        for (volatile int d = 0; d < 2000; d++);
    }

    alt_write_word(IMG_WR_CTRL_PIO_BASE, 0x2);
    alt_write_word(CNN_START_PIO_BASE, 1);
    alt_write_word(CNN_START_PIO_BASE, 0);

    printf("Running inference...\n");
    unsigned int result, predicted_class, done;
    do {
        result = alt_read_word(CNN_RESULT_PIO_BASE);
        done = (result >> 4) & 0x1;
    } while (!done);
    predicted_class = result & 0xF;

    if (predicted_class < 14)
        printf("\n=== Predicted card: %s  (class %u) ===\n",
               CARD_LABELS[predicted_class], predicted_class);
    else
        printf("\n=== Invalid class index %u ===\n", predicted_class);

    exit(0);
}
