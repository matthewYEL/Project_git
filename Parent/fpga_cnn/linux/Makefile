# Build the Linux card-CNN app.
#
# Intended to be built NATIVELY ON THE BOARD -- the Intel SoC EDS that ships
# with this project only provides arm-eabi (bare metal), not a
# arm-linux-gnueabihf cross toolchain, so there is nothing to cross-compile
# with on the development PC. The DE10-Nano Linux images all carry gcc:
#
#   scp -r linux/ root@<board>:/root/card_cnn
#   ssh root@<board> 'cd /root/card_cnn && make && ./card_cnn --stats'
#
# To cross-compile anyway, point CROSS_COMPILE at a hard-float ARM toolchain:
#   make CROSS_COMPILE=arm-linux-gnueabihf-

CROSS_COMPILE ?=
CC      := $(CROSS_COMPILE)gcc

TARGET  := card_cnn
SRC     := card_cnn_linux.c
HDR     := card_cnn_hw.h

# -O2 matters: the Monte-Carlo odds engine on CPU1 is the compute half of the
# demo, and at -O0 it is roughly 4x slower for no reason.
CFLAGS  ?= -O2 -Wall -Wextra -std=gnu99
LDFLAGS ?=
LDLIBS  := -lpthread

.PHONY: all clean check

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS) $(LDLIBS)

# Host-side check of the fixed-point front end against the offline model.
# Builds and runs with the native compiler; needs no board.
check: preprocess_check
	./preprocess_check

preprocess_check: preprocess_check.c $(HDR)
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET) preprocess_check
