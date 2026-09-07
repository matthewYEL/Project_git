#
# Copyright Altera 2013,2014,2015,2020
# All Rights Reserved.
#

C_SRC := hello.c  
MULTILIBFLAGS := -mfloat-abi=softfp -mtune=cortex-a9 -mcpu=cortex-a9  
CFLAGS := $(MULTILIBFLAGS) -g -O0 -Werror -Wall -Wstrict-prototypes

BOARD := cyclonev
ifeq ($(filter $(BOARD),arriav cyclonev),$(BOARD))
LINKER_SCRIPT := cvav-ocr.ld

else
ifeq ($(BOARD),arria10)
LINKER_SCRIPT := arria10-ocr.ld
else
$(error ERROR: BOARD setting "$(BOARD)" is invalid, please specify arria10 or cyclonev))
endif
endif

CROSS_COMPILE := arm-eabi-
CC := $(CROSS_COMPILE)gcc
LD := $(CROSS_COMPILE)g++
NM := $(CROSS_COMPILE)nm
OD := $(CROSS_COMPILE)objdump

ifeq ($(or $(COMSPEC),$(ComSpec)),)
RM := rm -rf
else
RM := cs-rm -rf
endif

ELF ?= $(basename $(firstword $(C_SRC))).axf
OBJ := $(patsubst %.c,%.o,$(C_SRC))

.PHONY: all
all: $(ELF)

.PHONY: clean
clean:
	$(RM) $(ELF) $(OBJ) *.objdump *.map

$(OBJ): %.o: %.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@

$(ELF): $(OBJ)
	$(LD) -T$(LINKER_SCRIPT) $(MULTILIBFLAGS) $(OBJ) -o $@
	$(OD) -d $@ > $@.objdump
	$(NM) $@ > $@.map
