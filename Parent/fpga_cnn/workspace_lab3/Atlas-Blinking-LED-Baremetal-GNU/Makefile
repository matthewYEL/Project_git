#*****************************************************************************
#
# Card CNN on the DE10-Nano: bare-metal Cortex-A9 application.
#
# Two builds come out of this one Makefile, selected by RTOS_MODE in
# app_config.h:
#
#   RTOS_MODE 0   the original single-threaded polling app (atlas_main.c only)
#   RTOS_MODE 1   FreeRTOS on CPU0 (three tasks) + a bare-metal worker on CPU1
#
# The FreeRTOS sources are compiled unconditionally -- they cost build time,
# not run time, and keeping the source list static means switching RTOS_MODE
# cannot break the build.
#
# Requires SOCEDS_DEST_ROOT, so start Arm DS from the Intel Embedded Command
# Shell.
#
#*****************************************************************************

COMPILER := GNU
#COMPILER := ARMCC

MEMORY ?= ddr
#MEMORY ?= ocr

SEMIHOSTED ?= 1
#SEMIHOSTED ?= 0

ADD_CFLAGS_GNU :=
ADD_CFLAGS_ARMCC :=

FREERTOS      := FreeRTOS-Kernel
FREERTOS_PORT := $(FREERTOS)/portable/GCC/ARM_CA9

# Basenames only: Makefile.inc turns each into $(OBJ_DIR)/<base>.o, so a path
# here would ask for an object in a subdirectory that does not exist. The
# directories go on VPATH below instead.
APP_SRC       := atlas_main.c app_rtos.c a9_startup.c amp.c
FREERTOS_SRC  := tasks.c queue.c list.c timers.c port.c heap_4.c

EXAMPLE_SRC := $(APP_SRC) $(FREERTOS_SRC)

ELF ?= atlas_main.axf

# alt_clock_manager gives the real mpu_periph_clk for the RTOS tick, instead
# of assuming one.
HWLIBS_SRC := alt_clock_manager.c

include Makefile.inc

# --- additions that must come after the include, because Makefile.inc uses
# --- immediate (:=) assignment for all of these.

VPATH  := $(VPATH):$(FREERTOS):$(FREERTOS_PORT):$(FREERTOS)/portable/MemMang
CFLAGS += -I. -I$(FREERTOS)/include -I$(FREERTOS_PORT)

# Header dependency tracking. Makefile.inc's rules depend only on the .c file
# and the makefiles, so before this a change to app_config.h, card_pipeline.h
# or FreeRTOSConfig.h left every object stale and the build silently kept the
# old settings -- including ZOOM_*, where a stale value means confidently wrong
# card readings rather than an obvious failure.
CFLAGS += -MMD -MP
-include $(OBJ_DIR)/*.d

# portASM.S is built by hand rather than through EXAMPLE_SRC.
#
# Makefile.inc carries a "$(OBJ_DIR)/%.o: %.s" rule whose $(AS)/$(ASMFLAGS) are
# only ever set in its ARMCC branch. Windows paths are case-insensitive, so
# that lower-case rule matches portASM.S, fires an empty recipe, and make then
# believes the object is up to date without ever creating it -- a silent
# failure that surfaces as a missing file at link time. Giving the object a
# name no pattern rule can claim avoids the whole collision.
FREERTOS_ASM_OBJ := $(OBJ_DIR)/freertos_portasm.o
OBJ += $(FREERTOS_ASM_OBJ)
$(ELF): $(FREERTOS_ASM_OBJ)

$(FREERTOS_ASM_OBJ): $(FREERTOS_PORT)/portASM.S FreeRTOSConfig.h Makefile Makefile.inc
	$(CC) $(CFLAGS) -c $< -o $@
