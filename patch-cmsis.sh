#!/usr/bin/env bash

echo "Replacing old cmsis-nn files ..."
# Issue 1: Mbed has a dependency to an old version of arm_math.h.
cp cmsis-nn_patch/DSP/arm_math.h mbed-os/cmsis/TARGET_CORTEX_M/arm_math.h


# Issue 2: There are definitions missing in cmsis_gcc.h.
cp cmsis-nn_patch/CORE/* mbed-os/cmsis/TARGET_CORTEX_M/

echo "Done."
