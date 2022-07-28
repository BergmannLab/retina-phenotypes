#!/bin/bash

source ../../configs/config_.sh

# common 2022_07_08_ventile5

nice Rscript cox_full.R $RUN_DIR 2022_07_08_ventile2 FALSE #$PARTICIPANT_STAT_ID FALSE
# args:
# 1) run ID
# 2) corrected trait file, using specific QC
# 3) only consider instance 0 measurements
