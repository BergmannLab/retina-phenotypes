#!/bin/bash

source ../../configs/config_.sh
begin=$(date +%s)

#### Different time points plot

nice python3.8 main_time_points.py $IMAGE_PHENO_DIR $FIGURES_DIR $MAIN_LABELS "$MAIN_NAMES"



echo FINISHED
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"