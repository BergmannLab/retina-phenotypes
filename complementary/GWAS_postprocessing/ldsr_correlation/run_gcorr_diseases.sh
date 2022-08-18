#!/bin/bash

source ../../../configs/config_.sh
begin=$(date +%s)

####
 
#cp $diseases_ss_raw * $diseases_gwas_dir

nice python3.8 main_diseases_ldsr_output.py $VENTILE $FIGURES_DIR $MAIN_LABELS $MAIN_LABELS $diseases_gwas_dir


echo FINISHED
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"