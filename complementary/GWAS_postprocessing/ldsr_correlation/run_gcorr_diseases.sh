#!/bin/bash

source ../../../configs/config_.sh
begin=$(date +%s)

####
 
#cp $diseases_ss_raw * $diseases_gwas_dir

nice python3.8 gcorr_diseases_ldsr.py $FIGURES_DIR $MAIN_LABELS "$MAIN_NAMES" $diseases_gwas_dir $What_type_phenotype $SUPPLEMENTARY_LABELS "$SUPPLEMENTARY_NAMES"


echo FINISHED
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"