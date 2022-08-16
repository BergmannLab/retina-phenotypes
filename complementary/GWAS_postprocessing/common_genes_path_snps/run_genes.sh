#!/bin/bash

source ../../../configs/config_.sh
begin=$(date +%s)

####

nice python3.8 main_gene_analysis.py $VENTILE $genes_input $genes_results $What_type_phenotype $MAIN_LABELS $MAIN_NAMES $SUPPLEMENTARY_LABELS $SUPPLEMENTARY_NAMES




echo FINISHED
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"