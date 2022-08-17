#!/bin/bash

source ../../configs/config_.sh
begin=$(date +%s)

#### 1) Violin plots, seaborn, histograms plots

nice python3.8 main_phenotypes_plots.py $phenofiles_dir_both $name_phenofile $FIGURES_DIR $MAIN_LABELS $MAIN_NAMES $What_type_phenotype $SUPPLEMENTARY_LABELS $SUPPLEMENTARY_NAMES $VENTILE



#### 2) PCA


echo FINISHED
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"

