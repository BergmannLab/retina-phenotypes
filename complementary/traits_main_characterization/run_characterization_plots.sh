#!/bin/bash

source ../../configs/config_.sh
begin=$(date +%s)

#### 1) Violin plots, seaborn, histograms plots

nice python3.8 main_phenotypes_plots.py $phenofiles_dir_both $name_phenofile $save_dist_dir $MAIN_LABELS $plot_violin $plot_histograms $plot_seaborn $MAIN_NAMES



#### 2) PCA


echo FINISHED
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"

