#!/bin/bash

######################################################################################################
# GOAL: From -> vessels positions, diameters, OD information, segments and AV classification
#       To -> Main and complementary measurements (or phenotypes). 
#       Including temporal angles, CRAE, CRVE, diameter variability, Fractal dimensionality, etc
#       Computed in a parallel fashion to speed the process
#
# INPUT: Vessels and OD information 
# OUTPUT: Phenotypes values (per image)
# PROGRAMMING LANGUAGES USED: Python
######################################################################################################

source ../configs/config_.sh
mkdir -p $phenotypes_dir

#echo $ALL_IMAGES $phenotypes_dir $dir_ARIA_output $classification_output_dir $OD_FILE $PHENOTYPE_OF_INTEREST

nice $python_dir measurePhenotype.py $ALL_IMAGES $phenotypes_dir/ $dir_ARIA_output/ $classification_output_dir/ $OD_FILE $PHENOTYPE_OF_INTEREST $delta $R_0 $min_ta $max_ta $lower_accept $upper_accept $norm_acceptance $neighborhood_cte $limit_diameter_main $mask_radius &
wait
