#!/bin/bash

source ../configs/config_.sh
mkdir -p $phenotypes_dir

nice $python_dir measurePhenotype.py $ALL_IMAGES $phenotypes_dir $dir_ARIA_output $classification_output_dir $OD_FILE $PHENOTYPE_OF_INTEREST &
