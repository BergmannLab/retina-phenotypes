#!/bin/bash

echo Start `basename $0`
date

source ../../configs/config_.sh

# common 2022_07_08_ventile5

nice Rscript cox_full.R $RUN_DIR $PARTICIPANT_STAT_ID FALSE $csv_z_name $csv_diseases_cov_name
# args:
# 1) run ID
# 2) corrected trait file, using specific QC
# 3) only consider instance 0 measurements

#### run MLR
nice python3.8 MLR_diseases.py $What_type_phenotype $diseases_pheno_cov_file $csv_diseases_cov_name $PARTICIPANT_PHENO_DIR $csv_z_name $MAIN_LABELS $SUPPLEMENTARY_LABELS "$MAIN_NAMES" "$SUPPLEMENTARY_NAMES" 

date
echo End `basename $0`
