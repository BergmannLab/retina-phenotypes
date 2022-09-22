#!/bin/bash

echo Start `basename $0`
date

source ../../configs/config_.sh

# common 2022_07_08_ventile5

nice Rscript cox_full.R $RUN_DIR $PARTICIPANT_STAT_ID FALSE
# args:
# 1) run ID
# 2) corrected trait file, using specific QC
# 3) only consider instance 0 measurements


#### run MLR
nice python3.8 MLR_diseases.py $VENTILE $What_type_phenotype $diseases_pheno_cov_file $csv_name $PARTICIPANT_PHENO_DIR $csv_z_name $SUPPLEMENTARY_LABELS $MAIN_LABELS

date
echo End `basename $0`
