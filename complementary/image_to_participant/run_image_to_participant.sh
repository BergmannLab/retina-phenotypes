#!/bin/bash

source ../../configs/config_.sh

#### 1) Image to participant

nice -n 1 python3.8 image_to_participant.py $QC $IMAGE_PHENO_DIR/ $PARTICIPANT_PHENO_DIR/ $SAMPLE_FILE $PARTICIPANT_STAT_ID $n_std

#### 2) Create disease & covar df

begin=$(date +%s)

nice -n 1 Rscript  $config_dir/../complementary/image_to_participant/main_create_csv_diseases_covariants.R $ukbb_files_dir $PARTICIPANT_PHENO_DIR $diseases_pheno_cov_file $name_phenofile $csv_name
nice -n 1 python3.8  $config_dir/../complementary/image_to_participant/disease_cov_csv.py $diseases_pheno_cov_file $csv_name

#### 3) Correct phenotypes for covariates

nice -n 1 python3.8 correct_phenotype_one_step.py $RUN_DIR/ $PARTICIPANT_STAT_ID z $MAIN_LABELS $MAIN_NAMES
nice -n 1 python3.8 correct_phenotype_one_step.py $RUN_DIR/ $PARTICIPANT_STAT_ID qqnorm $MAIN_LABELS $MAIN_NAMES

echo FINISHED
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"


