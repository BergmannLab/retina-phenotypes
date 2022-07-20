#!/bin/bash

source ../../configs/config_.sh

#### 1) Image to participant

nice python3.8 image_to_participant.py $QC $IMAGE_PHENO_DIR/ $PARTICIPANT_PHENO_DIR/ $SAMPLE_FILE $PARTICIPANT_STAT_ID


#### 2) Create disease & covar df

begin=$(date +%s)

nice Rscript  $config_dir/../complementary/image_to_participant/main_create_csv_diseases_covariants.R $ukbb_files_dir $phenofiles_dir_both $diseases_pheno_cov_file $name_phenofile $csv_name

echo FINISHED
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"

#### 3) Correct phenotypes for covariates

nice python3.8 correct_phenotype_two_step.py $RUN_DIR/ $PARTICIPANT_STAT_ID raw
nice python3.8 correct_phenotype_two_step.py $RUN_DIR/ $PARTICIPANT_STAT_ID qqnorm 
