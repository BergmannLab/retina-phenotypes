#!/bin/bash

source ../../configs/config_.sh

python3.8 image_to_participant.py $QC $IMAGE_PHENO_DIR/ $PARTICIPANT_PHENO_DIR/ $SAMPLE_FILE $PARTICIPANT_STAT_ID

begin=$(date +%s)


### Run 
Rscript  $config_dir/../complementary/image_to_participant/main_create_csv_diseases_covariants.R $ukbb_files_dir $phenofiles_dir_both $diseases_pheno_cov_file $name_phenofile $csv_name

echo FINISHED
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"
