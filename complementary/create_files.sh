#### Read the vairables requiered from config.sh:
source ../configs/config_.sh
begin=$(date +%s)

#mkdir -p $RUN_DIR

### Initialization (TO DO, add to the config_):
ukbb_files_dir='/NVME/decrypted/ukbb/labels/'
####phenofiles_dir='/NVME/decrypted/multitrait/image_phenotype/collection/'
phenofiles_dir_both='/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/participant_phenotype/' #'/NVME/decrypted/multitrait/diseases/Diseases_cov_phenotypes_both_eyes_'
diseases_pheno_cov_file='/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/diseases_cov/'
name_phenofile="2022_07_08_ventile5_raw_with_instance.csv"
csv_name='2022_07_08_ventile5_diseases_cov_test'

### Run 
Rscript $config_dir/../complementary/traits_association_with_diseases/main_create_csv_diseases_phenotypes_covariants.R $ukbb_files_dir $phenofiles_dir_both $diseases_pheno_cov_file $name_phenofile $csv_name

echo FINISHED
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"
