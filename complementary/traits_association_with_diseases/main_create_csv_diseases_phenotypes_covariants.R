library(stringr)
library(dplyr)
library(tidyr)

### Initialization:
#ukbb_files_dir <- '/NVME/decrypted/ukbb/labels/'
####phenofiles_dir <- '/NVME/decrypted/multitrait/image_phenotype/collection/'
#diseases_pheno_cov_file <- '/NVME/decrypted/multitrait/diseases/Diseases_cov_phenotypes_both_eyes_'
#phenofiles_dir_both <- '/NVME/decrypted/multitrait/participant_phenotype/'
#name_phenofile <-"2022_06_27_multitrait_full_ventile2_with_ids.csv"
#QC_name<-'Ventile2'

## Create the phenofiles+diseases+covariants file ########################################
### Read disease data:
source("~/retina-phenotypes/complementary/traits_association_with_diseases/auxiliar_create_csv.R") 
data_aux = read_disease_data(ukbb_files_dir) ## You can change the diseases on it

### Read phenofile data and create data set:
## For file with only one eye:
#data_all = create_dataset(data_aux, phenofiles_dir, 21015, 0, '0.png') ## You can change the phenofiles on it

## For file with both eyes:
data_all = create_dataset_both_eyes(data_aux, phenofiles_dir_both, name_phenofile)

colnames(data_all)

write.csv(data_all, paste(diseases_pheno_cov_file+ QC_name+'.csv', sep="") , row.names = FALSE)
