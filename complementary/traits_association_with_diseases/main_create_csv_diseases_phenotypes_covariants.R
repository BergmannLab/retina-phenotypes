library(stringr)
library(dplyr)
library(tidyr)

### Initialization:
#ukbb_files_dir <- '/NVME/decrypted/ukbb/labels/'
####phenofiles_dir <- '/NVME/decrypted/multitrait/image_phenotype/collection/'
#diseases_pheno_cov_file <- '/NVME/decrypted/multitrait/diseases/Diseases_cov_phenotypes_both_eyes_'
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

## Average columns:
data_all["SBP"]= (data_all["SBP_00"]+data_all["SBP_01"])/2
data_all["DBP"]= (data_all["DBP_00"]+data_all["DBP_01"])/2
data_all["PR"]= (data_all["PR_00"]+data_all["PR_01"])/2

data_all["spherical_power_00"]= (data_all["spherical_power_R_00"]+data_all["spherical_power_L_00"])/2
data_all["spherical_power_10"]= (data_all["spherical_power_R_10"]+data_all["spherical_power_L_10"])/2

data_all["cylindrical_power_00"]= (data_all["cylindrical_power_R_00"]+data_all["cylindrical_power_L_00"])/2
data_all["cylindrical_power_10"]= (data_all["cylindrical_power_R_10"]+data_all["cylindrical_power_L_10"])/2


## Squared and cubic:
data_all["age_center_2_00"]= data_all["age_center_00"]^2
data_all["age_center_2_10"]= data_all["age_center_10"]^2
data_all["age_center_2_20"]= data_all["age_center_20"]^2

data_all["age_center_3_00"]= data_all["age_center_00"]^3
data_all["age_center_3_10"]= data_all["age_center_10"]^3
data_all["age_center_3_20"]= data_all["age_center_20"]^3

data_all["spherical_power_2_00"]= data_all["spherical_power_00"]^2
data_all["spherical_power_2_10"]= data_all["spherical_power_10"]^2
data_all["cylindrical_power_2_00"]= data_all["cylindrical_power_00"]^2
data_all["cylindrical_power_2_10"]= data_all["cylindrical_power_10"]^2

data_all["spherical_power_3_00"]= data_all["spherical_power_00"]^3
data_all["spherical_power_3_10"]= data_all["spherical_power_10"]^3
data_all["cylindrical_power_3_00"]= data_all["cylindrical_power_00"]^3
data_all["cylindrical_power_3_10"]= data_all["cylindrical_power_10"]^3


write.csv(data_all, paste(diseases_pheno_cov_file+ QC_name+'.csv', sep="") , row.names = FALSE)
