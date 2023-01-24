library(stringr)
library(dplyr)
library(tidyr)

### Initialization:
myargs = commandArgs(trailingOnly=TRUE)

ukbb_files_dir <- myargs[1] #'/NVME/decrypted/ukbb/labels/'
####phenofiles_dir <- '/NVME/decrypted/multitrait/image_phenotype/collection/'
phenofiles_dir_both <- myargs[2] #'/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/participant_phenotype/' 
#'/NVME/decrypted/multitrait/diseases/Diseases_cov_phenotypes_both_eyes_'
diseases_pheno_cov_file <- myargs[3] #'/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/diseases_cov/'
name_phenofile <- myargs[4] #"2022_07_08_ventile2_raw_with_instance.csv"
csv_name <- myargs[5] #'2022_07_08_ventile2_diseases_cov'

## Create the phenofiles+diseases+covariants file ########################################
### Read disease data:
source("~/retina-phenotypes/complementary/image_to_participant/auxiliar_create_csv.R") 
data_aux = read_disease_data(ukbb_files_dir) ## You can change the diseases on it

### Read phenofile data and create data set:
## For file with only one eye:
#data_all = create_dataset(data_aux, phenofiles_dir, 21015, 0, '0.png') ## You can change the phenofiles on it

## For file with both eyes:
data_all = create_dataset_both_eyes(data_aux, phenofiles_dir_both, name_phenofile)

colnames(data_all)

## Average columns: 
data_all$SBP_00 <- apply(select(data_all, SBP_00, SBP_01),1,function(x) mean(na.omit(x)))
data_all$DBP_00 <- apply(select(data_all, DBP_00, DBP_01),1,function(x) mean(na.omit(x)))
data_all$PR_00 <- apply(select(data_all, PR_00, PR_01),1,function(x) mean(na.omit(x)))

data_all$SBP_10 <- apply(select(data_all, SBP_10, SBP_11),1,function(x) mean(na.omit(x)))
data_all$DBP_10 <- apply(select(data_all, DBP_10, DBP_11),1,function(x) mean(na.omit(x)))
data_all$PR_10 <- apply(select(data_all, PR_10, PR_11),1,function(x) mean(na.omit(x)))

## not included yet, but, BASED ON FLORENCE RECOMENDATION: TAKE THE MEDIAN OF THE 3 FIRST FOR SPHERICAL AND CYLINDRICAL 
data_all$spherical_power_00 <- apply(select(data_all, spherical_power_R_00, spherical_power_L_00),1,function(x) mean(na.omit(x)))
data_all$spherical_power_10 <- apply(select(data_all, spherical_power_R_10, spherical_power_L_10),1,function(x) mean(na.omit(x)))
data_all$cylindrical_power_00 <- apply(select(data_all, cylindrical_power_R_00, cylindrical_power_L_00),1,function(x) mean(na.omit(x)))
data_all$cylindrical_power_10 <- apply(select(data_all, cylindrical_power_R_10, cylindrical_power_L_10),1,function(x) mean(na.omit(x)))
                                     
#data_all["SBP"]= (data_all["SBP_00"]+data_all["SBP_01"])/2
#data_all["DBP"]= (data_all["DBP_00"]+data_all["DBP_01"])/2
#data_all["PR"]= (data_all["PR_00"]+data_all["PR_01"])/2

#data_all["spherical_power_00"]= (data_all["spherical_power_R_00"]+data_all["spherical_power_L_00"])/2
#data_all["spherical_power_10"]= (data_all["spherical_power_R_10"]+data_all["spherical_power_L_10"])/2

#data_all["cylindrical_power_00"]= (data_all["cylindrical_power_R_00"]+data_all["cylindrical_power_L_00"])/2
#data_all["cylindrical_power_10"]= (data_all["cylindrical_power_R_10"]+data_all["cylindrical_power_L_10"])/2


## Covariants select by instances
#data_all$instance2 <- data_all$instance
#data_all$instance2 <- ifelse(data_all$instance2 == 0, NaN, 1)

#data_all["age_center"] = data_all["age_center_10"]/data_all["instance2"] 
#data_all$age_center <- ifelse(is.na(data_all$age_center), data_all$age_center_00, data_all$age_center)
#data_all$age_center <- apply(select(data_all, age_center_00, age_center_10),1,function(x) mean(na.omit(x)))
#data_all["age_center_2"]= data_all["age_center"]^2
#data_all["age_center_3"]= data_all["age_center"]^3

#data_all["spherical_power"] = data_all["spherical_power_10"]/data_all["instance2"] 
#data_all$spherical_power <- ifelse(is.na(data_all$spherical_power), data_all$spherical_power_00, data_all$spherical_power)
#data_all$spherical_power_0 <- apply(select(data_all, spherical_power_00, spherical_power_10),1,function(x) mean(na.omit(x)))
#data_all["spherical_power_0_2"]= data_all["spherical_power"]^2
#data_all["spherical_power_0_3"]= data_all["spherical_power"]^3

#data_all["cylindrical_power"] = data_all["cylindrical_power_10"]/data_all["instance2"] 
#data_all$cylindrical_power <- ifelse(is.na(data_all$cylindrical_power), data_all$cylindrical_power_00, data_all$cylindrical_power)
#data_all$cylindrical_power_0 <- apply(select(data_all, cylindrical_power_00, cylindrical_power_10),1,function(x) mean(na.omit(x)))
#data_all["cylindrical_power_0_2"]= data_all["cylindrical_power"]^2
#data_all["cylindrical_power_0_3"]= data_all["cylindrical_power"]^3

#data_all$instance2 <- NULL

print('Number of nans of PC2, PC5, sex, age_center, spherical_power and cylindrical_power')
print(sum(is.na(data_all$PC2)))
print(sum(is.na(data_all$PC5)))
print(sum(is.na(data_all$sex)))
#print(sum(is.na(data_all$age_center)))
print(sum(is.na(data_all$spherical_power_10)))
print(sum(is.na(data_all$spherical_power_00)))
print(sum(is.na(data_all$cylindrical_power_00)))
print(sum(is.na(data_all$cylindrical_power_10)))
                                       
write.csv(data_all, paste(diseases_pheno_cov_file, csv_name, sep="") , row.names = FALSE)
#write.csv(data_all, paste('/SSD/home/sofia/retina-phenotypes/complementary/image_to_participant/', csv_name, sep="") , row.names = FALSE)
