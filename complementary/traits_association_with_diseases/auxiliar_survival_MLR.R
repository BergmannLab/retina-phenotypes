# Auxiliary functions

#install.packages("tidyverse")
#install.packages("Hmisc")
library(tidyverse)
library(Hmisc)
#install.packages("Hmisc")

read_disease_data <- function(ukbb_files_dir) 
{ 
  ### Read Ukbb files:
  data_1 <- read.csv(file= paste(ukbb_files_dir, "/1_data_extraction/ukb34181.csv", sep=""), header = TRUE, sep=",",check.names=FALSE)
  gwas_covar <- data_1
  gwas_covar <- gwas_covar[, c('21022-0.0', '31-0.0', '22009-0.1', '22009-0.2', '22009-0.3', '22009-0.4','22009-0.5', '22009-0.6', '22009-0.7', '22009-0.8', '22009-0.9', '22009-0.10', '22009-0.11', '22009-0.12', '22009-0.13', '22009-0.14', '22009-0.15', '22009-0.16', '22009-0.17', '22009-0.18', '22009-0.19', '22009-0.20', 'eid')]
  gwas_covar <- gwas_covar %>% 
    rename(
        'age_recruitment'='21022-0.0',
        'sex'='31-0.0',
        'cov1'='22009-0.1', 
        'cov2'='22009-0.2', 
        'cov3'='22009-0.3', 
        'cov4'='22009-0.4', 
        'cov5'='22009-0.5', 
        'cov6'='22009-0.6', 
        'cov7'='22009-0.7', 
        'cov8'='22009-0.8', 
        'cov9'='22009-0.9',
        'cov10'='22009-0.10', 
        'cov11'='22009-0.11', 
        'cov12'='22009-0.12', 
        'cov13'='22009-0.13', 
        'cov14'='22009-0.14', 
        'cov15'='22009-0.15', 
        'cov16'='22009-0.16', 
        'cov17'='22009-0.17', 
        'cov18'='22009-0.18',
        'cov19'='22009-0.19', 
        'cov20'='22009-0.20'
    )
  # names(gwas_covar) <- c('age', 'sex', 'cov1', 'cov2', 'cov3', 'cov4', 'cov5', 'cov6', 'cov7', 'cov8', 'cov9', 'eid')
  data_1 <- data_1[, c('34-0.0','53-0.0', '53-1.0', '53-2.0', '21003-0.0', '21003-1.0','21003-2.0',
                       '4079-0.0', '4080-0.0', '4079-0.1', '4080-0.1',
                       '102-0.0', '102-0.1', '20022-0.0', '3436-0.0', '1558-0.0', '21021-0.0', '6177-0.0',
                       '40000-0.0', 'eid')]
  data_1 <- data_1 %>% 
    rename(
        'year_birth'='34-0.0',
        'date_center_00'='53-0.0',
        'date_center_10'='53-1.0',
        'date_center_20'='53-2.0',
        'age_center_00'='21003-0.0',
        'age_center_10'='21003-1.0',
        'age_center_20'='21003-2.0',
        'DBP_00'='4079-0.0', 
        'SBP_00'='4080-0.0', 
        'DBP_01'='4079-0.1', 
        'SBP_01'='4080-0.1',
        'PR_00'='102-0.0',
        'PR_01'='102-0.1',
        'birth_weight'='20022-0.0',
        'age_current_smoker_00'='3436-0.0',
        'alcohol_intake_frequency_00'='1558-0.0',
        'pulse_wave_arterial_stiffness_index_00'='21021-0.0',
        'medication_cholesterol_BP_diabetes_00'='6177-0.0',
        'date_death'='40000-0.0'
    )
  # names(data_1) <- c('age_diabetes', 'age_angina', 'age_heartattack', 'age_DVT', 'age_stroke', 'DBP', 'SBP', 'date_death',  'eid')
  
  data_2 <- read.csv(file= paste(ukbb_files_dir, "/2_data_extraction_BMI_height_IMT/ukb42432.csv", sep=""), header = TRUE, sep=",",check.names=FALSE)
  data_2 <- data_2[, c('21001-0.0', '3456-0.0', '20161-0.0', '5084-0.0', '5085-0.0', '5086-0.0', '5087-0.0', 'eid')]
  data_2 <- data_2 %>% 
    rename('BMI'='21001-0.0',
          'N_cigarettes_curr_daily'= '3456-0.0',
          'Pack_year_smok'='20161-0.0',
          'spherical_power_R'='5084-0.0', 
          'spherical_power_L'='5085-0.0', 
          'cylindrical_power_L'='5086-0.0', 
          'cylindrical_power_R'='5087-0.0'
          )
    
  #data_3 <- read.csv(file=paste(ukbb_files_dir, "/3_data_extraction_tinnitus/ukb42625.csv", sep=""), header = TRUE, sep=",",check.names=FALSE)
    
  data_4 <- read.csv(file= paste(ukbb_files_dir, "/4_data_extraction_alzheimer_covid/ukb44505_alzheimer_covid.csv", sep=""), header = TRUE, sep=",",check.names=FALSE)
  data_4 <- data_4[, c('42020-0.0', 'eid')]
  data_4 <- data_4 %>% 
    rename('date_AD'='42020-0.0')
    
    
  data_5 <- read.csv(file=paste(ukbb_files_dir, "/5_data_extraction_OCT_HDL_LDL/ukb46188.csv", sep=""), header = TRUE, sep=",",check.names=FALSE)
  data_5 <- data_5[, c('30760-0.0', '30780-0.0', '30870-0.0', 'eid')]
  data_5 <- data_5 %>% 
    rename('HDL_cholesterol'='30760-0.0',
         'LDL_direct'='30780-0.0',
         'Triglycerides'='30870-0.0'
        )

    
  # TO DO: Re rewite how to rename!
  data_6 <- read.csv(file= paste(ukbb_files_dir, "/6_data_extraction/ukb49907.csv", sep=""),header = TRUE, sep=",",check.names=FALSE)
  data_6 <- data_6[, c('30750-0.0', '2976-0.0', '2976-1.0','2976-2.0','2976-3.0', '3627-0.0', '3627-1.0', '3627-2.0', '3627-3.0', 
                       '3894-0.0', '3894-1.0', '3894-2.0', '3894-3.0', '4012-0.0','4012-1.0','4012-2.0','4012-3.0', 
                       '4056-0.0', '4056-1.0', '4056-2.0', '4056-3.0', '40007-0.0', '4689-0.0', '4689-1.0', '4689-2.0', '4689-3.0',
                       '4700-0.0', '4700-1.0','4700-2.0','4700-3.0', '5408-0.0', '5408-1.0','5408-2.0','5408-3.0',
                       '5610-0.0', '5610-1.0', '5610-2.0', '5610-3.0', '5832-0.0', '5832-1.0', '5832-2.0', '5832-3.0',
                       '5843-0.0', '5843-1.0', '5843-2.0', '5843-3.0', '5855-0.0','5855-1.0','5855-2.0','5855-3.0',
                       '5890-0.0','5890-1.0','5890-2.0','5890-3.0', '5945-0.0', '5945-1.0', '5945-2.0', '5945-3.0',
                       '6150-0.0', '6150-0.1', '6150-0.2','6150-0.3','6150-1.0','6150-1.1','6150-1.2',
                       '6150-1.3','6150-2.0','6150-2.1','6150-2.2', '6150-2.3',
                       '6152-0.0', '1717-0.0', '1747-0.0', '1845-0.0', '2946-0.0','4022-0.0', '4022-1.0', '4022-2.0', '4022-3.0', 'eid')]
  data_6 <- data_6 %>% 
    rename(
        'HbA1c'='30750-0.0',
        'age_diabetes_00'='2976-0.0',
        'age_diabetes_10'='2976-1.0', 
        'age_diabetes_20'='2976-2.0', 
        'age_diabetes_30'='2976-3.0',
        'age_angina_00'='3627-0.0', 
        'age_angina_10'='3627-1.0', 
        'age_angina_20'='3627-2.0', 
        'age_angina_30'='3627-3.0', 
        'age_heartattack_00'='3894-0.0', 
        'age_heartattack_10'='3894-1.0', 
        'age_heartattack_20'='3894-2.0', 
        'age_heartattack_30'='3894-3.0', 
        'age_DVT_00'='4012-0.0', 
        'age_DVT_10'='4012-1.0',
        'age_DVT_20'='4012-2.0',
        'age_DVT_30'='4012-3.0',
        'age_stroke_00'='4056-0.0', 
        'age_stroke_10'='4056-1.0', 
        'age_stroke_20'='4056-2.0',
        'age_stroke_30'='4056-3.0', 
        'age_death'='40007-0.0', 
        'age_glaucoma_00'='4689-0.0',
        'age_glaucoma_10'='4689-1.0',
        'age_glaucoma_20'='4689-2.0',
        'age_glaucoma_30'='4689-3.0',
        'age_cataract_00'='4700-0.0', 
        'age_cataract_10'='4700-1.0',
        'age_cataract_20'='4700-2.0',
        'age_cataract_30'='4700-3.0',
        'eye_amblyopia_00'='5408-0.0', 
        'eye_amblyopia_10'='5408-1.0', 
        'eye_amblyopia_20'='5408-2.0',
        'eye_amblyopia_30'='5408-3.0', 
        'eye_presbyopia_00'='5610-0.0', 
        'eye_presbyopia_10'='5610-1.0',
        'eye_presbyopia_20'='5610-2.0',
        'eye_presbyopia_30'='5610-3.0',
        'eye_hypermetropia_00'='5832-0.0',
        'eye_hypermetropia_10'='5832-1.0',
        'eye_hypermetropia_20'='5832-2.0',
        'eye_hypermetropia_30'='5832-3.0',
        'eye_myopia_00'='5843-0.0', 
        'eye_myopia_10'='5843-1.0', 
        'eye_myopia_20'='5843-2.0', 
        'eye_myopia_30'='5843-3.0', 
        'eye_astigmatism_00'='5855-0.0', 
        'eye_astigmatism_10'='5855-1.0', 
        'eye_astigmatism_20'='5855-2.0', 
        'eye_astigmatism_30'='5855-3.0', 
        'eye_diabetes_00'='5890-0.0', 
        'eye_diabetes_10'='5890-1.0', 
        'eye_diabetes_20'='5890-2.0', 
        'eye_diabetes_30'='5890-3.0', 
        'age_other_serious_eye_condition_00'='5945-0.0', 
        'age_other_serious_eye_condition_10'='5945-1.0', 
        'age_other_serious_eye_condition_20'='5945-2.0', 
        'age_other_serious_eye_condition_30'='5945-3.0', 
        'vascular_heart_problems_00'='6150-0.0', 
        'vascular_heart_problems_01'='6150-0.1', 
        'vascular_heart_problems_02'='6150-0.2', 
        'vascular_heart_problems_03'='6150-0.3', 
        'vascular_heart_problems_10'='6150-1.0', 
        'vascular_heart_problems_11'='6150-1.1', 
        'vascular_heart_problems_12'='6150-1.2', 
        'vascular_heart_problems_13'='6150-1.3', 
        'vascular_heart_problems_20'='6150-2.0', 
        'vascular_heart_problems_21'='6150-2.1', 
        'vascular_heart_problems_22'='6150-2.2', 
        'vascular_heart_problems_23'='6150-2.3',
        '6152'='6152-0.0', 
        'skin_colour'='1717-0.0',  
        'hair_colour'='1747-0.0',
        'mothers_Age'='1845-0.0',
        'fathers_Age'='2946-0.0',
        'age_pulmonary_embolism_00'='4022-0.0',
        'age_pulmonary_embolism_10'='4022-1.0',
        'age_pulmonary_embolism_20'='4022-2.0',
        'age_pulmonary_embolism_30'='4022-3.0'
    )

  
  # data_7 <- read.csv("/.../7_data_extraction/ukb50488.csv", header = TRUE, sep=",",check.names=FALSE)
  
  data_8 <- read.csv(file= paste(ukbb_files_dir, "/8_data_extraction/ukb51076.csv", sep=""),  header = TRUE, sep=",",check.names=FALSE)
  data_8 <- data_8[, c('20262-0.0', 'eid')]
  names(data_8)  <- c('myopia', 'eid')
  
  data_all = merge(gwas_covar, data_1, by = "eid") 
  data_all = merge(data_all, data_2, by = "eid") 
  data_all = merge(data_all, data_4, by = "eid") 
  data_all = merge(data_all, data_5, by = "eid") 
  data_all = merge(data_all, data_6, by = "eid") 
  data_all = merge(data_all, data_8, by = "eid") 
  
  return(data_all)
}



read_survival_data <- function(survival_data_dir) 
{ 
  ################# Read survival data ############################
  data_survival_cov <- read.csv(file= paste(survival_data_dir, "/pruebas_survival.csv", sep=""), 
                                header = TRUE, sep=",",check.names=FALSE)
  colnames(data_survival_cov)
  data_survival_cov <- data_survival_cov %>% 
    rename(
      'age'='21022-0.0',
      'sex'='31-0.0',
      'etnia'='21000-0.0' 
    )
  
  # plot histograms
  dev.new()
  pdf(file= paste(survival_output_dir, "/histogramas.pdf", sep=""))
  hist.data.frame(data_survival_cov)
  dev.off()

  return(data_survival_cov)
}

read_survival_data_GWAS <- function(survival_data_dir) 
{ 
  ################# Read survival data ############################
  data_survival_cov <- read.csv(file= paste(survival_data_dir, "/pruebas_survival.csv", sep=""), 
                                header = TRUE, sep=",",check.names=FALSE)
  colnames(data_survival_cov)
  data_survival_cov <- data_survival_cov %>% 
    rename(
      'age'='21022-0.0',
      'sex'='31-0.0',
      'cov1'='22009-0.1' , 
      'cov2'='22009-0.2', 
      'cov3'='22009-0.5', 
      'cov4'='22009-0.6', 
      'cov5'='22009-0.7', 
      'cov6'='22009-0.8', 
      'cov7'='22009-0.16', 
      'cov8'='22009-0.17', 
      'cov9'='22009-0.18'
    )
  
  # plot histograms
  #dev.new()
  #pdf(file= paste(survival_output_dir, "/histogramas.pdf", sep=""))
  #hist.data.frame(data_survival_cov)
  #dev.off()
  # names(data_survival_cov) <- c( 'eid', '40000-0.0', '40000-1.0', 'age', 'sex', 
  #                                'cov1', 'cov2', 'cov3', 'cov4', 'cov5', 'cov6', 
  #                                'cov7', 'cov8', 'cov9', 'year_death', 'death') 
  # "eid"        "40000-0.0"  "40000-1.0"  "21022-0.0"  "31-0.0"     "22009-0.1"  "22009-0.2"  "22009-0.5" 
  # "22009-0.6"  "22009-0.7"  "22009-0.8"  "22009-0.16" "22009-0.17" "22009-0.18" "year_death"       "death"
  
  # Lo uso para comprobar que funcione lo de pintar: dev.off() 
  #plot(rnorm(50), rnorm(50))
  return(data_survival_cov)
}


create_dataset <- function(data_cov, phenofiles_dir, eye_selected, year_selected, instance_selected) 
  { 
  ################# Read phenofiles data ############################

  pheno_N_bif <- read.csv(file= paste(phenofiles_dir, "/2022-06-08_bifurcations.csv", sep=""),
                          header = TRUE, sep=",",check.names=FALSE)
  pheno_tVA <- read.csv(file= paste(phenofiles_dir, "/2022-06-08_tva.csv", sep=""),
                        header = TRUE, sep=",",check.names=FALSE)
  pheno_tAA <- read.csv(file= paste(phenofiles_dir, "/2022-06-08_taa.csv", sep=""),
                        header = TRUE, sep=",",check.names=FALSE)
  pheno_ratios_CRAE_CRVE <- read.csv(file= paste(phenofiles_dir, "/2022-06-08_ratios_CRAE_CRVE.csv", sep=""),
                                header = TRUE, sep=",",check.names=FALSE)
  pheno_ratios_ARIA <- read.csv(file= paste(phenofiles_dir, "/2022-06-09_ratios_aria_phenotypes.csv", sep=""),
                                header = TRUE, sep=",",check.names=FALSE)
  pheno_dia_var <- read.csv(file= paste(phenofiles_dir, "/2022-06-08_diameter_variability.csv", sep=""),
                                header = TRUE, sep=",",check.names=FALSE)
  pheno_FD <- read.csv(file= paste(phenofiles_dir, "/2022-06-08_fractal_dimension.csv", sep=""),
                       header = TRUE, sep=",",check.names=FALSE)
  pheno_VD <- read.csv(file= paste(phenofiles_dir, "/2022-06-08_vascular_density.csv", sep=""),
                       header = TRUE, sep=",",check.names=FALSE)
  pheno_baseline <- read.csv(file= paste(phenofiles_dir, "/2022-06-08_baseline.csv", sep=""),
                       header = TRUE, sep=",",check.names=FALSE)
  
  # From images names to eids
  #colnames(pheno_ARIA)
  names(pheno_N_bif)[1] <- 'eid'
  names(pheno_tVA)[1] <- 'eid'
  names(pheno_tAA)[1] <- 'eid'
  names(pheno_ratios_CRAE_CRVE)[1] <- 'eid'
  names(pheno_ratios_ARIA)[1] <- 'eid'
  names(pheno_dia_var)[1] <- 'eid'
  names(pheno_FD)[1] <- 'eid'
  names(pheno_VD)[1] <- 'eid'
  names(pheno_baseline)[1] <- 'eid'
  
  pheno_N_bif[c('eid', 'image', 'year', 'instance')] <- str_split_fixed(pheno_N_bif$eid, '_', 4)
  pheno_tVA[c('eid', 'image', 'year', 'instance')] <- str_split_fixed(pheno_tVA$eid, '_', 4)
  pheno_tAA[c('eid', 'image', 'year', 'instance')] <- str_split_fixed(pheno_tAA$eid, '_', 4)
  pheno_ratios_CRAE_CRVE[c('eid', 'image', 'year', 'instance')] <- str_split_fixed(pheno_ratios_CRAE_CRVE$eid, '_', 4)
  pheno_ratios_ARIA[c('eid', 'image', 'year', 'instance')] <- str_split_fixed(pheno_ratios_ARIA$eid, '_', 4)
  pheno_dia_var[c('eid', 'image', 'year', 'instance')] <- str_split_fixed(pheno_dia_var$eid, '_', 4)
  pheno_FD[c('eid', 'image', 'year', 'instance')] <- str_split_fixed(pheno_FD$eid, '_', 4)
  pheno_VD[c('eid', 'image', 'year', 'instance')] <- str_split_fixed(pheno_VD$eid, '_', 4)
  pheno_baseline[c('eid', 'image', 'year', 'instance')] <- str_split_fixed(pheno_baseline$eid, '_', 4)
  
  print('nrow(pheno_N_bif) antes de nada')
  print(nrow(pheno_N_bif))
  
  # Only select 21015
  pheno_N_bif <- pheno_N_bif %>% group_by(image) %>% filter(image == eye_selected)
  pheno_tVA <- pheno_tVA %>% group_by(image) %>% filter(image == eye_selected)
  pheno_tAA <- pheno_tAA %>% group_by(image) %>% filter(image == eye_selected)
  pheno_ratios_CRAE_CRVE <- pheno_ratios_CRAE_CRVE %>% group_by(image) %>% filter(image == eye_selected)
  pheno_ratios_ARIA <- pheno_ratios_ARIA %>% group_by(image) %>% filter(image == eye_selected)
  pheno_dia_var <- pheno_dia_var %>% group_by(image) %>% filter(image == eye_selected)
  pheno_FD <- pheno_FD %>% group_by(image) %>% filter(image == eye_selected)
  pheno_VD <- pheno_VD %>% group_by(image) %>% filter(image == eye_selected)
  pheno_baseline <- pheno_baseline %>% group_by(image) %>% filter(image == eye_selected)
  
  print('nrow(pheno_N_bif) only ', str(eye_selected))
  print(nrow(pheno_N_bif))
    
  
  # Only select year 0 and instance 0
  pheno_N_bif <- pheno_N_bif %>% group_by(year) %>% filter(year == year_selected)
  pheno_tVA <- pheno_tVA %>% group_by(year) %>% filter(year == year_selected)
  pheno_tAA <- pheno_tAA %>% group_by(year) %>% filter(year == year_selected)
  pheno_ratios_CRAE_CRVE <- pheno_ratios_CRAE_CRVE %>% group_by(year) %>% filter(year == year_selected)
  pheno_ratios_ARIA <- pheno_ratios_ARIA %>% group_by(year) %>% filter(year == year_selected)
  pheno_dia_var <- pheno_dia_var %>% group_by(year) %>% filter(year == year_selected)
  pheno_FD <- pheno_FD %>% group_by(year) %>% filter(year == year_selected)
  pheno_VD <- pheno_VD %>% group_by(year) %>% filter(year == year_selected)
  pheno_baseline <- pheno_baseline %>% group_by(year) %>% filter(year == year_selected)
  
  print('nrow(pheno_N_bif) only', str(year_selected))
  print(nrow(pheno_N_bif))
    
  pheno_N_bif <- pheno_N_bif %>% group_by(instance) %>% filter(instance == instance_selected)
  pheno_tVA <- pheno_tVA %>% group_by(instance) %>% filter(instance == instance_selected)
  pheno_tAA <- pheno_tAA %>% group_by(instance) %>% filter(instance == instance_selected)
  pheno_ratios_CRAE_CRVE <- pheno_ratios_CRAE_CRVE %>% group_by(instance) %>% filter(instance == instance_selected)
  pheno_ratios_ARIA <- pheno_ratios_ARIA %>% group_by(instance) %>% filter(instance == instance_selected)
  pheno_dia_var <- pheno_dia_var %>% group_by(instance) %>% filter(instance == instance_selected)
  pheno_FD <- pheno_FD %>% group_by(instance) %>% filter(instance == instance_selected)
  pheno_VD <- pheno_VD %>% group_by(instance) %>% filter(instance == instance_selected)
  pheno_baseline <- pheno_baseline %>% group_by(instance) %>% filter(instance == instance_selected)
  
  print('nrow(pheno_N_bif) only ', +str(instance_selected) )
  print(nrow(pheno_N_bif))
  
  ################# Read phenofiles data ############################
  g = merge(pheno_N_bif, data_cov, by = "eid", no.dups = TRUE) 
  print(nrow(g))
  g = merge(g, pheno_tVA, by = "eid", no.dups = TRUE) 
  print(nrow(g))
  g = merge(g, pheno_tAA, by = "eid", no.dups = TRUE) # To do: avoid warnings!
  print(nrow(g))
  g = merge(g, pheno_ratios_CRAE_CRVE, by = "eid", no.dups = TRUE) 
  print(nrow(g))
  g = merge(g, pheno_ratios_ARIA, by = "eid", no.dups = TRUE) 
  print(nrow(g))
  g = merge(g, pheno_dia_var, by = "eid", no.dups = TRUE) 
  print(nrow(g))
  g = merge(g, pheno_FD, by = "eid", no.dups = TRUE)
  print(nrow(g))
  g = merge(g, pheno_VD, by = "eid", no.dups = TRUE)
  print(nrow(g))
  g = merge(g, pheno_baseline, by = "eid", no.dups = TRUE) 
  print(nrow(g))
  
  return(g)
  } 


create_dataset_both_eyes <- function(data_cov, phenofiles_dir, name_phenofile) 
  { 
  ################# Read phenofiles data ############################
  
  pheno_df <- read.csv(file= paste(phenofiles_dir, name_phenofile, sep=""),
                                               header = TRUE, sep=",",check.names=FALSE)
  names(pheno_df)[1] <- 'eid'
  g = merge(pheno_df, data_cov, by = "eid", no.dups = TRUE) 
  print(nrow(g))
  #   
  return(g)
                       
  # pheno_df <- read.csv(file= paste(phenofiles_dir, "participant_phenotype2022_06_21_multitrait_ventile2.csv", sep=""),
  #                         header = TRUE, sep=" ",check.names=FALSE)
  # 
  # 
  # pheno_eid <- read.csv(file= paste(phenofiles_dir, "participant_phenotype2022_06_21_multitrait_ventile2_instances.csv", sep=""),
  #                       header = TRUE, sep=",",check.names=FALSE)
  # print('nrow(pheno_df) y pheno_eid antes de nada')
  # print(nrow(pheno_df))
  # print(nrow(pheno_eid))
  # 
  # if(nrow(pheno_df)== nrow(pheno_eid)) 
  # {   
  #   names(pheno_eid)[1] <- 'eid'
  #   pheno_df$eid <- pheno_eid$eid
  #   print(nrow(pheno_df))
  #   
  #   ################# Read phenofiles data ############################
  #   g = merge(pheno_df, data_cov, by = "eid", no.dups = TRUE) 
  #   print(nrow(g))
  #   
  #   return(g)
  #   }
  # else print("Error, different sizes!")
  } 
