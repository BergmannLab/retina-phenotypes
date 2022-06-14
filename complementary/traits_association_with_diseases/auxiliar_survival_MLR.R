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
        'age'='21022-0.0',
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
  data_1 <- data_1[, c('2976-0.0', '3627-0.0', '3894-0.0', '4012-0.0', '4056-0.0', '4079-0.0', '4080-0.0', '4079-0.1', '4080-0.1',
                       '102-0.0', '102-0.1', '20022-0.0', '3436-0.0', '1558-0.0', '21021-0.0', '6177-0.0',
                       '40000-0.0', 'eid')]
  data_1 <- data_1 %>% 
    rename(
        'age_diabetes'='2976-0.0', 
        'age_angina'='3627-0.0', 
        'age_heartattack'='3894-0.0', 
        'age_DVT'='4012-0.0', 
        'age_stroke'='4056-0.0', 
        'DBP1'='4079-0.0', 
        'SBP1'='4080-0.0', 
        'DBP2'='4079-0.1', 
        'SBP2'='4080-0.1',
        'PR1'='102-0.0',
        'PR2'='102-0.1',
        'birth_weight'='20022-0.0',
        'age_current_smoker'='3436-0.0',
        'alcohol_intake_frequency'='1558-0.0',
        'pulse_wave_arterial_stiffness_index'='21021-0.0',
        'medication_cholesterol_BP_diabetes'='6177-0.0',
        'date_death'='40000-0.0'
    )
  # names(data_1) <- c('age_diabetes', 'age_angina', 'age_heartattack', 'age_DVT', 'age_stroke', 'DBP', 'SBP', 'date_death',  'eid')
  
  data_2 <- read.csv(file= paste(ukbb_files_dir, "/2_data_extraction_BMI_height_IMT/ukb42432.csv", sep=""), header = TRUE, sep=",",check.names=FALSE)
  data_2 <- data_2[, c('21001-0.0', '3456-0.0', '20161-0.0', 'eid')]
  data_2 <- data_2 %>% 
    rename('BMI'='21001-0.0',
          'N_cigarettes_curr_daily'= '3456-0.0',
          'Pack_year_smok'='20161-0.0'
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
  data_6 <- data_6[, c('30750-0.0', '40007-0.0', '4689-0.0', '4700-0.0', '5408-0.0', '5610-0.0', '5832-0.0', 
                       '5843-0.0', '5855-0.0', '5890-0.0', '5945-0.0', '6150-0.0', '6152-0.0', '1717-0.0', '1747-0.0',
                       '1845-0.0', '2946-0.0','4022-0.0', 'eid')]
  data_6 <- data_6 %>% 
    rename(
        'HbA1c'='30750-0.0', 
        'age_death'='40007-0.0', 
        'age_glaucoma'='4689-0.0', 
        'age_cataract'='4700-0.0', 
        'eye_amblyopia'='5408-0.0', 
        'eye_presbyopia'='5610-0.0', 
        'eye_hypermetropia'='5832-0.0',
        'eye_myopia'='5843-0.0', 
        'eye_astigmatism'='5855-0.0', 
        'eye_diabetes'='5890-0.0', 
        'age_other_serious_eye_condition'='5945-0.0', 
        'vascular_heart_problems'='6150-0.0', 
        '6152'='6152-0.0', 
        'skin_colour'='1717-0.0',  
        'hair_colour'='1747-0.0',
        'mothers_Age'='1845-0.0',
        'fathers_Age'='2946-0.0',
        'age_pulmonary_embolism'='4022-0.0'
    )

  
  # data_7 <- read.csv("/.../7_data_extraction/ukb50488.csv", header = TRUE, sep=",",check.names=FALSE)
  
  data_8 <- read.csv(file= paste(ukbb_files_dir, "/8_data_extraction/ukb51076.csv", sep=""),  header = TRUE, sep=",",check.names=FALSE)
  data_8 <- data_8[, c('20262-0.0', 'eid')]
  names(data_8)  <- c('myopia', 'eid')
  
  data_all = merge(gwas_covar, data_1, by = "eid") 
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


create_dataset <- function(data_cov, phenofiles_dir) 
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
  pheno_N_bif <- pheno_N_bif %>% group_by(image) %>% filter(image == 21015)
  pheno_tVA <- pheno_tVA %>% group_by(image) %>% filter(image == 21015)
  pheno_tAA <- pheno_tAA %>% group_by(image) %>% filter(image == 21015)
  pheno_ratios_CRAE_CRVE <- pheno_ratios_CRAE_CRVE %>% group_by(image) %>% filter(image == 21015)
  pheno_ratios_ARIA <- pheno_ratios_ARIA %>% group_by(image) %>% filter(image == 21015)
  pheno_dia_var <- pheno_dia_var %>% group_by(image) %>% filter(image == 21015)
  pheno_FD <- pheno_FD %>% group_by(image) %>% filter(image == 21015)
  pheno_VD <- pheno_VD %>% group_by(image) %>% filter(image == 21015)
  pheno_baseline <- pheno_baseline %>% group_by(image) %>% filter(image == 21015)
  
  print('nrow(pheno_N_bif) solo 21015')
  print(nrow(pheno_N_bif))
    
  
  # Only select year 0 and instance 0
  pheno_N_bif <- pheno_N_bif %>% group_by(year) %>% filter(year == 0)
  pheno_tVA <- pheno_tVA %>% group_by(year) %>% filter(year == 0)
  pheno_tAA <- pheno_tAA %>% group_by(year) %>% filter(year == 0)
  pheno_ratios_CRAE_CRVE <- pheno_ratios_CRAE_CRVE %>% group_by(year) %>% filter(year == 0)
  pheno_ratios_ARIA <- pheno_ratios_ARIA %>% group_by(year) %>% filter(year == 0)
  pheno_dia_var <- pheno_dia_var %>% group_by(year) %>% filter(year == 0)
  pheno_FD <- pheno_FD %>% group_by(year) %>% filter(year == 0)
  pheno_VD <- pheno_VD %>% group_by(year) %>% filter(year == 0)
  pheno_baseline <- pheno_baseline %>% group_by(year) %>% filter(year == 0)
  
  print('nrow(pheno_N_bif) solo 0')
  print(nrow(pheno_N_bif))
    
  pheno_N_bif <- pheno_N_bif %>% group_by(instance) %>% filter(instance == '0.png')
  pheno_tVA <- pheno_tVA %>% group_by(instance) %>% filter(instance == '0.png')
  pheno_tAA <- pheno_tAA %>% group_by(instance) %>% filter(instance == '0.png')
  pheno_ratios_CRAE_CRVE <- pheno_ratios_CRAE_CRVE %>% group_by(instance) %>% filter(instance == '0.png')
  pheno_ratios_ARIA <- pheno_ratios_ARIA %>% group_by(instance) %>% filter(instance == '0.png')
  pheno_dia_var <- pheno_dia_var %>% group_by(instance) %>% filter(instance == '0.png')
  pheno_FD <- pheno_FD %>% group_by(instance) %>% filter(instance == '0.png')
  pheno_VD <- pheno_VD %>% group_by(instance) %>% filter(instance == '0.png')
  pheno_baseline <- pheno_baseline %>% group_by(instance) %>% filter(instance == '0.png')
  
  print('nrow(pheno_N_bif) solo 0.png')
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


