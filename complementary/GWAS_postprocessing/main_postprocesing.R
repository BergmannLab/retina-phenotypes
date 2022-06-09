setwd("/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/GWAS/2022_06_08_all_phenotypes_LWNet_Decile3/")



pheno_list <- c("bifurcations", "median_CRAE", "eq_CRAE", "median_CRVE", "eq_CRVE", "ratio_median_CRAE_CRVE", "ratio_CRAE_CRVE", 
"medianDiameter_artery", "medianDiameter_vein", "DF_artery", "DF_vein", "ratio_AV_medianDiameter", "ratio_VA_medianDiameter", 
"ratio_AV_DF", "ratio_VA_DF", "mean_angle_taa", "mean_angle_tva", "D_median_std", "D_mean_std", "D_std_std",
"D_A_median_std", "D_A_mean_std", "D_A_std_std", "D_V_median_std", "D_V_mean_std", "D_V_std_std", "VD_orig_all",
"VD_orig_artery", "VD_orig_vein", "VD_small_all", "VD_small_artery", "VD_small_vein", "std_intensity", "mean_intensity", "median_intensity")

source("/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/GWAS/plotGWAS.R")
foo_plotGWAS(pheno_list)

source("/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/GWAS/hit_to_csv.R")
foo_hit_to_csv(pheno_list)

#source("/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/GWAS/prepare_input.R")
#foo_prepare_input(pheno_list)

#LD_prune_all should be analyze separated
#source("LD_prune_all.R")
