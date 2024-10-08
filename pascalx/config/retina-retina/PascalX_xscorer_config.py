## PascalX parameters
gpu = True
n_cpu = 8
## Set side to test 'coherence', 'anti-coherence' or both
direction = ['coherence', 'anti-coherence']
## Perform sample overlap correction
overlap_corr = True

## Directories
refpanel_dir = '/NVME/decrypted/scratch/uk10k_hg19/uk10k'
genome_dir = '/NVME/scratch/olga/data/reference/pascal/GRCh37_protein_coding_lincRNA.txt'
gwas_a_dir = '/NVME/decrypted/scratch/multitrait/UK_BIOBANK_PREPRINT/gwas/2022_11_23_config_fix/'
gwas_b_dir = '/NVME/decrypted/scratch/multitrait/UK_BIOBANK_PREPRINT/gwas/2022_11_23_config_fix/'
ldsc_res_dir = gwas_a_dir # for sample overlap correction
ldsc_res_fmt = 'A-B__gc.log' # for sample overlap correction
out_dir = '/NVME/scratch/olga/output/PascalX/xscorer/retina-retina/2022-11-25/'
keepfile = None

## List of phenotypes for GWAS A
list_a = ['tau1_artery', 'tau1_vein', 'ratio_AV_DF', 'D_A_std', 'D_V_std', 'bifurcations', 'VD_orig_artery', 'VD_orig_vein', 'ratio_VD', 'mean_angle_taa', 'mean_angle_tva', 'eq_CRAE', 'eq_CRVE', 'ratio_CRAE_CRVE', 'medianDiameter_artery', 'medianDiameter_vein', 'ratio_AV_medianDiameter', 'D_CVMe_A', 'D_CVMe_V', 'D_CVMe', 'sd_mean_size', 'D_median_CVMe']
fmt_a = '__gwas_sumstats.tsv'
## List of phenotypes for GWAS B
list_b = ['tau1_artery', 'tau1_vein', 'ratio_AV_DF', 'D_A_std', 'D_V_std', 'bifurcations', 'VD_orig_artery', 'VD_orig_vein', 'ratio_VD', 'mean_angle_taa', 'mean_angle_tva', 'eq_CRAE', 'eq_CRVE', 'ratio_CRAE_CRVE', 'medianDiameter_artery', 'medianDiameter_vein', 'ratio_AV_medianDiameter', 'D_CVMe_A', 'D_CVMe_V', 'D_CVMe', 'sd_mean_size', 'D_median_CVMe']
fmt_b = '__gwas_sumstats.tsv'

## rscol, pcol, bcol, a1col, a2col and delimiter in GWAS A and B files
rscol_a = 0
pcol_a = 3
bcol_a = 4
a1col_a = 2
a2col_a = 1
del_a = '\t'

rscol_b = 0
pcol_b = 3
bcol_b = 4
a1col_b = 2
a2col_b = 1
del_b = '\t'
