## PascalX parameters
maf = 0.001
gpu = True
n_cpu = 8
## Set side to test 'coherence', 'anti-coherence' or both
direction = ['coherence', 'anti-coherence']
## Perform sample overlap correction
overlap_corr = True

## Directories
refpanel_dir = '/NVME/decrypted/scratch/uk10k_hg19/uk10k'
genome_dir = '/NVME/scratch/olga/data/reference/pascal/GRCh37_protein_coding_lincRNA.txt'
gwas_a_dir = '/NVME/decrypted/scratch/multitrait/UK_BIOBANK_PREPRINT/gwas/2022_11_23_covar_fix/'
gwas_b_dir = '/HDD/data/ukbb/disease_sumstats/VARIANTS/'
ldsc_res_dir = '/NVME/decrypted/scratch/multitrait/UK_BIOBANK_PREPRINT/gwas/gcorr_diseases/'
ldsc_res_fmt = 'A__munged.sumstats.gz_B.ldsc.imputed_v3.both_sexes.tsv.log'
out_dir = '/NVME/scratch/olga/output/PascalX/xscorer/retina-disease/2023-02-13/'
keepfile = None

## List of phenotypes for GWAS A
# Retinal traits ordered according to phenotypic correlation clustering (from general CONFIG file)
traits = 'mean_angle_taa,mean_angle_tva,tau1_vein,tau1_artery,ratio_AV_DF,eq_CRAE,ratio_CRAE_CRVE,D_A_std,D_V_std,eq_CRVE,ratio_VD,VD_orig_artery,bifurcations,VD_orig_vein,medianDiameter_artery,medianDiameter_vein,ratio_AV_medianDiameter'.split(',')
with open ('/NVME/decrypted/scratch/multitrait/UK_BIOBANK_PREPRINT/participant_phenotype/2022_11_23_covar_fix__labels_order.csv', 'r') as f:
    for r in f: #there's only one row
        order = [int(i) for i in r.split(',')]

list_a = [traits[i] for i in order]
pfx_a = '' # gwas file prefix
sfx_a = '__gwas_sumstats.tsv' # gwas file suffix

## List of phenotypes for GWAS B
list_b = ['4079_irnt', '4080_irnt', '6150_4', '102_irnt', '21021_irnt', '30760_irnt', '30780_irnt', '30870_irnt', '2443', '20002_1094', '6150_1', '6150_2', '6150_3', '6148_4', '20116_0', '20116_2', '1558']
pfx_b = 'qc_only_rs_notna_mod_variants_' # gwas file prefix
sfx_b = '.gwas.imputed_v3.both_sexes.tsv' # gwas file suffix

## rscol, pcol, bcol, a1col, a2col and delimiter in GWAS A and B files
rscol_a_name = 'rsid'
pcol_a_name = 'P'
bcol_a_name = 'beta'
a1col_a_name = 'A2'
a2col_a_name = 'A1'
del_a = '\t'

rscol_b_name = 'rsid'
pcol_b_name = 'pval'
bcol_b_name = 'beta'
a1col_b_name = 'alt'
a2col_b_name = 'ref'
del_b = '\t'
