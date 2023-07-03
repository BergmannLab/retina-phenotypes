#!/bin/bash

echo Raw images
ls /NVME/decrypted/ukbb/fundus/raw/CLRIS | wc -l

echo Raw subjects
ls /NVME/decrypted/ukbb/fundus/raw/CLRIS | cut -f1 -d_ | sort | uniq | wc -l

echo Images after QC
wc -l /NVME/decrypted/scratch/multitrait/UK_BIOBANK_PREPRINT/qc/zekavat_centercropped_corrected_percentile25.txt

echo Subjects after QC
cat /NVME/decrypted/scratch/multitrait/UK_BIOBANK_PREPRINT/qc/zekavat_centercropped_corrected_percentile25.txt | cut -f1 -d_ | sort | uniq | wc -l

echo Images with optic disc
sed '/,,,,,,/d' /NVME/decrypted/scratch/multitrait/UK_BIOBANK_PREPRINT/optic_disc/od_all.csv | wc -l

echo Subjects with optic disc
sed '/,,,,,,/d' /NVME/decrypted/scratch/multitrait/UK_BIOBANK_PREPRINT/optic_disc/od_all.csv | cut -f1 -d_ | sort | uniq | wc -l

echo Subjects per trait, phenotypic
python3.7 -c "import pandas as pd; df=pd.read_csv(\"/NVME/decrypted/scratch/multitrait/UK_BIOBANK_PREPRINT/participant_phenotype/2022_11_23_covar_fix_raw.csv\", usecols=[\"mean_angle_taa\",\"mean_angle_tva\",\"tau1_vein\",\"tau1_artery\",\"ratio_AV_DF\",\"eq_CRAE\",\"ratio_CRAE_CRVE\",\"D_A_std\",\"D_V_std\",\"eq_CRVE\",\"ratio_VD\",\"VD_orig_artery\",\"bifurcations\",\"VD_orig_vein\",\"medianDiameter_artery\",\"medianDiameter_vein\",\"ratio_AV_medianDiameter\"]); df.columns=[\"A temporal angle\",\"V temporal angle\",\"V tortuosity\",\"A tortuosity\",\"ratio tortuosity\",\"A central retinal eq\",\"ratio central retinal eq\",\"A std diameter\",\"V std diameter\",\"V central retinal eq\",\"ratio vascular density\",\"A vascular density\",\"bifurcations\",\"V vascular density\",\"A median diameter\",\"V median diameter\",\"ratio median diameter\"]; print(df.columns), print(df.notna().sum().values)"

echo Subjects per trait, genetic
python3.7 -c "import pandas as pd; df=pd.read_csv(\"/NVME/decrypted/scratch/multitrait/UK_BIOBANK_PREPRINT/gwas/2022_11_23_covar_fix/sample_sizes.txt\", sep=\" \", usecols=[\"mean_angle_taa\",\"mean_angle_tva\",\"tau1_vein\",\"tau1_artery\",\"ratio_AV_DF\",\"eq_CRAE\",\"ratio_CRAE_CRVE\",\"D_A_std\",\"D_V_std\",\"eq_CRVE\",\"ratio_VD\",\"VD_orig_artery\",\"bifurcations\",\"VD_orig_vein\",\"medianDiameter_artery\",\"medianDiameter_vein\",\"ratio_AV_medianDiameter\"]); df.columns=[\"A temporal angle\",\"V temporal angle\",\"V tortuosity\",\"A tortuosity\",\"ratio tortuosity\",\"A central retinal eq\",\"ratio central retinal eq\",\"A std diameter\",\"V std diameter\",\"V central retinal eq\",\"ratio vascular density\",\"A vascular density\",\"bifurcations\",\"V vascular density\",\"A median diameter\",\"V median diameter\",\"ratio median diameter\"]; print(df.T[0].values)"