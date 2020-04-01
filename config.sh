# CONFIG FILE for Retina pipeline

# RAW IMAGE DATA
ARIA_data_dir=/data/FAC/FBM/DBC/sbergman/retina/UKBiob/fundus/REVIEW/
raw_data_dirf=$ARIA_data_dir/CLRIS/

# empirically determined quality threshold relative to vessel stat "length__TOT"
quality_thr=11000

# BuildTestDatasetHypertension
# number of hypertension cases in dataset 
# (twice as many controls will be added)
limit=1000

# config GPU usage for DL
# gpuid=-1 for CPU
gpuid=-1

# backups
archive=/archive/unilcbg/mtomason/ 
# location of raw data, software, and permanent pipeline outputs
data=/data/FAC/FBM/DBC/sbergman/
# location of scratch folder (all pipeline outputs and code)
scratch=/scratch/beegfs/FAC/FBM/DBC/sbergman/
# MATLAB
matlab_runtime=/software/Development/Languages/Matlab_Compiler_Runtime/96
# BGENIE
bgenie_dir=$data/retina/software/bgenie
# ARIA (compiled)
ARIA_dir=$data/retina/software/ARIA

