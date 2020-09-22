#!/bin/bash -l
#SBATCH --account=sbergman_retina
#SBATCH --job-name=VesselStatsToPhenofile
#SBATCH --output=helpers/VesselStatsToPhenofile/slurm_runs/slurm-%x_%j.out
#SBATCH --error=helpers/VesselStatsToPhenofile/slurm_runs/slurm-%x_%j.err
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 1G
#SBATCH --time 00-02:00:00
#SBATCH --partition normal

source $HOME/retina/configs/config.sh

# define inputs and outputs
output_dir=$scratch/retina/GWAS/output/VesselStatsToPhenofile/
output=$output_dir/phenofile.csv

#UKBB
sample_file=$data/retina/UKBiob/genotypes/ukb43805_imp_chr1_v3_s487297.sample
stats_dir=$scratch/retina/preprocessing/output/backup/2020_08_10__18_33_09__116639_D9/

#SkiPOGH
#sample_file=$data/retina/SkiPOGH/genotypes/SkiPOGH.sample
#stats_dir=$scratch/retina/preprocessing/output/backup/2020_04_16__20_56_55__1133_SkiPOGH/

echo "Producing phenofile for vessel statistics for run: ${stats_dir: -22}"

# extract vessel stats phenotypes
source /dcsrsoft/spack/bin/setup_dcsrsoft
module purge
module load gcc/8.3.0
module load python/3.7.6
module load py-biopython
python3.7 $PWD/helpers/VesselStatsToPhenofile/run.py $output $sample_file $stats_dir
module purge

# NOTE: we have decided not to qqnorm here, but, rather, to qqnorm the residuals (after removing covars)
# qq normalize
#qq_input=$output
#qq_output=$output_dir/phenofile_qqnorm.csv
#source /dcsrsoft/spack/bin/setup_dcsrsoft
#module purge
#module load gcc/8.3.0
#module load r/3.6.2
#Rscript $PWD/helpers/utils/QQnorm/QQnormMatrix.R $qq_input $qq_output
#module purge

echo FINISHED: output has been written to $output_dir
