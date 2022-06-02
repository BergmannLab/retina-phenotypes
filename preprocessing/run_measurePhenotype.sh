#!/bin/bash
#SBATCH --account=sbergman_retina
#SBATCH --job-name=measurePhenotype
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 50
#SBATCH --mem 200GB
#SBATCH --partition normal
#SBATCH --time 00-20:00:00

source ../configs/config_.sh
mkdir -p $phenotypes_dir

### For some traits it is needed to know the position of the Optic Disk (OD). For that we modified an existing code: 
# If you want to use the DRIVE dataset or for the UKBB we already provide the OD positions file. Otherwise you will need to provide your own OD position file (in the same format).
OD_file_dir=$dir_input

### Compute the trait (PHENOTYPE_OF_INTEREST) selected for all the images:
nohup $python_dir measurePhenotype.py $ALL_IMAGES $phenotypes_dir $dir_ARIA_output $classification_output_dir $OD_file_dir $PHENOTYPE_OF_INTEREST &
# Remark: putting measurements in scratch really makes a difference! (1.5 instead of 20 minutes -> 10x speedup!!)
