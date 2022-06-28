#!/bin/bash
#SBATCH --account=sbergman_retina
#SBATCH --job-name=statsToPhenofile
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 32GB
#SBATCH --partition normal
#SBATCH --time 00-01:00:00

source ../../configs/config_.sh

python3.8 image_to_participant.py $PHENOFILE_ID $QC $IMAGE_PHENO_DIR/ $PARTICIPANT_PHENO_DIR/
