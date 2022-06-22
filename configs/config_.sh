############################## CONFIG FILE for Retina pipeline ##############################

config_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#### quality thresholds for ARIA
min_QCthreshold_1=0
max_QCthreshold_1=9000000
min_QCthreshold_2=0
max_QCthreshold_2=100000

#### QUALITY THRESHOLDS OF LWNET ARTERY/VEIN CLASSIFICATION: 
AV_threshold=0.79 # Add comment

#### configurations to adapt by user
source $config_dir/config_personal.sh

# IMAGE TO PARTICIPANT
PHENOFILE_ID=2022_06_21_multitrait_ventile2
QC=/HDD/data/ukbb/fundus/qc/ageCorrected_min1Percent_ventiles2.txt
PARTICIPANT_PHENO_DIR=/NVME/decrypted/multitrait/participant_phenotype # participant-level phenotyoes
IMAGE_PHENO_DIR=/NVME/decrypted/multitrait/image_phenotype/current # image-level phenotypes
