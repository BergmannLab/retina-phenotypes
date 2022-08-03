############################## CONFIG FILE for fundus multitrait project ##############################

#### PROJECT COMPUTATION DIRECTORY

PROJECT_DIR=/NVME/decrypted/scratch/multitrait

#### CONFIGURATIONS DIRECTORY

config_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # returns parent directory of this file

#### DATASET OF CHOICE

data_set=UK_BIOBANK #options: DRIVE, IOSTAR, CHASEDB1, UK_BIOBANK

#image_type options: *.jpg, *.tif, *.png, ...
if [[ data_set == UK_BIOBANK ]]; then
	image_type=*.png
elif [[ data_set == DRIVE ]]; then
	image_type=*.tif
else
	image_type=*.png
fi
# ARIA processor
# needs to be specified now already, for the repo to remain compatible with ARIA nomenclature:
# DRIVE has its own processor, UK_BIOBANK we allocate to CLRIS
if [[ $data_set == UK_BIOBANK ]]; then
        aria_processor=CLRIS
else
        aria_processor=$data_set
fi

#### PROJECT DATA

dir_images2=/NVME/decrypted/ukbb/fundus/raw/CLRIS/ # folder containing raw fundus images

#### UNIQUE RUN LABEL

RUN="$data_set"_ZERO

#### FOLDER STRUCTURE

RUN_DIR=$PROJECT_DIR/$RUN
dir_input=$RUN_DIR/input
dir_images=$dir_input/$aria_processor


folders=(
        AV_maps
        image_phenotype
        image_phenotype/current
        input
	gwas
        optic_disc
        participant_phenotype
        skeletons_etc
)

for i in "${folders[@]}"; do mkdir -p -m u=rwx,g=rwx,o=rx $RUN_DIR/$i; done # create folders if do not exist, exceptionally allow group write permissions, as this is a collaborative project

# One additional folder, required for ARIA
# If raw images are already in PNG format, dir_images is a symbolic link pointing to the raw images
if [[ "$image_type" = *.png ]]; then
        rm $dir_input/$aria_processor # clean if pipeline was run before
	ln -s $dir_images2 $dir_input/$aria_processor
else
        # clean, and regenerate directory
        rm -rf $dir_input/$aria_processor
        mkdir -p $dir_input/$aria_processor
fi

#### LIST OF RAW IMAGES

ls $dir_images > $dir_input/all_raw_images.txt
ALL_IMAGES=$dir_input/all_raw_images.txt
n_img=$(cat $ALL_IMAGES | wc -l)
echo Number of raw fundus images to be processed: $n_img

#### PIXEL-WISE, ARTERY-VEIN SPECIFIC VESSEL SEGMENTATION

classification_output_dir=$RUN_DIR/AV_maps

#### OPTIC DISC SEGMENTATION

OD_FILE=$RUN_DIR/optic_disc/od_all.csv

#### SKELETONIZATION, CUTTING VESSELS AT BRANCHINGS AND CROSSINGS

dir_ARIA_output=$RUN_DIR/skeletons_etc

TYPE_OF_VESSEL_OF_INTEREST=all # ARIA legacy parameter
AV_threshold=0.79 # segment score threshold

# quality thresholds for ARIA
min_QCthreshold_1=0
max_QCthreshold_1=9000000
min_QCthreshold_2=0
max_QCthreshold_2=100000

#### FUNDUS MEASUREMENTS

PHENOTYPE_OF_INTEREST='taa,tva,CRAE,CRVE,ratios_CRAE_CRVE,bifurcations,diameter_variability,aria_phenotypes,ratios,fractal_dimension,vascular_density,baseline,neo_vascularization,N_main_arteires,N_main_veins'
phenotypes_dir=$RUN_DIR/image_phenotype/

#### IMAGE MEASUREMENTS TO PARTICIPANT MEASUREMENTS

PARTICIPANT_STAT_ID=2022_07_08_ventile5
QC=/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles5.txt
PARTICIPANT_PHENO_DIR=$RUN_DIR/participant_phenotype/ # participant-level phenotyoes
IMAGE_PHENO_DIR=$phenotypes_dir/current # image-level phenotypes
SAMPLE_FILE=/NVME/decrypted/ukbb/fundus/ukb_imp_v3_subset_fundus.sample # file determining participant order for bgenie GWAS

# COVARIATES, DISEASES
ukbb_files_dir='/NVME/decrypted/ukbb/labels/'
phenofiles_dir_both='/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/participant_phenotype/' 
diseases_pheno_cov_file='/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/diseases_cov/'
name_phenofile="2022_07_08_ventile5_raw_with_instance.csv"
csv_name='2022_07_08_ventile5_diseases_cov'

##### SUPPLEMENTARY PHENOTYPES
# pass the unique labels as names
SUPPLEMENTARY_LABELS='DF_all,FD_all,VD_all,bifurcations'
SUPPLEMENTARY_NAMES='Distance factor,Fractal dimension, Bifurcations'

##### MAIN PHENOTYPES
MAIN_LABELS='DF_all,FD_all,VD_all,bifurcations'
MAIN_NAMES='Distance factor,Fractal dimension, Bifurcations'

#### PARALLEL COMPUTING

#to run one by one, set n_cpu=1
n_cpu=50
step_size=$((n_img/n_cpu))
batch_max=$((n_cpu * step_size))
remainder=$(( n_img - step_size * n_cpu ))

#### USER-SPECIFIC CONFIGURATIONS

source $config_dir/config_personal.sh

