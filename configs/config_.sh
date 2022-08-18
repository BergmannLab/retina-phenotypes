############################## CONFIG FILE for fundus multitrait project ##############################

#### PROJECT COMPUTATION DIRECTORY

PROJECT_DIR=/NVME/decrypted/scratch/multitrait

#### CONFIGURATIONS DIRECTORY

config_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # returns parent directory of this file

#### DATASET OF CHOICE

data_set=UK_BIOBANK #options: DRIVE, IOSTAR, CHASEDB1, UK_BIOBANK

#image_type options: *.jpg, *.tif, *.png, ...
if [[ $data_set = UK_BIOBANK ]]; then
	image_type=*.png
elif [[ $data_set = DRIVE ]]; then
	image_type=*.tif
else
	image_type=*.png
fi

echo Data set: $data_set
echo Image type: $image_type

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
	diseases_cov
	figures
        image_phenotype
        image_phenotype/current
        input
	gwas
        optic_disc
        participant_phenotype
        "$PARTICIPANT_STAT_ID"/gcorr_diseases
        skeletons_etc
	qc
)

for i in "${folders[@]}"; do mkdir -p -m u=rwx,g=rwx,o=rx $RUN_DIR/$i; done # create folders if do not exist, exceptionally allow group write permissions, as this is a collaborative project

# One additional folder, required for ARIA
# If raw images are already in PNG format, dir_images is a symbolic link pointing to the raw images
if [[ "$image_type" = *.png ]]; then
        rm $dir_input/$aria_processor # clean if pipeline was run before
	ln -s $dir_images2 $dir_input/$aria_processor
#else
        # clean, and regenerate directory
        #rm -rf $dir_input/$aria_processor
        #mkdir -p $dir_input/$aria_processor
fi

#### COUNT RAW IMAGES, STORE LABELS

for i in $(ls $dir_images2 | cut -f1 -d.); do echo $i.png; done > $dir_input/all_raw_images.txt
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

VENTILE=4 # Vascular density threshold (every image with values below the treshold will be removed)

PARTICIPANT_STAT_ID=2022_08_16_ventile"$VENTILE"
QC=$RUN_DIR/qc/ageCorrected_ventiles"$VENTILE".txt
PARTICIPANT_PHENO_DIR=$RUN_DIR/participant_phenotype/ # participant-level phenotyoes
IMAGE_PHENO_DIR=$phenotypes_dir/current # image-level phenotypes
SAMPLE_FILE=/NVME/decrypted/ukbb/fundus/ukb_imp_v3_subset_fundus.sample # file determining participant order for bgenie GWAS

# COVARIATES, DISEASES
ukbb_files_dir='/NVME/decrypted/ukbb/labels/'
diseases_pheno_cov_file="$RUN_DIR"/diseases_cov/
name_phenofile="$PARTICIPANT_STAT_ID"_raw_with_instance.csv
csv_name="$PARTICIPANT_STAT_ID"_diseases_cov
csv_z_name="$PARTICIPANT_STAT_ID"_corrected_z.csv

#### GWAS
gwas_dir="$RUN_DIR"/gwas/"$PARTICIPANT_STAT_ID"/
diseases_ss_raw="/HDD/data/ukbb/disease_sumstats/files_modified/"
diseases_gwas_dir="$gwas_dir"/gcorr_diseases/

#### SUPPLEMENTARY PHENOTYPES
SUPPLEMENTARY_LABELS='tau1_all,tau1_artery,tau1_vein,ratio_AV_DF,tau2_all,tau2_artery,tau2_vein,tau4_all,tau4_artery,tau4_vein,D_std,D_A_std,D_V_std,D_CVMe,D_CVMe_A,D_CVMe_V,N_median_main_arteries,N_median_main_veins,arcLength_artery,arcLength_vein,bifurcations,VD_orig_all,VD_orig_artery,VD_orig_vein,ratio_VD,slope,slope_artery,slope_vein,mean_angle_taa,mean_angle_tva,eq_CRAE,eq_CRVE,median_CRAE,median_CRVE,ratio_CRAE_CRVE,ratio_median_CRAE_CRVE,medianDiameter_all,medianDiameter_artery,medianDiameter_vein,ratio_AV_medianDiameter'
SUPPLEMENTARY_NAMES='Tortuosity,Tortuosity A,Tortuosity V,Tortuosity ratio,Tortuosity2,Tortuosity2 A,Tortuosity2 V,tortuosity3,Tortuosity3 A,Tortuosity3 V,Std diameter,Std diameter A,Std diameter V,CVMe diameter,CVMe diameter A,CVMe diameter V,N main artery,N main vein,Arc length A,Arc length V,Bifurcations,Vascular density,Vascular density A,Vascular density V,Vascular density ratio,Fractal dimension,Fractal dimension A,Fractal dimension V,tAA,tVA,CRAE,CRVE,Median CRAE,Median CRVE,CRAE CRVE ratio,Median CRAE CRVE ratio,Median diameter,Median diameter A,Median diameter V,Median diameter ratio'

##### MAIN PHENOTYPES
MAIN_LABELS='tau1_artery,tau1_vein,ratio_AV_DF,D_A_std,D_V_std,bifurcations,VD_orig_artery,VD_orig_vein,ratio_VD,mean_angle_taa,mean_angle_tva,eq_CRAE,eq_CRVE,ratio_CRAE_CRVE,medianDiameter_artery,medianDiameter_vein,ratio_AV_medianDiameter'
MAIN_NAMES='Tortuosity A,Tortuosity V,Tortuosity ratio,Std diameter A,Std diameter V,Bifurcations,Vascular density A,Vascular density V,Vascular density ratio,tAA,tVA,CRAE,CRVE,CRAE CRVE ratio,Median diameter A,Median diameter V,Median diameter ratio'

##### FIGURES
FIGURES_DIR=$PARTICIPANT_PHENO_DIR/figures/
What_type_phenotype='main' #suplementary # fror MLR, Violin, Histogram, Genes


#### PARALLEL COMPUTING

#to run one by one, set n_cpu=1
n_cpu=50
step_size=$((n_img/n_cpu))
batch_max=$((n_cpu * step_size))
remainder=$(( n_img - step_size * n_cpu ))

#### USER-SPECIFIC CONFIGURATIONS

source $config_dir/config_personal.sh

