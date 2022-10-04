############################## CONFIG FILE for fundus multitrait project ##############################

#### PROJECT COMPUTATION DIRECTORY

PROJECT_DIR=/NVME/decrypted/scratch/multitrait

#### CONFIGURATIONS DIRECTORY

config_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # returns parent directory of this file

#### PARALLEL COMPUTING 

#to run one by one, set n_cpu=1
n_cpu=50
step_size=$((n_img/n_cpu))
batch_max=$((n_cpu * step_size))
remainder=$(( n_img - step_size * n_cpu ))

#### USER-SPECIFIC CONFIGURATIONS

source $config_dir/config_personal.sh

############################## FROM IMAGES TO PHENOTYPES ##############################

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
	gwas/"$PARTICIPANT_STAT_ID"/gcorr_diseases
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

# ARIA parameter to decide if you analize arteries, veins or both(=all)
TYPE_OF_VESSEL_OF_INTEREST=all

#AV_threshold parameter is used when we only looked at arteries or veins. Segments with lower artery/vein scores than 0.79 were ignored. In this project we use the TYPE_OF_VESSEL_OF_INTEREST='all' option, so the 0.79 value is ignored.
AV_threshold=0.79 # segment score threshold 


### quality thresholds for ARIA (in this project we select very extrem values(-1, 999999), so we do not filter)
# Minumun total vessel accepted length and number of vessels
min_QCthreshold_1=-1

# Maximum total vessel accepted length
max_QCthreshold_1=999999

#Maximum total number of vessels accepted
min_QCthreshold_2=-1

#Maximum total number of vessels accepted
max_QCthreshold_2=999999

#### FUNDUS MEASUREMENTS

PHENOTYPE_OF_INTEREST='taa,tva,CRAE,CRVE,ratios_CRAE_CRVE,bifurcations,diameter_variability,aria_phenotypes,ratios,fractal_dimension,vascular_density,baseline,neo_vascularization,N_main_arteires,N_main_veins'
phenotypes_dir=$RUN_DIR/image_phenotype/

### PHENOTYPES PARAMETERS:
## Parameters for temporal angle
# R_0 is the initial radius. This value was selected base on the UKBB images
R_0=240 #also used for N_main_vessels  #Need to be tested if the dataset is different
# Distance between 2 concentrical radius. This value was selected base on the UKBB images
delta=10 #also usedfor N_main_vessels  #Need to be tested if the dataset is different

#Minor temporal value angle accepted (≈75° based on opthalmological advise)
min_ta=75

#Maximun temporal value angle accepted (≈200° based on opthalmological advise)
max_ta=200 

#The following values are selected to make a 'majority vote'. Since we are working with Real numbers, we accept they vote for the same if the new values is: upper_accept <= x <= lower_accept (please go to the article/documentation for more detail)
lower_accept=15 #Need to be tested if the dataset is different
upper_accept=2 #Need to be tested if the dataset is different

## Parameters for bifurcations
# neighborhood_cte define the maximal distance to decide if two end points are continuous segments
neighborhood_cte=3.5 #Need to be tested if the dataset is different

# Bifurcations with a distance > than 'norm_acceptance' are going to be delete to avoid overcounting
norm_acceptance=7.5 #Need to be tested if the dataset is different

## N main vessels
# min diameter accepted as a possible value for a main diamter (≈10 based on opthalmological advise)
limit_diameter_main=10 

## Vascular density, fractal dimension
mask_radius=660 # works for UKBB, may be adapted in other datasets, though only used for PBV (percent annotated as blood vessels) phenotype


###################### COMPLEMENTARY ANALYSIS: association with diseases, GWAS, ...  ##############################

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
csv_name="$PARTICIPANT_STAT_ID"_diseases_cov.csv
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
