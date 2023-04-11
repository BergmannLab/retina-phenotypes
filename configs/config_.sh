############################## CONFIG FILE for fundus multitrait project ##############################

#### PROJECT COMPUTATION DIRECTORY

PROJECT_DIR=/NVME/decrypted/scratch/multitrait

#### CONFIGURATIONS DIRECTORY

config_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # returns parent directory of this file

############################# FROM IMAGES TO PHENOTYPES ##############################

#### DATASET OF CHOICE

data_set=UK_BIOBANK

# Check in case using new data
if [[ $data_set = DRIVE ]]; then
    image_type=*.tif
else
    image_type=*.png
fi

echo Data set: $data_set
echo Image type: $image_type


# ARIA options: DRIVE, IOSTAR, CHASEDB1
aria_processor=CLRIS # Works best for UK Biobank fundus images

#### PROJECT DATA

dir_images2=/NVME/decrypted/ukbb/fundus/raw/CLRIS/ # folder containing raw fundus images

#### UNIQUE RUN LABEL

RUN="$data_set"_PREPRINT

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
    rm -f $dir_input/$aria_processor # remove symbolic link if it exists
    ln -s $dir_images2 $dir_input/$aria_processor # (re-) create symbolic link
fi
# Else, basic_preprocessing.sh will convert raw images to PNG format and store them in dir_images

#### COUNT RAW IMAGES, CREATE IMAGE LABELS FILE

# If file already exists, move to all_raw_images.txt.prev
if [ -f "$dir_input"/all_raw_images.txt ]; then
    mv "$dir_input"/all_raw_images.txt "$dir_input"/all_raw_images.txt.prev
    echo Moved all_raw_images.txt to all_raw_images.txt.prev
fi
for i in $(ls $dir_images2 | cut -f1 -d.); do
    echo $i.png >> "$dir_input"/all_raw_images.txt
done

ALL_IMAGES=$dir_input/all_raw_images.txt
n_img=$(cat $ALL_IMAGES | wc -l)
echo Number of raw images: $n_img

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

PHENOTYPE_OF_INTEREST='taa,tva,CRAE,CRVE,ratios_CRAE_CRVE,ratios_VD,bifurcations,diameter_variability,aria_phenotypes,ratios,fractal_dimension,vascular_density,baseline,neo_vascularization,N_main_arteires,N_main_veins'
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

#VENTILE=4 # Vascular density threshold (every image with values below the treshold will be removed)

PARTICIPANT_STAT_ID=2022_11_23_covar_fix
QC=$RUN_DIR/qc/zekavat_centercropped_percentile25.txt #ageCorrected_ventiles"$VENTILE".txt
PARTICIPANT_PHENO_DIR=$RUN_DIR/participant_phenotype/ # participant-level phenotyoes
IMAGE_PHENO_DIR=$phenotypes_dir/current # image-level phenotypes
SAMPLE_FILE=/NVME/decrypted/ukbb/fundus/ukb_imp_v3_subset_fundus.sample # file determining participant order for bgenie GWAS

# OUTLIER REMOVAL
n_std=10 # all measurement>mean+n_std*std | measurement<mean-n_std*std are removed from analysis

# COVARIATES, DISEASES
ukbb_files_dir='/NVME/decrypted/ukbb/labels/'
diseases_pheno_cov_file="$RUN_DIR"/diseases_cov/
name_phenofile="$PARTICIPANT_STAT_ID"_raw_with_instance.csv
csv_diseases_cov_name="$PARTICIPANT_STAT_ID"_diseases_cov.csv
csv_z_name="$PARTICIPANT_STAT_ID"_z_corrected.csv
describe_baselines_file="$PARTICIPANT_STAT_ID"_describe_baselines.csv

#### GWAS
gwas_dir="$RUN_DIR"/gwas/"$PARTICIPANT_STAT_ID"/
diseases_ss_raw="/HDD/data/ukbb/disease_sumstats/files_modified/"
diseases_gwas_dir="$gwas_dir"/gcorr_diseases/

#### SUPPLEMENTARY PHENOTYPES
SUPPLEMENTARY_LABELS='tau1_all,tau1_artery,tau1_vein,ratio_AV_DF,tau2_all,tau2_artery,tau2_vein,tau4_all,tau4_artery,tau4_vein,D_std,D_A_std,D_V_std,D_CVMe,D_CVMe_A,D_CVMe_V,sd_mean_size,N_median_main_arteries,N_median_main_veins,arcLength_artery,arcLength_vein,bifurcations,VD_orig_all,VD_orig_artery,VD_orig_vein,ratio_VD,FD_all,FD_artery,FD_vein,mean_angle_taa,mean_angle_tva,eq_CRAE,eq_CRVE,median_CRAE,median_CRVE,CRAE,CRVE,ratio_CRAE_CRVE,ratio_median_CRAE_CRVE,ratio_standard_CRE,medianDiameter_all,medianDiameter_artery,medianDiameter_vein,ratio_AV_medianDiameter'
SUPPLEMENTARY_NAMES='tortuosity,A tortuosity,V tortuosity,ratio tortuosity,tortuosity2,A tortuosity2,V tortuosity2,tortuosity3,A tortuosity3,V tortuosity3,std diameter,A std diameter,V std diameter,CVMe diameter,A CVMe diameter,V CVMe diameter,std norm diameter,A num main,V num main,A arc length,V arc length,bifurcations,vascular density,A vascular density,V vascular density,ratio vascular density,fractal dimension,A fractal dimension,V fractal dimension,A temporal angle,V temporal angle,A central retinal eq,V central retinal eq,A main diameter,V main diameter,A central retinal eq2,V central retinal eq2,ratio central retinal eq,ratio main diameter,ratio central retinal eq2,median diameter,A median diameter,V median diameter,ratio median diameter'

##### MAIN PHENOTYPES
MAIN_LABELS='mean_angle_taa,mean_angle_tva,tau1_vein,tau1_artery,ratio_AV_DF,eq_CRAE,ratio_CRAE_CRVE,D_A_std,D_V_std,eq_CRVE,ratio_VD,VD_orig_artery,bifurcations,VD_orig_vein,medianDiameter_artery,medianDiameter_vein,ratio_AV_medianDiameter'
MAIN_NAMES='A temporal angle,V temporal angle,V tortuosity,A tortuosity,ratio tortuosity,A central retinal eq,ratio central retinal eq,A std diameter,V std diameter,V central retinal eq,ratio vascular density,A vascular density,bifurcations,V vascular density,A median diameter,V median diameter,ratio median diameter'

##### FIGURES
FIGURES_DIR=$RUN_DIR//figures/
What_type_phenotype='main' #suplementary # fror MLR, Violin, Histogram, Genes

#### PARALLEL COMPUTING 

#to run one by one, set n_cpu=1
n_cpu=10
step_size=$((n_img/n_cpu))
batch_max=$((n_cpu * step_size))
remainder=$(( n_img - step_size * n_cpu ))

#### USER-SPECIFIC CONFIGURATIONS

source $config_dir/config_personal.sh
