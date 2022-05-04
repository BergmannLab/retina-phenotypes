############################## CONFIG FILE for Retina pipeline ##############################
#### SELECT THE DATA SET YOU WANT TO USE:
data_set=CLRIS #options: DRIVE, IOSTAR, CHASEDB1
image_type=*.png #options: *.jpg, *.tif, ...
#### TYPE_OF_VESSEL_OF_INTERE ST:
TYPE_OF_VESSEL_OF_INTEREST="all" # [artery|vein|all] 

#### PHENOTYPE_OF_INTEREST:
PHENOTYPE_OF_INTEREST='green_segments' #posibilities: 'tva', 'taa', 'bifurcations', 'green_segments', 'neo_vascularization', 'aria_phenotypes', 'fractal_dimension', 'ratios' , etc # TO DO: Add option to select all
type_run="sbatch" # ["bash", "sbatch"]
step=10 # You can change it

#### BASE DIRS:
archive=/stornext/CHUV1/archive/FAC/FBM/DBC/sbergman/retina/
data=/data/FAC/FBM/DBC/sbergman/retina/
scratch=/scratch/beegfs/FAC/FBM/DBC/sbergman/retina/


classification_output_dir=$data/UKBiob/fundus/AV_maps/ #LWNET_DIR
dir_ARIA_output=$scratch/preprocessing/output/backup/2021_10_06_rawMeasurements_withoutQC/ #ARIA_MEASUREMENTS_DIR
phenotypes_dir=$scratch/UKBiob/fundus/phenofiles/ #PHENOFILES_DIR
OD_output_dir=$scratch/beegfs/FAC/FBM/DBC/sbergman/retina/ #new!
ALL_IMAGES=$data/UKBiob/fundus/index_files/noQC.txt #ALL_IMAGES

dir_images=$code_dir'/input/'$data_set'_images/'$data_set'/'
dir_images2=$code_dir'/input/'$data_set'_images/'

lwnet_dir='/Users/sortinve/develop/Codigos_github/lwnet/'
ARIA_dir=$code_dir'/preprocessing/helpers/MeasureVessels/src/petebankhead-ARIA-328853d/ARIA_tests/'
MeasureVessels_dir=$code_dir'/output/VesselMeasurements/'$data_set'/'

genpath_dir=$code_dir


### Quality thresholds for the images in ARIA base on the number of segments (You can modify it if you think it is needed)
min_QCthreshold_1=0
max_QCthreshold_1=200000
min_QCthreshold_2=0
max_QCthreshold_2=2000

#### QUALITY THRESHOLDS OF LWNET ARTERY/VEIN CLASSIFICATION: 
AV_threshold=0.0 # If 0.0 Consider all classified vessels 
