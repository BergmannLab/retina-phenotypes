############################## CONFIG FILE for Retina pipeline ##############################

# REMEMBER!: Do not use spaces: "dir = x", will lead to errors, instead use "dir=x"


#### SELECT THE DATA SET YOU WANT TO USE:
data_set=DRIVE #options: DRIVE, IOSTAR, CHASEDB1
image_type=*.tif #opttions: *.jpg, *.tif
#### TYPE_OF_VESSEL_OF_INTEREST:
TYPE_OF_VESSEL_OF_INTEREST="all" # [artery|vein|all] 

#### PHENOTYPE_OF_INTEREST:
PHENOTYPE_OF_INTEREST='green_segments' #posibilities: 'tva', 'taa', 'bifurcations', 'green_segments', 'neo_vascularization', 'aria_phenotypes', 'fractal_dimension', 'ratios'
# TO DO: Add option to select all

###############################################################################

#### SELECT NUMBER OF RAW IMAGES DEPENDING ON THE DATA SET SELECTED:
# TO DO: INSTEAD OF DEFINE IT COUNT THE NUMBER OF IMAGES IN THE FOLDER!  using: ls | wc -l
if [ "$data_set" = "CHASEDB1" ]; then
    num_images=28 #28
elif [ "$data_set" = "DRIVE" ]; then
    num_images=20 #20
elif [ "$data_set" = "IOSTAR" ]; then
    num_images=30 #30
else
    num_images=0 # TO DO: Add error!
fi

echo Number of images equal to $num_images	



# quality thresholds for ARIA
min_QCthreshold_1=1100
max_QCthreshold_1=20000
min_QCthreshold_2=50
max_QCthreshold_2=250


#### BASE DIRS:
#   ----------------------------------------------------------------
#    If you do not change the pipeline you do not need to worry about this. 
#	Otherwise, you will need to specify the directory of: 
#       	- Raw images: dir_images
#       	- LWnet software: classification_output_dir
#       	- LWnet images output: lwnet_dir
#   ---------------------------------------------------------------- 
# TO DO!: change the directories in a more automated fashion!

#Needed to change########
code_dir='/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/__CODIGO/retina-phenotypes/' 
lwnet_dir='/Users/sortinve/develop/Codigos_github/lwnet/'
##############

dir_preprocessing=$code_dir'/preprocessing/'
dir_images=$code_dir'/input/'$data_set'_images/'$data_set'/'
dir_images2=$code_dir'/input/'$data_set'_images/'
classification_output_dir=$code_dir'/input/'$data_set'_AV_maps/'
dir_ARIA_output=$code_dir'/output/ARIA_output_'$data_set'/'
ARIA_dir=$code_dir'/preprocessing/helpers/MeasureVessels/src/petebankhead-ARIA-328853d/ARIA_tests/'
MeasureVessels_dir=$code_dir'/output/VesselMeasurements/'$data_set'/'
phenotypes_dir=$code_dir'/output/phenotypes_'$data_set'_'$TYPE_OF_VESSEL_OF_INTEREST'/'
OD_output_dir=$code_dir'/output/'$data_set'_OD/'

#### QUALITY THRESHOLDS OF ARTERY/VEIN CLASSIFICATION: 
AV_threshold=0.0 # Consider all classified vessels


#### OTRAS para modificiar?
ALL_IMAGES=$dir_images2/noQC.txt

