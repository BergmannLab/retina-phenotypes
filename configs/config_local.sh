############################## CONFIG FILE for Retina pipeline ##############################
# REMEMBER!: Do not use spaces: "dir = x", this will lead to errors, instead use "dir=x"

#### SELECT THE DATA SET YOU WANT TO USE:
data_set=DRIVE #options: DRIVE, IOSTAR, CHASEDB1
image_type=*.tif #options: *.jpg, *.tif, ...
#### TYPE_OF_VESSEL_OF_INTERE ST:
TYPE_OF_VESSEL_OF_INTEREST="all" # [artery|vein|all] 

#### PHENOTYPE_OF_INTEREST:
PHENOTYPE_OF_INTEREST='green_segments' #posibilities: 'tva', 'taa', 'bifurcations', 'green_segments', 'neo_vascularization', 'aria_phenotypes', 'fractal_dimension', 'ratios' , etc # TO DO: Add option to select all
type_run="bash" # ["bash", "sbatch"]
step=2 # You can change it

#### BASE DIRS:
#   You have to define:
#           - Your code directory: 'code_dir'
#           - Where did you installed the lwnet code: 'lwnet_dir'
#           - The python venv that you want to use: 'python_dir' # This is to try to avoid packages problems :)
#   ----------------------------------------------------------------
#   If you do not change the pipeline you do not need to worry about this. 
#	Otherwise, you will need to specify the directory of: 
#       	- Raw images: 'dir_images' and 'dir_images2' 
#       	- LWnet images output: 'classification_output_dir'
#   Remark: ARIA is programed to read the raw images from a folfer like: '$data_set'_images/'$data_set'/' If you do not want to change ARIA you should stick to this format
#   Remark: Remember that you have to include the images and the matlab code with you run the matlab code: addpath(genpath('$genpath_dir'))
#   ---------------------------------------------------------------- 


#### Needed to be changed:
code_dir='/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/__CODIGO/retina-phenotypes/' 
lwnet_dir='/Users/sortinve/develop/Codigos_github/lwnet/'
python_dir=/Users/sortinve/PycharmProjects/pythonProject/venv/bin/python
code_dir='/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/__CODIGO/retina-phenotypes/' #Needed to change


dir_images=$code_dir'/input/'$data_set'_images/'$data_set'/'
dir_images2=$code_dir'/input/'$data_set'_images/'
lwnet_dir='/Users/sortinve/develop/Codigos_github/lwnet/'
classification_output_dir=$code_dir'/input/'$data_set'_AV_maps/'
dir_ARIA_output=$code_dir'/output/ARIA_output_'$data_set'/'
ARIA_dir=$code_dir'/preprocessing/helpers/MeasureVessels/src/petebankhead-ARIA-328853d/ARIA_tests/'
MeasureVessels_dir=$code_dir'/output/VesselMeasurements/'$data_set'/'
phenotypes_dir=$code_dir'/output/phenotypes_'$data_set'_'$TYPE_OF_VESSEL_OF_INTEREST'/'
ALL_IMAGES=$dir_images2/noQC.txt
genpath_dir=$code_dir

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


### Quality thresholds for the images in ARIA base on the number of segments (You can modify it if you think it is needed)
min_QCthreshold_1=1100
max_QCthreshold_1=20000
min_QCthreshold_2=50
max_QCthreshold_2=250

#### QUALITY THRESHOLDS OF LWNET ARTERY/VEIN CLASSIFICATION: 
AV_threshold=0.0 # If 0.0 Consider all classified vessels 
