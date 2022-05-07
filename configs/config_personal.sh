############################## CONFIG FILE for Retina pipeline ##############################
# REMEMBER!: Do not use spaces: "dir = x", this will lead to errors, instead use "dir=x"

#### SELECT THE DATA SET YOU WANT TO USE:
data_set=DRIVE #options: DRIVE, IOSTAR, CHASEDB1, UK_BIOBANK
image_type=*.tif #options: *.jpg, *.tif, *.png, ...

#### TYPE OF VESSEL OF INTEREST:
TYPE_OF_VESSEL_OF_INTEREST="all" # [artery|vein|all] 

PHENOTYPE_OF_INTEREST='bifurcations' #posibilities: 'tva', 'taa', 'bifurcations', 'green_segments', 'neo_vascularization', 'aria_phenotypes', 'fractal_dimension', 'ratios' , 'vascular_density', etc # TO DO: Add option to select all

#### BASE DIRS:
#   You have to define:
#           - Your code directory: 'code_dir'
#           - Where did you install the lwnet code: 'lwnet_dir'
#           - The python venv that you want to use: 'python_dir' # This is to try to avoid packages problems :)
#   ----------------------------------------------------------------
#   If you do not change the pipeline, you do not need to worry about this. 
#	Otherwise, you will need to specify the directory of: 
#       	- Raw images: 'dir_images2' 
#       	- LWnet images output: 'classification_output_dir'
#   Remark: ARIA is programed to read the raw images from a folfer like: '$data_set'_images/'$data_set'/' If you do not want to change ARIA you should stick to this format
#   Remark: Remember that you have to include the images and the matlab code with you run the matlab code: addpath(genpath('$genpath_dir'))
#   ---------------------------------------------------------------- 

#### BASE DIRECTORY
code_dir='/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/__CODIGO/retina-phenotypes/'

# SOFTWARE
ARIA_dir=$code_dir'/preprocessing/helpers/MeasureVessels/src/petebankhead-ARIA-328853d/ARIA_tests/'
# aria_processor needs to be specified because we process UK Biobank images with the CLRIS processor:
if [[ $data_set == UK_BIOBANK ]]; then
        aria_processor=CLRIS
else
        aria_processor=$data_set
fi
lwnet_dir='/Users/sortinve/develop/Codigos_github/lwnet/'
lwnet_gpu=False
python_dir=/Users/sortinve/PycharmProjects/pythonProject/venv/bin/python

#### INPUT
dir_images2=$code_dir'/input/'$data_set'_images/'
dir_images=$dir_images2/$aria_processor'/'
# If raw images are already in PNG format, dir_images is a symbolic link pointing to the raw images
if [[ "$image_type" = *.png ]]; then
	rm $dir_images2/$aria_processor # clean if pipeline was run before
	ln -s $dir_images2 $dir_images2/$aria_processor
fi
ALL_IMAGES=$dir_images2/noQC.txt

n_img=$(ls $dir_images2/$image_type | wc -l)
echo Number of images equal to $n_img

#### OUTPUT
dir_ARIA_output=$code_dir'/output/ARIA_output_'$data_set'/'
MeasureVessels_dir=$code_dir'/output/VesselMeasurements/'$data_set'/'
phenotypes_dir=$code_dir'/output/phenotypes_'$data_set'_'$TYPE_OF_VESSEL_OF_INTEREST'/'
genpath_dir=$code_dir
classification_output_dir=$code_dir'/input/'$data_set'_AV_maps/'

#### PARALLEL COMPUTING
type_run="one_by_one" # ["one_by_one", "parallel"]
n_cpu=4
step_size=$((n_img/n_cpu))
batch_max=$((n_cpu * step_size))
remainder=$(( n_img - step_size * n_cpu ))
