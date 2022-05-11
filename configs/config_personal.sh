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

#### PROJECT ROOT
code_dir='/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/__CODIGO/retina-phenotypes/'

#### SOFTWARE
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
matlab_dir=/Applications/MATLAB_R2020b.app/bin/matlab

#### INPUT
dir_input=$code_dir/input/$data_set/
dir_images2=$dir_input/raw/
dir_images=$dir_input/$aria_processor/
# If raw images are already in PNG format, dir_images is a symbolic link pointing to the raw images
if [[ "$image_type" = *.png ]]; then
	rm $dir_input/$aria_processor # clean if pipeline was run before
	ln -s $dir_images2 $dir_input/$aria_processor
fi
ALL_IMAGES=$dir_input/noQC.txt
n_img=$(ls $dir_images2 | wc -l)
echo Number of images equal to $n_img

classification_output_dir=$dir_input/AV_maps/

#### OUTPUT
dir_ARIA_output=$code_dir/output/$data_set/ARIA_output/
MeasureVessels_dir=$code_dir/output/$data_set/VesselMeasurements/
phenotypes_dir=$code_dir/output/$data_set/phenotypes_$TYPE_OF_VESSEL_OF_INTEREST/
genpath_dir=$code_dir

#### PARALLEL COMPUTING
#to run one by one, set n_cpu=1
n_cpu=4
step_size=$((n_img/n_cpu))
batch_max=$((n_cpu * step_size))
remainder=$(( n_img - step_size * n_cpu ))
