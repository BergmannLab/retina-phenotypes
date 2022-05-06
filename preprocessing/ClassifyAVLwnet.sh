#!/bin/bash  ##TO DO, can we delete???
#SBATCH --account=sbergman_retina
#SBATCH --job-name=⚡lwnet⚡
#SBATCH --output=helpers/ClassifyAVUncertain/slurm_runs/slurm-%x_%j.out
#SBATCH --error=helpers/ClassifyAVUncertain/slurm_runs/slurm-%x_%j.err
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 12GB
#SBATCH --partition normal
#SBATCH --time 01:30:00
#SBATCH --array=1-582

#### Read the vairables requiered from config.sh:
source ../configs/config_.sh
begin=$(date +%s)

#### Create the folder where the after preprocessing images are going to be located (for the dataset selected):
mkdir $dir_images
# TO DO: Add a step to can avoid this step if needed:
#### Preprocessing: .png format, avoid spaces in names, and create file with images names:

$python_dir basic_preprocessing.py $dir_images2 $dir_images $image_type

#### Create the folder where the AV maps are going to be located (for the dataset selected):
mkdir $classification_output_dir

if [ $type_run = "one_by_one" ]; then
    #### Artery Vein segementation using WNET:
    cd $lwnet_dir
    raw_imgs=( "$dir_images"* )
    for i in $(eval echo "{1..$num_images}"); do 
        image="${raw_imgs[i]}"
        $python_dir predict_one_image_av.py --model_path experiments/big_wnet_drive_av/ --im_path $image --result_path $classification_output_dir
    done

elif [ $type_run = "parallel" ]; then
    max=$((num_images-step))
    for i in $(eval echo "{1..$max}");
    do
    echo "$i $((i+step))"
    i=$((i+step))
    done > $PWD/helpers/ClassifyAVUncertain/j_array_params.txt

    j_array_params=$PWD/helpers/ClassifyAVUncertain/j_array_params.txt 
    PARAM=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $j_array_params)
    chunk_start=$(echo $PARAM | cut -d" " -f1)
    chunk_size=$(echo $PARAM | cut -d" " -f2)

    #### Artery Vein segementation using WNET: (Analyze many image at the same time)
    cd $lwnet_dir
    raw_imgs=( "$dir_images"* )
    # Code if you want to anaylze the images one by one: for i in $(eval echo "{1..$num_images}"); do 
    for i in $(seq $chunk_start $(($chunk_start+$chunk_size-1))); do
        image="${raw_imgs[i]}"
        $python_dir predict_one_image_av.py --model_path experiments/big_wnet_drive_av/ --im_path $image --result_path $classification_output_dir &
    done

else
    echo "You only can run with bash or sbatch, specify what you want to use on config"
fi

### Rename the LWnet output (for ARIA you need the raw_image and the AV_image to have the same name):
$python_dir $code_dir/preprocessing/Change_the_name_LWNEToutput.py $classification_output_dir 

echo FINISHED: Images have been classified, and written to $classification_output_dir
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"
