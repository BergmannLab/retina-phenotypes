#!/bin/bash
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

mkdir -p $RUN_DIR

#### Create the folder where the after preprocessing images are going to be located (for the dataset selected):
if [[ "$image_type" != *.png ]]; then
	mkdir $dir_images
fi
# TO DO: Add a step to can avoid this step if needed:
#### Preprocessing: .png format, avoid spaces in names, and create file with images names:

$python_dir basic_preprocessing.py $dir_images2 $dir_images $dir_input $image_type

#### Create the folder where the AV maps are going to be located (for the dataset selected):
echo $classification_output_dir
mkdir $classification_output_dir

#### Artery Vein segementation using WNET: (Analyze many image at the same time)
parallel_lwnet () {
for j in $(seq $1 $2 ); do
	image=$dir_images/"${raw_imgs[ $(( j - 1 )) ]}"
	#echo $image
	if [ $lwnet_gpu = "False" ]; then
		# taskset: limits script to specific CPU
		nice taskset -c $3 $python_dir predict_one_image_av.py --model_path experiments/big_wnet_drive_av/ --im_path $image --result_path $classification_output_dir --device cpu
        else
        	nice taskset -c $3 $python_dir predict_one_image_av.py --model_path experiments/big_wnet_drive_av/ --im_path $image --result_path $classification_output_dir --device cuda:0
        fi
    done
}

cd $lwnet_dir
readarray -t raw_imgs < $ALL_IMAGES
#echo $raw_imgs | head -n10

#for i in $(seq 1 20); do echo ${raw_imgs[i]}; done
echo lwnet input $dir_images
for i in $(seq 1 $(( $n_cpu + 1 )) ); do #n_cpu + 1 to force a remainder iteration
    a=$(( i * step_size ))
    b=$n_img
    lower_lim=$(( 1 + $(( $(( i - 1 )) * step_size )) ))
    upper_lim=$(( a < b ? a : b )) # minimum operation
    
    echo Batch $i: from $lower_lim to $upper_lim
    parallel_lwnet $lower_lim $upper_lim $(( i - 1 )) & # 3rd variable: uniquely allocated CPU per parallel task, speeds up things!
done

wait

### Rename the LWnet output (for ARIA you need the raw_image and the AV_image to have the same name):
$python_dir $config_dir/../preprocessing/Change_the_name_LWNEToutput.py $classification_output_dir 

echo FINISHED: Images have been classified, and written to $classification_output_dir
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"
