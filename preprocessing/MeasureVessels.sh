#!/bin/bash # TO DO: delete?
#SBATCH --account=sbergman_retina
#SBATCH --job-name=MeasureVessels
#SBATCH --output=helpers/MeasureVessels/slurm_runs/slurm-%x_%j.out
#SBATCH --error=helpers/MeasureVessels/slurm_runs/slurm-%x_%j.err
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 6G
#SBATCH --partition normal
####### --time 00-01:00:00
#SBATCH --time 00-03:30:00
####### --array=1-582 #UKBB
#SBATCH --array=1-26 #CoLaus is_color=False

source ../configs/config_.sh
begin=$(date +%s)

if [ $type_run = "one_by_one" ]; then
    chunk_start=1 
    chunk_size=$num_images 
elif [ $type_run = "parallel" ]; then
    # job array
    j_array_params=$PWD/helpers/MeasureVessels/j_array_params.txt
    PARAM=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $j_array_params)
    chunk_start=$(echo $PARAM | cut -d" " -f1)
    chunk_size=$(echo $PARAM | cut -d" " -f2)
else
    echo "You only can run it one ba one or in parallel, specify what you want to use on config_."
fi

mkdir -p $dir_ARIA_output

### Run vessels measurements with ARIA (Matlab):
script_dir=$PWD/helpers/MeasureVessels/src/petebankhead-ARIA-328853d/ARIA_tests/
script_parmeters="0 REVIEW $dir_images2 $classification_output_dir $TYPE_OF_VESSEL_OF_INTEREST $AV_threshold $script_dir $chunk_start $chunk_size $min_QCthreshold_1 $max_QCthreshold_1 $min_QCthreshold_2 $max_QCthreshold_2 $dir_ARIA_output $data_set"
# ARIA_run_tests 0 REVIEW /Users/sortinve/develop/retina/input/DRIVE_images/ /Users/sortinve/develop/retina/input/DRIVE_AV_maps/ all 0.0 /Users/sortinve/develop/retina/preprocessing/helpers/MeasureVessels/src/petebankhead-ARIA-328853d/ARIA_tests/ 1 20 1100 20000 50 250 /Users/sortinve/develop/retina/output/ARIA_output_DRIVE/ DRIVE

## OPTION 1: (Recomended!) TO DO: there are many 'warning' errors when running Matlab
#cd $script_dir && /Applications/MATLAB_R2020b.app/bin/matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/__CODIGO/retina-phenotypes/'));ARIA_run_tests $script_parmeters ;quit;"
cd $script_dir && /Applications/MATLAB_R2020b.app/bin/matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('$code_dir'));ARIA_run_tests $script_parmeters ;quit;"

## OPTION 2: if only INTERPRETER IS AVAILABLE
# TO DO: Complete explanation: First you have to run: ... (after compiling using the compileMAT.sh in the ARIA_tests folder)
#$ARIA_dir/run_ARIA_run_tests.sh $matlab_runtime $script_parmeters

echo FINISHED: files have been written to: $dir_ARIA_output
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"
