#!/bin/bash
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

mkdir -p $dir_ARIA_output

### Run vessels measurements with ARIA (Matlab):
base_dir=$PWD
script_dir=$PWD/helpers/MeasureVessels/src/petebankhead-ARIA-328853d/ARIA_tests

#for i in $(seq 1 $step_size $batch_max); do echo "ARIA_run_tests 0 REVIEW $dir_images2 $classification_output_dir $TYPE_OF_VESSEL_OF_INTEREST 0.79 $script_dir ${i} $step_size -1 999999 -1 999999 $dir_ARIA_output HAAVA"; done

# OPTION 0
rm $dir_ARIA_output/* # clear old ARIA output
cd $script_dir
if (( $batch_max > 0 )); then
    for i in $(seq 1 $step_size $batch_max); do (nohup nice matlab -nodisplay -nosplash -nodesktop -r "ARIA_run_tests 0 REVIEW $dir_input $classification_output_dir $TYPE_OF_VESSEL_OF_INTEREST 0.79 $script_dir $i $step_size -1 999999 -1 999999 $dir_ARIA_output $aria_processor" > $base_dir/batch$i.txt &); done
fi
nohup nice matlab -nodisplay -nosplash -nodesktop -r "ARIA_run_tests 0 REVIEW $dir_input $classification_output_dir $TYPE_OF_VESSEL_OF_INTEREST 0.79 $script_dir $(($batch_max + 1)) $remainder -1 999999 -1 999999 $dir_ARIA_output $aria_processor" > $base_dir/batch$(($batch_max + 1 )).txt &

## OPTION 1: (Recomended!) TO DO: there are many 'warning' errors when running Matlab
#cd $script_dir && /Applications/MATLAB_R2020b.app/bin/matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/__CODIGO/retina-phenotypes/'));ARIA_run_tests $script_parmeters ;quit;"

#cd $script_dir && /Applications/MATLAB_R2020b.app/bin/matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('$code_dir'));ARIA_run_tests $script_parmeters ;quit;"

## OPTION 2: if only INTERPRETER IS AVAILABLE
# TO DO: Complete explanation: First you have to run: ... (after compiling using the compileMAT.sh in the ARIA_tests folder)
#$ARIA_dir/run_ARIA_run_tests.sh $matlab_runtime $script_parmeters

#echo FINISHED: files have been written to: $dir_ARIA_output
#end=$(date +%s) # calculate execution time
#tottime=$(expr $end - $begin)
#echo "execution time: $tottime sec"
