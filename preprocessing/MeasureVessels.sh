#!/bin/bash

######################################################################################################
# GOAL: From AV segemented images to: centerlines, diameters, segments specification and 
#       average measures as the median tortuosity (per image) and the median diameter (per image)
#       Computed in a parallel fashion to speed the process
#
# INPUT: Artery - Vein segmented images and config_.sh values
# OUTPUT: Files with vessels coordinates, diameters, segmentes, ... and averages measures 
#        (as median diameter and tortuosity)
# EXTERNAL SOFTWARES: ARIA (Matlab) -> Modified
# PROGRAMMING LANGUAGES USED: Matlab
######################################################################################################

source ../configs/config_.sh
begin=$(date +%s)

### Create a folder to save the results
mkdir -p $dir_ARIA_output

### Depending if you use the interpreter or not you have different options. These are commented below
## OPTION by default
rm -rf $dir_ARIA_output # clear old ARIA output
mkdir -p $dir_ARIA_output
if (( $batch_max > 0 )); then
	for i in $(seq 1 $step_size $batch_max); do # i: batch start, step_size: batch size
		./aria_batch.sh $i $step_size & # if don't set to background, Matlab will run in an endless loop
	done
	wait
fi
./aria_batch.sh $(($batch_max + 1)) $remainder & # if I don't set to background, Matlab will run in an endless loop
wait

## OPTION 2: (Recomended!) TO DO: there are many 'warning' errors when running Matlab
#cd $script_dir && /Applications/MATLAB_R2020b.app/bin/matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/__CODIGO/retina-phenotypes/'));ARIA_run_tests $script_parmeters ;quit;"

#cd $script_dir && /Applications/MATLAB_R2020b.app/bin/matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('$code_dir'));ARIA_run_tests $script_parmeters ;quit;"

## OPTION 3: if only INTERPRETER IS AVAILABLE
# TO DO: Complete explanation: First you have to run: ... (after compiling using the compileMAT.sh in the ARIA_tests folder)
#$ARIA_dir/run_ARIA_run_tests.sh $matlab_runtime $script_parmeters


echo FINISHED: files have been written to: $dir_ARIA_output
end=$(date +%s) # calculate execution time
tottime=$(expr $end - $begin)
echo "execution time: $tottime sec"
