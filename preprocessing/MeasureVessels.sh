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

### Set up the directories
base_dir=$PWD
script_dir=$PWD/helpers/MeasureVessels/src/petebankhead-ARIA-328853d/ARIA_tests
helper_dir=$base_dir/helpers/MeasureVessels

### Depending if you use the interpreter or not you have different options. These are commented below
# NOTE: The values -1 999999 -1 999999 are respectively: the mininum and maximun number of segments and vessel pixels
#       that is accepted. These values can be used as a Quality Control, however, if you want to not apply this filter,
#       use these values. Regardint the value 0.79 is an acceptance parameter for the AV classification (recomended). 
#       If you want to use any tolerance value you can use 0.0

## OPTION by default
rm -rf $dir_ARIA_output # clear old ARIA output
mkdir -p $dir_ARIA_output
cd $script_dir
if (( $batch_max > 0 )); then
    for i in $(seq 1 $step_size $batch_max); do (nice $matlab_dir -nodisplay -nosplash -nodesktop -r "ARIA_run_tests 0 REVIEW $dir_input $classification_output_dir $TYPE_OF_VESSEL_OF_INTEREST 0.79 $script_dir $i $step_size -1 999999 -1 999999 $dir_ARIA_output $aria_processor $ALL_IMAGES" > $helper_dir/batch$i.txt &); done
fi
nice $matlab_dir -nodisplay -nosplash -nodesktop -r "ARIA_run_tests 0 REVIEW $dir_input $classification_output_dir $TYPE_OF_VESSEL_OF_INTEREST 0.79 $script_dir $(($batch_max + 1)) $remainder -1 999999 -1 999999 $dir_ARIA_output $aria_processor $ALL_IMAGES" > $helper_dir/batch$(($batch_max + 1 )).txt &

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
