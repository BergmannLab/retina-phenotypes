#!/bin/bash

# created because Matlab was causing problems when setting a Matlab command to run in background
# now MeasureVessels invokes this script, which is set to run in background

# source ../configs/config_.sh # was causing problems when run simultaneously from multiple batches

### Set up the directories
base_dir=$PWD
aria_dir=$PWD/helpers/MeasureVessels/src/petebankhead-ARIA-328853d/ARIA_tests
helper_dir=$base_dir/helpers/MeasureVessels

batch_start=$1
batch_size=$2

cd $aria_dir
nice $matlab_dir -nodisplay -nosplash -nodesktop -r "ARIA_run_tests 0 REVIEW $dir_input $classification_output_dir $TYPE_OF_VESSEL_OF_INTEREST $AV_threshold $aria_dir $batch_start $batch_size $min_QCthreshold_1 $max_QCthreshold_1 $min_QCthreshold_2 $max_QCthreshold_2 $dir_ARIA_output $aria_processor $ALL_IMAGES" > "$helper_dir"/batch"$batch_start".txt
