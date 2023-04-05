#!/bin/bash

######################################################################################################
# GOAL: From raw images to Optic Disk position and size, computed in a parallel fashion
#
# INPUT: fundus images and config_.sh values
# OUTPUT: Optic Disk position and size
# EXTERNAL SOFTWARES: optic-nerve-cnn (Python) -> Modified (https://github.com/seva100/optic-nerve-cnn)
# PROGRAMMING LANGUAGES USED: Python
######################################################################################################

## Remark: The code was modified and re-trained to work in our dataset UKBB

source ../configs/config_.sh

optic_disc_dir=$RUN_DIR/optic_disc/

mkdir -p $optic_disc_dir

## Organize the datasets
$python_dir optic-nerve-cnn/scripts/TEST_organize_datasets.py $dir_images/ $optic_disc_dir $n_cpu $ALL_IMAGES

## UNET to measure OD postition and size
$python_dir optic-nerve-cnn/scripts/TEST_unet_od_on_ukbiobank.py $optic_disc_dir $n_cpu $data_set
