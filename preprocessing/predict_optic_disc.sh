#!/bin/bash

source ../configs/config_.sh

optic_disc_dir=$RUN_DIR/optic_disc/
mkdir -p $optic_disc_dir

$python_dir optic-nerve-cnn/scripts/TEST_organize_datasets.py $dir_images/ $optic_disc_dir $n_cpu $ALL_IMAGES
$python_dir optic-nerve-cnn/scripts/TEST_unet_od_on_ukbiobank.py $optic_disc_dir $n_cpu $data_set
