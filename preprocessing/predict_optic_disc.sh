#!/bin/bash

source ../configs/config_.sh

output_dir=$code_dir/output/$data_set/optic_disc/
mkdir -p $output_dir

$python_dir optic-nerve-cnn/scripts/TEST_organize_datasets.py $dir_images $output_dir $n_cpu
$python_dir optic-nerve-cnn/scripts/TEST_unet_od_on_ukbiobank.py $output_dir $n_cpu $data_set
