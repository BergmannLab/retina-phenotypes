#!/bin/bash

source ../../configs/config_.sh

python3.8 image_to_participant.py $QC $IMAGE_PHENO_DIR/ $PARTICIPANT_PHENO_DIR/ $SAMPLE_FILE $PARTICIPANT_STAT_ID
