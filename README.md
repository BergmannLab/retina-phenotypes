# Retina images traits: 

## Traits measured:
* Median diameter of: all the vessels, only arteries, only veins (' ')
* Median tortuosity of: all the vessels, only arteries, only veins (' ')
* Ratio between the diameters of the arteies and the diameters of the veins (' ')
* Ratio between the tortuosity of the arteies and the tortuosity of the veins (' ')
* Number of bifurcations and branching ('bifurcations')
* Main Temporal Venular Angle ('tva')
* Main Temporal Arteriolar Angle ('taa') 
* TO DO: define N_green or others 
* Fractal Dimensionality ()
* Vascular Density () 


## Requirements:

* You will need to download WNET 
* Matlab licence (if you have not acess there are still some traits you can measure)

## Pipeline:
1- Modify `configs/config_local.sh`

2 - Run `preprocessing/ClassifyAVLwnet.sh`. 
Output: AV maps for your images.  In the folder: ....
Code: `bash ClassifyAVLwnet.sh' or `sbatch ClassifyAVLwnet.sh' 

3 - Run `preprocessing/MeasureVessels.sh`. 
Output: Matlab output.  In the folder: ....
Code: `bash MeasureVessels.sh' or  `sbatch MeasureVessels.sh' 

4 - Run `preprocessing/run_measurePhenotype.sh`. 
Output: Trait measurements 
Code: `bash run_measurePhenotype.sh' or  `sbatch run_measurePhenotype.sh' 



## Some possible errors:
* LWNET, no image generated in DATASET_AV_maps:   AttributeError: module 'skimage.draw' has no attribute 'circle' . You need "your_python_dir/python -m pip install scikit-image==0.16.2" and python3
# python3 -m pip install --upgrade Pillow

## Reminders:
If you are not familiar with bash scripts and you want to change the code, the spaces are very imports!(Avoid when define variables, as use when conditions)

# Optic disc prediction using training weights from UK Biobank
1 - Generate the test dataset in required square 256x256px format: `preprocessing/optic-nerve-cnn/scripts/TEST_organize_datasets.ipynb` : Point to your dataset of choice, define output path, and modify the number of cores, and then run all the cells.
2 - Predict using the generated dataset : `preprocessing/optic-nerve-cnn/scripts/TEST_unet_od_on_ukbiobank.ipynb` : Change indir,outdir and no_batches accordingly, then run sequentially.
