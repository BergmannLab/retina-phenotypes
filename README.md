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
* Fractal Dimensionality 


## Requirements:

* You will need to download WNET 
* Matlab licence (if you have not acess there are still some traits you can measure)

## Pipeline:
1- Modify `configs/config_sofia.sh`

2 - Run `preprocessing/ClassifyAVLwnet_sofia.sh`. Output: AV maps for your images 

3 - Run `preprocessing/MeasureVessels_Sofia.sh`. Output: Matlab output

4 - Run `preprocessing/run_measurePhenotype_sofia.sh`. Output: Trait measurements



## Some possible errors:

LWNET, no image generated in DATASET_AV_maps:   AttributeError: module 'skimage.draw' has no attribute 'circle'

# Optic disc prediction using training weights from UK Biobank
1 - Generate the test dataset in required square 256x256px format: `preprocessing/optic-nerve-cnn/scripts/Organize test dataset.ipynb` : Point to your dataset of choice, define output path, and modify the number of cores, and then run all the cells.
2 - Predict using the generated dataset : TO-DO
