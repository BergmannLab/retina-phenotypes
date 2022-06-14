# Bryhnild directories (to avoid duplications):
Sensitive data (i.e. with eid) is going to be located on '/NVME/decrypted/' and not sensitive is going to be located on '/HDD/data/':

* All the coordinates values (with centerlines) and the ARIA diameters for different QC can be found in: '/NVME/decrypted/ukbb/fundus/2021_10_rawMeasurements/'
* All the QC files can be found in: '/NVME/decrypted/ukbb/fundus/qc/'

* All the files with the phenotypes measurement (with images names) can be found in: '/NVME/decrypted/ukbb/fundus/phenotypes/'
* All the phenofiles and the phenofiles_qqnorm can be found in: '/HDD/data/UKBiob/phenofiles/'

* All the GWAS results can be found in: '/HDD/data/UKBiob/GWAS/' 
* * All the Manhatan, and qqplots an results can be found in: '/HDD/data/UKBiob/GWAS/*YourGWAS*/Manhattan_QQplots/'
* * All the PascalX results can be found in: '/HDD/data/UKBiob/GWAS/*YourGWAS*/PascalX/'
* * All the ldsr results can be found in: '/HDD/data/UKBiob/GWAS/*YourGWAS*/ldscr/'

* All the UKBB data can be found in: '/NVME/decrypted/ukbb/labels/'
* A merge file between the diseases and the phenotypes of interest for the MLR can be also found in: '/NVME/decrypted/ukbb/labels/'

* Drive Ground Truth for bifurcations, crossings, and end points can be found in: '/HDD/data/Other_datasets/DRIVE/RetinalFeatures_bif_cross'

# CODE - FROM RETINA IMAGES TO TRAITS (only needed the folders: 'configs', 'input' and 'preprocessing'): 

## Requirements:
* You will need to download WNET 
* Matlab licence (if you have not acess there are still some traits you can measure)
* TO DO: Create a file with all the packages needed

## Pipeline:
1- Modify `configs/config_local.sh`

2 - Run `preprocessing/ClassifyAVLwnet.sh`. 
Output: AV maps for your images.  By default in the folder: `input/*your_data_set*/AV_maps/`
Code: `bash ClassifyAVLwnet.sh` 

3 - Run `preprocessing/MeasureVessels.sh`. 
Output: Centerlines output.  By default in the folder: `output/*your_data_set*/ARIA_output/`
Code: `bash MeasureVessels.sh`

3.a - NOTE: For some phenotypes you would need the Optic Disk poistion. For DRIVE we already provide a file, but if you want to compute the it for other Datasets we provide additional code for 'Optic disc prediction using training weights from UK Biobank':
* 1 - Generate the test dataset in required square 256x256px format: `preprocessing/optic-nerve-cnn/scripts/TEST_organize_datasets.ipynb` : Point to your dataset of choice, define output path, and modify the number of cores, and then run all the cells.
* 2 - Predict using the generated dataset : `preprocessing/optic-nerve-cnn/scripts/TEST_unet_od_on_ukbiobank.ipynb` : Change indir,outdir and no_batches accordingly, then run sequentially.

4 - Run `preprocessing/run_measurePhenotype.sh`. 
Output: Trait measurements. By default in the folder: `output/*your_data_set*/phenotypes_all/`
Code: `bash run_measurePhenotype.sh` 

### Some possible errors and reminders:
* LWNET, no image generated in DATASET_AV_maps:   AttributeError: module 'skimage.draw' has no attribute 'circle' . You need "your_python_dir/python -m pip install scikit-image==0.16.2" and python3

* python3 -m pip install --upgrade Pillow

* If you are not familiar with bash scripts and you want to change the code, the spaces are very imports!(Avoid when define variables, and use then for conditions)

# CODE - COMPLEMENTARY ANALYSIS (only needed the folder: 'complementary'):  

## GWAS postprocessing:
This folder include code to plot the Manhattan and QQplots and to prepare: PascalX, LDSCR and LD prune input.
HOW to run it?:


Besides, in the subfolders:

* `ldsr_correlation` you can find code to compute the h^2 and the genetic correlation (you will need additionality to have ldsr code!)

HOW to run it?:

* `common_genes_path_snps` you can find code to explore the SNPs, genes, and set of genes that several phenotypes have in common (using PascalX results as input)

HOW to run it?:


## Traits association with diseases:

## Traits different time points:
For some subjects we have different retina images at differen time points. In this folder we included code to analyse how the different traits (phenotypes) for these subjects vary depending on the time point. 
Also, you can find code to see how different are the traits for this subjects between the right and the left eye. (This is to see how much natural variation we can expect and analyse better the differetn time points)

HOW to run it?:

## Traits main characterization:
In here you can find code to obtain histograms and pairplots of your phenotypes.

HOW to run it?:


# PHENOTYPES (TRAITS) MEASURED 
## Main traits measured:
* Median diameter of: all the vessels, only arteries, only veins (' ')
* Median tortuosity (DF) of: all the vessels, only arteries, only veins (' ')
* Ratio between the diameters of the arteies and the diameters of the veins (' ')
* Ratio between the tortuosity of the arteies and the tortuosity of the veins (' ')
* Number of bifurcations and branching ('bifurcations')
* Main Temporal Venular Angle ('tva') and Main Temporal Arteriolar Angle ('taa') 
* CRAE and CRVE 
* Variability on the diameter and on the tortuosity
* Fractal Dimensionality ()
* Vascular Density () 
* Vascular Density () 

## Baseline traits measured:
* Intensity variability
* others?
* The follow are probably to delete:  N_green_pixels, N_green_segments, OD_segments, etc
