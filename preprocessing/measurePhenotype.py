import itertools
import os, sys
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import csv
import PIL
import math
import cv2
from skimage import data, io, filters
from datetime import datetime
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from scipy import stats
from multiprocessing import Pool
from PIL import Image
from matplotlib import cm
from IDVP import Measure_IDVP

DATE = datetime.now().strftime("%Y-%m-%d")


def extract_class_init():
    ## Command line argumetns
    phenotype_dir = sys.argv[2]
    aria_measurements_dir = sys.argv[3]
    OD_POSITIONS = sys.argv[5]

    ## Parameters for temporal angle
    delta = float(sys.argv[7]) 
    R_0 = float(sys.argv[8])
    min_ta = float(sys.argv[9])
    max_ta = float(sys.argv[10]) 
    lower_accept = float(sys.argv[11]) 
    upper_accept = float(sys.argv[12]) 
    ## Parameters for bifurcations
    norm_acceptance = float(sys.argv[13]) 
    neighborhood_cte = float(sys.argv[14])
    ## Parameter for N main vessels
    limit_diameter_main = float(sys.argv[15]) 
    ## Parameter for Vascular Density
    mask_radius=int(sys.argv[16])  # works for UKBB, may be adapted in other datasets, though only used for PBV (percent annotated as blood vessels) phenotype
    return phenotype_dir,aria_measurements_dir,OD_POSITIONS,delta,R_0,min_ta,max_ta,lower_accept,upper_accept,norm_acceptance,neighborhood_cte,limit_diameter_main,mask_radius



if __name__ == '__main__':
    qcFile = sys.argv[1]
    traits = sys.argv[6].split(',')
    lwnet_dir = sys.argv[4] 
    n_cpu=int(sys.argv[17])

    # all the images
    imgfiles = pd.read_csv(qcFile, header=None)
    imgfiles = imgfiles[0].values
    
    # development param
    imgfiles_length = len(imgfiles)  # len(imgfiles) is default

    # Read class attributes
    phenotype_dir, aria_measurements_dir, OD_POSITIONS, delta, R_0, min_ta, max_ta, lower_accept, upper_accept, norm_acceptance, neighborhood_cte, limit_diameter_main, mask_radius = extract_class_init()

    # Initializate class attributes
    m_IDVP = Measure_IDVP(phenotype_dir, aria_measurements_dir, OD_POSITIONS, delta, R_0, min_ta, max_ta, lower_accept, upper_accept, norm_acceptance,
    neighborhood_cte, limit_diameter_main, mask_radius)


    for function_to_execute in traits:
        print("\nStarting function", function_to_execute, '\n')

        # computing the phenotype as a parallel process
        os.chdir(lwnet_dir)
        pool = Pool(n_cpu)

        if function_to_execute in {'taa', 'tva'}:
            filter_tva_taa = 1 if function_to_execute == 'taa' else (-1 if function_to_execute == 'tva' else None)
            imgages_and_filter = list(zip(imgfiles[:imgfiles_length], imgfiles_length * [filter_tva_taa]))
            out = pool.map(m_IDVP.main_tva_or_taa, imgages_and_filter)
        elif function_to_execute in {'CRAE', 'CRVE'}:
            filter_CRAE_CRVE = 1 if function_to_execute == 'CRAE' else (-1 if function_to_execute == 'CRVE' else None)
            imgages_and_filter = list(zip(imgfiles[:imgfiles_length], imgfiles_length * [filter_CRAE_CRVE]))
            out = pool.map(m_IDVP.main_CRAE_CRVE, imgages_and_filter)
        elif function_to_execute in {'N_main_arteires', 'N_main_veins'}:
            filter_N_main = 1 if function_to_execute == 'N_main_arteires' else (-1 if function_to_execute == 'N_main_veins' else None)
            imgages_and_filter = list(zip(imgfiles[:imgfiles_length], imgfiles_length * [filter_N_main]))
            out = pool.map(m_IDVP.main_N_main_vessels, imgages_and_filter)
        elif function_to_execute == 'bifurcations':
            out = pool.map(m_IDVP.main_bifurcations, imgfiles[:imgfiles_length])
        elif function_to_execute == 'diameter_variability':
            out = pool.map(m_IDVP.main_diameter_variability, imgfiles[:imgfiles_length]) 
        elif function_to_execute == 'aria_phenotypes':
            out = pool.map(m_IDVP.main_aria_phenotypes, imgfiles[:imgfiles_length])
        elif function_to_execute == 'fractal_dimension':
            out = pool.map(m_IDVP.main_fractal_dimension, imgfiles[:imgfiles_length])
        elif function_to_execute == 'vascular_density':
            out = pool.map(m_IDVP.main_vascular_density, imgfiles[:imgfiles_length])
        elif function_to_execute == 'baseline':
            out = pool.map(m_IDVP.main_baseline_traits, imgfiles[:imgfiles_length]) 
        else:
            out = None

        pool.close()
        m_IDVP.create_output_(out, imgfiles, function_to_execute, imgfiles_length) if out else print("WARNING Your function is ", function_to_execute, ".\nIf it is a ratio, it is al right, else, this function does not exist. Options:taa,tva,CRAE,CRVE,ratios_CRAE_CRVE,bifurcations,diameter_variability,aria_phenotypes,ratios,fractal_dimension,vascular_density,baseline,neo_vascularization,N_main_arteires,N_main_veins", sep='')
        
        if function_to_execute == 'ratios':  # For measure ratios as qqnorm(ratio)
            df_data = pd.read_csv(phenotype_dir+DATE+"_aria_phenotypes.csv", sep=',')
            df_data = df_data[['Unnamed: 0', 'medianDiameter_all', 'medianDiameter_artery', 'medianDiameter_vein', 'DF_all', 'DF_artery', 'DF_vein', 'DF_longestFifth_artery', 'DF_longestFifth_vein', 'medianDiameter_longestFifth_artery', 'medianDiameter_longestFifth_vein', 'tau2_longestFifth_artery', 'tau2_longestFifth_vein', 'tau3_longestFifth_artery', 'tau3_longestFifth_vein', 'tau4_longestFifth_artery', 'tau4_longestFifth_vein']]
            print(df_data)
            df_data['ratio_AV_medianDiameter'] = df_data['medianDiameter_artery'] / df_data['medianDiameter_vein']
            #df_data['ratio_VA_medianDiameter'] = df_data['medianDiameter_vein'] / df_data['medianDiameter_artery']
            df_data['ratio_AV_DF'] = df_data['DF_artery'] / df_data['DF_vein']
            #df_data['ratio_VA_DF'] = df_data['DF_vein'] / df_data['DF_artery']
            df_data['ratio_medianDiameter_longest'] = df_data['medianDiameter_longestFifth_artery'] / df_data['medianDiameter_longestFifth_vein']
            df_data['ratio_DF_longest'] = df_data['DF_longestFifth_artery'] / df_data['DF_longestFifth_vein']
            df_data['ratio_tau2_longest'] = df_data['tau2_longestFifth_artery'] / df_data['tau2_longestFifth_vein']
            df_data['ratio_tau3_longest'] = df_data['tau3_longestFifth_artery'] / df_data['tau3_longestFifth_vein']
            df_data['ratio_tau4_longest'] = df_data['tau4_longestFifth_artery'] / df_data['tau4_longestFifth_vein']
            # only select the ratios
            df_data.drop(['medianDiameter_all', 'medianDiameter_artery', 'medianDiameter_vein', 'DF_all', 'DF_artery', 'DF_vein', 'DF_longestFifth_artery', 
                          'DF_longestFifth_vein', 'medianDiameter_longestFifth_artery', 'medianDiameter_longestFifth_vein', 'tau2_longestFifth_artery', 'tau2_longestFifth_vein',
                          'tau3_longestFifth_artery', 'tau3_longestFifth_vein', 'tau4_longestFifth_artery', 'tau4_longestFifth_vein'], axis=1, inplace=True)
            df_data.to_csv(phenotype_dir + DATE + "_ratios_aria_phenotypes.csv", sep=',', index=False)
            
        elif function_to_execute == 'ratios_CRAE_CRVE':
            df_data_CRAE = pd.read_csv(phenotype_dir+DATE+"_CRAE.csv", sep=',')
            df_data_CRAE.rename(columns={ df_data_CRAE.columns[0]: "Unnamed: 0" }, inplace = True)
            df_data_CRAE.rename(columns={'median_CRE': 'median_CRAE', 'eq_CRE': 'eq_CRAE', 'CRE': 'CRAE'}, inplace=True)
            
            df_data_CRVE = pd.read_csv(phenotype_dir+DATE+"_CRVE.csv", sep=',')
            df_data_CRVE.rename(columns={ df_data_CRVE.columns[0]: "Unnamed: 0" }, inplace = True)
            df_data_CRVE.rename(columns={'median_CRE': 'median_CRVE', 'eq_CRE': 'eq_CRVE', 'CRE': 'CRVE'}, inplace=True)
            df_merge=df_data_CRAE.merge(df_data_CRVE, how='inner', on='Unnamed: 0')
            df_merge['ratio_median_CRAE_CRVE'] = df_merge['median_CRAE'] / df_merge['median_CRVE']
            df_merge['ratio_CRAE_CRVE'] = df_merge['eq_CRAE'] / df_merge['eq_CRVE']
            df_merge['ratio_standard_CRE'] = df_merge['CRAE'] / df_merge['CRVE']

            df_merge.to_csv(phenotype_dir + DATE + "_ratios_CRAE_CRVE.csv", sep=',', index=False)
        
        elif function_to_execute == 'ratios_VD':
            df_data_VD = pd.read_csv(phenotype_dir+DATE+"_vascular_density.csv", sep=',')
            df_data_VD.rename(columns={ df_data_VD.columns[0]: "Unnamed: 0" }, inplace = True)
            df_data_VD['ratio_VD'] = df_data_VD['VD_orig_artery'] / df_data_VD['VD_orig_vein']
            df_data_VD.to_csv(phenotype_dir + DATE + "_ratios_VD.csv", sep=',', index=False)
            
        #Renaming column names
        m_IDVP.renaming_colum_names(function_to_execute, True)
