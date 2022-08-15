## First version 10-06-2022
## last modification: 15-08-2022

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle5 as pickle
import seaborn as sns
import functions_time_points as f_t
import os
import sys
from datetime import datetime

DATE = datetime.now().strftime("%Y-%m-%d")

#### Analize subjects with images at different instances
dir_ukb_csv_1 = '/NVME/decrypted/ukbb/labels/1_data_extraction/ukb34181.csv'

phenofiles_dir = '/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/image_phenotype/current/' #sys.argv[1] #IMAGE_PHENO_DIR 
dir_save_results = '/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/different_time_points/' #sys.argv[2] # #config ADD
MAIN_LABELS = 'tau1_artery,tau1_vein,ratio_AV_DF,D_A_std,D_V_std,bifurcations,VD_orig_artery,VD_orig_vein,ratio_VD,mean_angle_taa,mean_angle_tva,eq_CRAE,eq_CRVE,ratio_CRAE_CRVE,medianDiameter_artery,medianDiameter_vein,ratio_AV_medianDiameter' #MAIN_LABELS config
ventile_num = 5 #sys.argv[4] #5 #config

l_pheno_name = MAIN_LABELS.split(",")

df_right_intersection, df_left_intersection = f_t.right_left(dir_ukb_csv_1)

# Read phenotypes per image (Make sure that the phenotypes_files and pheno_ARIA, etc match!)
phenotypes_files = os.listdir(phenofiles_dir)
print(phenotypes_files)
count=0
for pheno_file in phenotypes_files:
    count = count+ 1 
    df_pheno_aux = f_t.read_phenotypes_per_image(phenofiles_dir, pheno_file)
    if count==1:
        pheno_N_bif = df_pheno_aux
    else:
        pheno_N_bif = pheno_N_bif.merge(df_pheno_aux, how='inner', on='image') 
    if (df_pheno_aux.shape[0]!=pheno_N_bif.shape[0]):
        print('Error! Different sizes', pheno_file, df_pheno_aux.shape[0], pheno_N_bif.shape[0])
        sys.exit()
    #print(pheno_file, df_pheno_aux.shape[0], pheno_N_bif.shape[0])


for pheno_name in l_pheno_name:

    header_pheno_name=['image'] + pheno_name.split(",")
    pheno_of_interest = pheno_N_bif[header_pheno_name]

    ## right eye. Only 00 and 10
    df_right_intersection_00 = df_right_intersection.merge(pheno_of_interest, how='left', left_on=['image_00'], 
                                                           right_on=['image'], suffixes=('', '_y'))
    df_right_intersection_00.drop(df_right_intersection_00.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
   
    df_right_intersection_10 = df_right_intersection.merge(pheno_of_interest, how='left', left_on=['image_10'], 
                                                           right_on=['image'], suffixes=('', '_y'))
    df_right_intersection_10.drop(df_right_intersection_10.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

    ## left eye. Only 00 and 10
    df_left_intersection_00 = df_left_intersection.merge(pheno_of_interest, how='left', left_on=['image_00'], 
                                                         right_on=['image'], suffixes=('', '_y'))
    df_left_intersection_00.drop(df_left_intersection_00.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
    df_left_intersection_10 = df_left_intersection.merge(pheno_of_interest, how='left', left_on=['image_10'], 
                                                         right_on=['image'], suffixes=('', '_y'))
    df_left_intersection_10.drop(df_left_intersection_10.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

    
    # To avaid having the same name
    ## right 
    df_right_intersection_00.rename(columns = {pheno_name: pheno_name+'_00'}, inplace = True)
    df_right_intersection_10.rename(columns = {pheno_name: pheno_name+'_10'}, inplace = True)

    ## left
    df_left_intersection_00.rename(columns = {pheno_name: pheno_name+'_00'}, inplace = True)
    df_left_intersection_10.rename(columns = {pheno_name: pheno_name+'_10'}, inplace = True)
    
    #print(df_right_intersection_00.head(2), df_right_intersection_10.head(2), df_left_intersection_00.head(2), df_left_intersection_10.head(2), pheno_name)
    
    ################ Same for second instances ##################################################
    #df_right_intersection_01 = df_right_intersection.merge(pheno_of_interest, how='left', left_on=['image_01'], right_on=['image'], suffixes=('', '_y'))

    #df_right_intersection_01.drop(df_right_intersection_01.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

    #df_right_intersection_11 = df_right_intersection.merge(pheno_of_interest, how='left', left_on=['image_11'], right_on=['image'], suffixes=('', '_y'))
    #df_right_intersection_11.drop(df_right_intersection_11.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
    #df_left_intersection_01 = df_left_intersection.merge(pheno_of_interest, how='left', left_on=['image_01'], right_on=['image'], suffixes=('', '_y'))
    #df_left_intersection_01.drop(df_left_intersection_01.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
    #df_left_intersection_11 = df_left_intersection.merge(pheno_of_interest, how='left', left_on=['image_11'], right_on=['image'], suffixes=('', '_y'))
    #df_left_intersection_11.drop(df_left_intersection_11.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

    #df_right_intersection_01.rename(columns = {pheno_name: pheno_name+'_01'}, inplace = True)
    #df_right_intersection_11.rename(columns = {pheno_name: pheno_name+'_11'}, inplace = True)


    #df_left_intersection_01.rename(columns = {pheno_name: pheno_name+'_01'}, inplace = True)
    #df_left_intersection_11.rename(columns = {pheno_name: pheno_name+'_11'}, inplace = True)
    ####################################################################################################



    ## Create datafields with (00-10) and |00-10|:
    df_right_intersection_all, df_left_intersection_all=f_t.create_different_time_points_df(df_right_intersection_00,
                        df_right_intersection_10, df_left_intersection_00, df_left_intersection_10, pheno_name)

    ## PLOTS: 
    #f_t.plt_RL_00_menis_10(df_right_intersection_all, df_left_intersection_all, pheno_name, ' QC')

    ## Save file:
    df_right_intersection_all.to_csv(dir_save_results + pheno_name +'_ventile_'+str(ventile_num)+ '_right.csv') 
    df_left_intersection_all.to_csv(dir_save_results + pheno_name +'_ventile_'+str(ventile_num)+ '_left.csv')



pheno_list=[]

csv_time_files = os.listdir(dir_save_results)

for i in range(len(csv_time_files)):
    phenotype_name=csv_time_files[i].split('_ventile_')[0]
    pheno_list.append(phenotype_name)
pheno_list = list(dict.fromkeys(pheno_list))



pheno_1of3, pheno_23of3 = f_t.split_list(pheno_list)
pheno_2of3, pheno_3of3 = f_t.split_list_half(pheno_23of3)

if (len(pheno_list) == len(pheno_1of3) +len(pheno_2of3)+ len(pheno_3of3))!=True:
    print('Error in the pheno split to make the plot')
    exit()
    
f_t.plot_density_time_dif(dir_save_results, '00_menos_10', pheno_1of3, 'first')
f_t.plot_density_time_dif(dir_save_results, '00_menos_10', pheno_2of3, 'second')
f_t.plot_density_time_dif(dir_save_results, '00_menos_10', pheno_3of3, 'third')