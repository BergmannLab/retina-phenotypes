# Sofia 10-06-2022
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle5 as pickle
import seaborn as sns
import functions_time_points as f_t

#### Analize subjects with images at different instances

dir_ukb_csv_1 = '/NVME/decrypted/ukbb/labels/1_data_extraction/ukb34181.csv'
phenofiles_dir = '/NVME/decrypted/ukbb/fundus/phenotypes/lwnet_QC2/'
dir_save_results = '~/retina-phenotypes/complementary/traits_different_time_points/results/' #to modify
QC_dir = '/NVME/decrypted/ukbb/fundus/qc/' #to modify
QC_file = 'newQC_lwnetDecile3.txt' # to modify

df_QC = pd.read_csv(QC_dir + QC_file, sep=',', header=None)
df_QC.columns = ['image_QC']

# Select the files
list_right_eyes = ['eid', '21016-0.0', '21016-1.0', '21016-0.1', '21016-1.1']
list_left_eyes = ['eid', '21015-0.0', '21015-1.0', '21015-0.1', '21015-1.1']

# Create dfs
df_data_right,df_data_left = f_t.create_ukbb_dfs(dir_ukb_csv_1,list_right_eyes, list_left_eyes)

# Replace na by 0 to avoid nans issues
df_data_right.fillna(0, inplace=True)
df_data_left.fillna(0, inplace=True)

# Only select subjects with: (0.0 or 0.1) and (1.0 or 1.1) !=nan
df_right_intersection = df_data_right[((df_data_right['21016-0.0']!=0)|(df_data_right['21016-0.1']!=0)) & 
                                    ((df_data_right['21016-1.0']!=0)|(df_data_right['21016-1.1']!=0))]
df_left_intersection = df_data_left[((df_data_left['21015-0.0']!=0)|(df_data_left['21015-0.1']!=0)) & 
                                    ((df_data_left['21015-1.0']!=0)|(df_data_left['21015-1.1']!=0))]

print('len df_right_intersection:',len(df_right_intersection), ', and len df_left_intersection:',len(df_left_intersection))

# Uniformizate keys, from eid to image names
df_right_intersection['image_00']=df_right_intersection['eid'].astype(str) + '_21016_0_0.png'
df_right_intersection['image_01']=df_right_intersection['eid'].astype(str) + '_21016_0_1.png'
df_right_intersection['image_10']=df_right_intersection['eid'].astype(str) + '_21016_1_0.png'
df_right_intersection['image_11']=df_right_intersection['eid'].astype(str) + '_21016_1_1.png'

df_left_intersection['image_00']=df_left_intersection['eid'].astype(str) + '_21015_0_0.png'
df_left_intersection['image_01']=df_left_intersection['eid'].astype(str) + '_21015_0_1.png'
df_left_intersection['image_10']=df_left_intersection['eid'].astype(str) + '_21015_1_0.png'
df_left_intersection['image_11']=df_left_intersection['eid'].astype(str) + '_21015_1_1.png'

# Read phenotypes per image (Make sure that the phenotypes_files and pheno_ARIA, etc match!)


phenotypes_files = ['2022-06-08_bifurcations.csv', '/2022-06-08_taa.csv', '2022-06-08_tva.csv', 
                    '/2022-06-09_ratios_aria_phenotypes.csv', '/2022-06-08_ratios_CRAE_CRVE.csv',
                    '/2022-06-08_diameter_variability.csv', '/2022-06-08_fractal_dimension.csv', 
                    '/2022-06-08_vascular_density.csv']

pheno_N_bif, pheno_tAA, pheno_tVA, pheno_ARIA, pheno_CRAVE, pheno_diam_var, pheno_FD, pheno_VD = f_t.read_phenotypes_per_image(phenofiles_dir, phenotypes_files)

l_pheno_of_interest = [pheno_N_bif, pheno_ARIA, pheno_ARIA, pheno_tAA, pheno_tVA, pheno_CRAVE, 
                       pheno_CRAVE, pheno_diam_var, pheno_FD, pheno_VD, pheno_ARIA, pheno_ARIA, pheno_CRAVE] 

l_pheno_name= ['bifurcations', 'medianDiameter_all', 'DF_all', 'mean_angle_taa', 'mean_angle_tva', 'median_CRAE', 'median_CRVE','D_std_std', 'slope', 'VD_orig_all', 'ratio_AV_medianDiameter', 'ratio_AV_DF', 'ratio_CRAE_CRVE']

#####OLD:
#phenotypes_files = ['/2021-12-28_ARIA_phenotypes.csv', '/2022-02-01_N_green_pixels.csv', '/2022-02-04_bifurcations.csv', '/2022-02-13_tVA_phenotypes.csv', '/2022-02-14_tAA_phenotypes.csv', '/2022-02-17_NeovasOD_phenotypes.csv', "/2022-02-21_green_pixels_over_total_OD_phenotypes.csv", "/2022-02-21_N_green_segments_phenotypes.csv", "/2021-11-30_fractalDimension.csv", "/2022-04-12_vascular_density.csv")]

#pheno_ARIA,pheno_N_green,pheno_N_bif,pheno_tVA,pheno_tAA,pheno_NeoOD,pheno_greenOD,pheno_N_green_seg,pheno_FD,pheno_#VD = f_t.read_phenotypes_per_image(phenofiles_dir, phenotypes_files)
#########


for i in range(len(l_pheno_of_interest)):
    pheno_of_interest = l_pheno_of_interest[i]
    pheno_name = l_pheno_name[i]

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
    
    print(df_right_intersection_00.head(2),
                        df_right_intersection_10.head(2), df_left_intersection_00.head(2), df_left_intersection_10.head(2), pheno_name)
    
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


    ### PLOTS: (without checking the QC)
    #f_t.plt_RL_00_10(df_right_intersection_00, df_right_intersection_10, df_left_intersection_00, #
    #df_left_intersection_10, pheno_name)

    ## Create datafields with (00-10) and |00-10|:
    df_right_intersection_all, df_left_intersection_all=f_t.create_different_time_points_df(df_right_intersection_00,
                        df_right_intersection_10, df_left_intersection_00, df_left_intersection_10, pheno_name)

    ## PLOTS: (Only the images in the QC)
    f_t.plt_RL_00_menis_10(df_right_intersection_all, df_left_intersection_all, pheno_name, ' No QC')

    ## Save file before QC:
    df_right_intersection_all.to_csv(dir_save_results + pheno_name + '_right_before_QC.csv') 
    df_left_intersection_all.to_csv(dir_save_results + pheno_name + '_left_before_QC.csv')  

    ## Filter by QC
    df_right_QC, df_left_QC = f_t.create_different_time_points_df2(df_QC, df_right_intersection_all, 
                                                               df_left_intersection_all, pheno_name)

    ### Saving files after QC
    df_right_QC.to_csv(dir_save_results + pheno_name + '_right_QC.csv') 
    df_left_QC.to_csv(dir_save_results + pheno_name + '_left_QC.csv')

    #f_t.plt_RL_00_menis_10(df_right_QC, df_left_QC, pheno_name, ' with QC')