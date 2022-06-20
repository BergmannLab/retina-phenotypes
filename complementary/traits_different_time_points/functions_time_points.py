# Sofia 10-06-2022
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle5 as pickle
import seaborn as sns

def create_ukbb_dfs(dir_ukb_csv_1,list_right_eyes, list_left_eyes):
    df_data = pd.read_csv(dir_ukb_csv_1, sep=',')
    df_data_right = df_data[list_right_eyes]
    df_data_left = df_data[list_left_eyes]
    return df_data_right,df_data_left


def read_phenotypes_per_image(phenofiles_dir, phenotypes_files):
    pheno_N_bif = pd.read_csv(phenofiles_dir + phenotypes_files[0])
    pheno_tAA = pd.read_csv(phenofiles_dir + phenotypes_files[1])
    pheno_tVA = pd.read_csv(phenofiles_dir+ phenotypes_files[2])
    pheno_ARIA = pd.read_csv(phenofiles_dir+ phenotypes_files[3])
    pheno_CRAVE = pd.read_csv(phenofiles_dir + phenotypes_files[4])
    pheno_diam_var = pd.read_csv(phenofiles_dir + phenotypes_files[5])
    pheno_FD = pd.read_csv(phenofiles_dir + phenotypes_files[6])
    pheno_VD = pd.read_csv(phenofiles_dir + phenotypes_files[7])

    ## Add name to the first column and solve a misslabeling! (This should be solved!)
    pheno_N_bif.rename(columns={pheno_N_bif.columns[0]: 'image'}, inplace=True)
    pheno_tAA.rename(columns={pheno_tAA.columns[0]: 'image'}, inplace=True)
    pheno_tVA.rename(columns={pheno_tVA.columns[0]: 'image'}, inplace=True)
    pheno_ARIA.rename(columns={pheno_ARIA.columns[0]: 'image'}, inplace=True)
    pheno_CRAVE.rename(columns={pheno_CRAVE.columns[0]: 'image'}, inplace=True)
    pheno_diam_var.rename(columns={pheno_diam_var.columns[0]: 'image'}, inplace=True)
    pheno_FD.rename(columns={pheno_FD.columns[0]: 'image'}, inplace=True)
    pheno_VD.rename(columns={pheno_VD.columns[0]: 'image'}, inplace=True)

    #print(pheno_N_bif.columns, pheno_tAA.columns, pheno_tVA.columns)
    return pheno_N_bif, pheno_tAA, pheno_tVA, pheno_ARIA, pheno_CRAVE, pheno_diam_var, pheno_FD, pheno_VD 


def plt_RL_00_10(df_right_intersection_00, df_right_intersection_10, df_left_intersection_00, 
                 df_left_intersection_10, pheno_name):
    y1 = df_right_intersection_00[pheno_name+'_00']
    print('len(df_right_intersection_00): ', len(df_right_intersection_00))

    y2 = df_left_intersection_00[pheno_name+'_00']
    print('len(df_left_intersection_00): ', len(df_left_intersection_00))

    y3 = df_right_intersection_10[pheno_name+'_10']
    print('len(df_right_intersection_10): ', len(df_right_intersection_10))

    y4 = df_left_intersection_10[pheno_name+'_10']
    print('len(df_left_intersection_10): ', len(df_left_intersection_10))

    fig = sns.kdeplot(y1, shade=True)#, color="r")
    fig = sns.kdeplot(y2, shade=True)#, color="b")
    fig = sns.kdeplot(y3, shade=True)#, color="r")
    fig = sns.kdeplot(y4, shade=True)#, color="b")
    plt.legend(['R00', 'L00', 'R10', 'L10'])
    #plt.axes()
    #plt.title('Right eye')
    plt.show()

def plt_RL_00_menis_10(df_right_intersection_all, df_left_intersection_all, pheno_name, QC):
    y_a = df_right_intersection_all['00_menos_10']
    y_b = df_left_intersection_all['00_menos_10']

    print('len(df_right_intersection_all): ', len(df_right_intersection_all),' and len(df_left_intersection_all): ', len(df_left_intersection_all))

    fig = sns.kdeplot(y_a, shade=True)#, color="r")
    fig = sns.kdeplot(y_b, shade=True)#, color="b")
    plt.legend(['R', 'L'])
    #plt.axes()
    plt.title(pheno_name+ QC)
    plt.show()


def create_different_time_points_df(df_right_intersection_00, df_right_intersection_10, df_left_intersection_00, 
                                    df_left_intersection_10, pheno_name):
    df_right_intersection_all = pd.DataFrame([])
    df_left_intersection_all = pd.DataFrame([])

    ## Merge 00 and 10
    # Right: 
    df_right_intersection_all = df_right_intersection_00.merge(df_right_intersection_10, how='inner', on='image_00', 
                                                               suffixes=('', '_y'))
    df_right_intersection_all.drop(df_right_intersection_all.filter(regex='_y$').columns.tolist(),axis=1, 
                                   inplace=True)
    print('len(df_right_intersection_all)', len(df_right_intersection_all))

    # Left: 
    df_left_intersection_all = df_left_intersection_00.merge(df_left_intersection_10, how='inner', on='image_00', 
                                                             suffixes=('', '_y'))
    df_left_intersection_all.drop(df_left_intersection_all.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
    print('len(df_left_intersection_all)', len(df_left_intersection_all))

    ### Create two new columns abs(00-10) and |00-10|
    #Right 
    df_right_intersection_all['00_menos_10']=(df_right_intersection_all[pheno_name+'_00']-
                                              df_right_intersection_all[pheno_name+'_10'])
    df_right_intersection_all['00_menos_10']=(df_right_intersection_all['00_menos_10']-df_right_intersection_all['00_menos_10'].mean())/df_right_intersection_all['00_menos_10'].std()
    #df_right_intersection_all['00_menos_10'].hist(bins=20)
    print('len(df_right_intersection_all)', len(df_right_intersection_all))

    #Left 
    df_left_intersection_all['00_menos_10']=(df_left_intersection_all[pheno_name+'_00']-
                                             df_left_intersection_all[pheno_name+'_10'])
    df_left_intersection_all['00_menos_10']=(df_left_intersection_all['00_menos_10']-df_left_intersection_all['00_menos_10'].mean())/df_left_intersection_all['00_menos_10'].std()
    print('len(df_left_intersection_all)', len(df_left_intersection_all))

    return df_right_intersection_all, df_left_intersection_all

def create_different_time_points_df2(df_QC, df_right_intersection_all, df_left_intersection_all, pheno_name):
    df_right_QC = pd.DataFrame([])
    df_left_QC = pd.DataFrame([])

    ## Merge 00 and 10
    #Right:
    df_right_QC = pd.merge(df_QC, df_right_intersection_all, how='inner', left_on=['image_QC'],right_on=['image_00'], 
                           suffixes=('', '_y'))
    df_right_QC.drop(df_right_QC.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
    df_right_QC = pd.merge(df_QC, df_right_intersection_all, how='inner', left_on=['image_QC'],right_on=['image_10'], 
                           suffixes=('', '_y'))
    df_right_QC.drop(df_right_QC.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

    df_right_QC['abs00_menos_10']=abs(df_right_QC[pheno_name+'_00']-df_right_QC[pheno_name+'_10'])
    df_right_QC['abs00_menos_10']=(df_right_QC['abs00_menos_10']-df_right_QC['abs00_menos_10'].mean())/df_right_QC['abs00_menos_10'].std()
    print('len(df_right_QC): ', len(df_right_QC))
    #df_right_QC['abs00_menos_10'].hist(bins=20)

    df_right_QC['00_menos_10']=(df_right_QC[pheno_name+'_00']-df_right_QC[pheno_name+'_10'])
    df_right_QC['00_menos_10']=(df_right_QC['00_menos_10']-df_right_QC['00_menos_10'].mean())/df_right_QC['00_menos_10'].std()
    #df_right_QC['00_menos_10'].hist(bins=20)

    #Left:
    df_left_QC = pd.merge(df_QC, df_left_intersection_all, how='inner', left_on=['image_QC'],right_on=['image_00'], 
                          suffixes=('', '_y'))
    df_left_QC.drop(df_left_QC.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
    df_left_QC = pd.merge(df_QC, df_left_intersection_all, how='inner', left_on=['image_QC'],right_on=['image_10'], 
                          suffixes=('', '_y'))
    df_left_QC.drop(df_left_QC.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

    df_left_QC['abs00_menos_10']=abs(df_left_QC[pheno_name+'_00']-df_left_QC[pheno_name+'_10'])
    df_left_QC['abs00_menos_10']=(df_left_QC['abs00_menos_10']-df_left_QC['abs00_menos_10'].mean())/df_left_QC['abs00_menos_10'].std()
    print('len(df_left_QC): ', len(df_left_QC))
    #df_left_QC['abs00_menos_10'].hist(bins=20)

    df_left_QC['00_menos_10']=(df_left_QC[pheno_name+'_00']-df_left_QC[pheno_name+'_10'])
    df_left_QC['00_menos_10']=(df_left_QC['00_menos_10']-df_left_QC['00_menos_10'].mean())/df_left_QC['00_menos_10'].std()
    #df_left_QC['00_menos_10'].hist(bins=20)

    return df_right_QC, df_left_QC