import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle5 as pickle
import seaborn as sns

figsize_1=10
figsize_2=5
font_size=8
axes_titlesize=12
axes_labelsize=12


def create_ukbb_dfs(dir_ukb_csv_1,list_right_eyes, list_left_eyes):
    df_data = pd.read_csv(dir_ukb_csv_1, sep=',')
    df_data_right = df_data[list_right_eyes]
    df_data_left = df_data[list_left_eyes]
    return df_data_right,df_data_left

def read_phenotypes_per_image(phenofiles_dir, phenotype_file):
    pheno_def = pd.read_csv(phenofiles_dir + phenotype_file)
    pheno_def.rename(columns={pheno_def.columns[0]: 'image'}, inplace=True)
    return pheno_def


def plt_RL_00_10(df_right_intersection_00, df_right_intersection_10, df_left_intersection_00, 
                 df_left_intersection_10, pheno_name):
    y1 = df_right_intersection_00[pheno_name+'_00']
    #print('len(df_right_intersection_00): ', len(df_right_intersection_00))

    y2 = df_left_intersection_00[pheno_name+'_00']
    #print('len(df_left_intersection_00): ', len(df_left_intersection_00))

    y3 = df_right_intersection_10[pheno_name+'_10']
    #print('len(df_right_intersection_10): ', len(df_right_intersection_10))

    y4 = df_left_intersection_10[pheno_name+'_10']
    #print('len(df_left_intersection_10): ', len(df_left_intersection_10))

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

    #print('len(df_right_intersection_all): ', len(df_right_intersection_all),' and len(df_left_intersection_all): ', len(df_left_intersection_all))

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
    #print('len(df_right_intersection_all)', len(df_right_intersection_all))

    # Left: 
    df_left_intersection_all = df_left_intersection_00.merge(df_left_intersection_10, how='inner', on='image_00', 
                                                             suffixes=('', '_y'))
    df_left_intersection_all.drop(df_left_intersection_all.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
    #print('len(df_left_intersection_all)', len(df_left_intersection_all))

    ### Create two new columns abs(00-10) and |00-10|
    #Right 
    df_right_intersection_all['00_menos_10']=(df_right_intersection_all[pheno_name+'_00']-
                                              df_right_intersection_all[pheno_name+'_10'])
    df_right_intersection_all['00_menos_10']=(df_right_intersection_all['00_menos_10']-df_right_intersection_all['00_menos_10'].mean())/df_right_intersection_all['00_menos_10'].std()
    #df_right_intersection_all['00_menos_10'].hist(bins=20)
    #print('len(df_right_intersection_all)', len(df_right_intersection_all))

    #Left 
    df_left_intersection_all['00_menos_10']=(df_left_intersection_all[pheno_name+'_00']-
                                             df_left_intersection_all[pheno_name+'_10'])
    df_left_intersection_all['00_menos_10']=(df_left_intersection_all['00_menos_10']-df_left_intersection_all['00_menos_10'].mean())/df_left_intersection_all['00_menos_10'].std()
    #print('len(df_left_intersection_all)', len(df_left_intersection_all))

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
    #print('len(df_right_QC): ', len(df_right_QC))
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
    #print('len(df_left_QC): ', len(df_left_QC))
    #df_left_QC['abs00_menos_10'].hist(bins=20)

    df_left_QC['00_menos_10']=(df_left_QC[pheno_name+'_00']-df_left_QC[pheno_name+'_10'])
    df_left_QC['00_menos_10']=(df_left_QC['00_menos_10']-df_left_QC['00_menos_10'].mean())/df_left_QC['00_menos_10'].std()
    #df_left_QC['00_menos_10'].hist(bins=20)

    return df_right_QC, df_left_QC

def right_left(dir_ukb_csv_1):
    # Select the files
    list_right_eyes = ['eid', '21016-0.0', '21016-1.0', '21016-0.1', '21016-1.1']
    list_left_eyes = ['eid', '21015-0.0', '21015-1.0', '21015-0.1', '21015-1.1']

    # Create dfs
    df_data_right,df_data_left = create_ukbb_dfs(dir_ukb_csv_1,list_right_eyes, list_left_eyes)

    # Replace na by 0 to avoid nans issues
    df_data_right.fillna(0, inplace=True)
    df_data_left.fillna(0, inplace=True)

    # Only select subjects with: (0.0 or 0.1) and (1.0 or 1.1) !=nan
    df_right_intersection = df_data_right[((df_data_right['21016-0.0']!=0)|(df_data_right['21016-0.1']!=0)) & 
                                        ((df_data_right['21016-1.0']!=0)|(df_data_right['21016-1.1']!=0))]
    df_left_intersection = df_data_left[((df_data_left['21015-0.0']!=0)|(df_data_left['21015-0.1']!=0)) & 
                                        ((df_data_left['21015-1.0']!=0)|(df_data_left['21015-1.1']!=0))]

    #print('len df_right_intersection:',len(df_right_intersection), ', and len df_left_intersection:',len(df_left_intersection))

    # Uniformizate keys, from eid to image names
    df_right_intersection['image_00']=df_right_intersection['eid'].astype(str) + '_21016_0_0.png'
    df_right_intersection['image_01']=df_right_intersection['eid'].astype(str) + '_21016_0_1.png'
    df_right_intersection['image_10']=df_right_intersection['eid'].astype(str) + '_21016_1_0.png'
    df_right_intersection['image_11']=df_right_intersection['eid'].astype(str) + '_21016_1_1.png'

    df_left_intersection['image_00']=df_left_intersection['eid'].astype(str) + '_21015_0_0.png'
    df_left_intersection['image_01']=df_left_intersection['eid'].astype(str) + '_21015_0_1.png'
    df_left_intersection['image_10']=df_left_intersection['eid'].astype(str) + '_21015_1_0.png'
    df_left_intersection['image_11']=df_left_intersection['eid'].astype(str) + '_21015_1_1.png'
    return df_right_intersection, df_left_intersection

def prepare_data_for_plot(dir_input, pheno_name, RorL,type):
    df = pd.read_csv(dir_input+pheno_name+'_ventile_5_'+RorL+'.csv')
    y1 = df[type]
    return y1


def plot_density_time_dif(dir_input, type, pheno_list):
    ### Density plots
    l_legends= pheno_list
    xlabel_name='t0 - t1'
    title = type

    fig, axs = plt.subplots(2,2,figsize=(figsize_1,figsize_2))
    plt.rcParams['font.size'] = font_size
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    sns.set_context("paper", rc={"font.size":font_size,"axes.titlesize":axes_titlesize,"axes.labelsize":axes_labelsize})
    for phenotype in pheno_list:
        df_y = prepare_data_for_plot(dir_input, phenotype, 'right', type)
        fig = sns.kdeplot(df_y, shade=True, alpha=0.003,linewidth = 1.5)#, color="r")
    plt.legend(l_legends, loc='best')
    plt.title('Right eye')
    plt.xlabel(xlabel_name)

    plt.subplot(1, 2, 2) # index 2
    sns.set_context("paper", rc={"font.size":font_size,"axes.titlesize":axes_titlesize,"axes.labelsize":axes_labelsize})
    for phenotype in pheno_list:
        df_y = prepare_data_for_plot(dir_input, phenotype, 'left', type)
        fig = sns.kdeplot(df_y, shade=True, alpha=0.003,linewidth = 1.5)#, color="r")
    plt.legend(l_legends, loc='best')
    plt.title('Left eye')
    plt.xlabel(xlabel_name)
    plt.tight_layout()
    plt.savefig(dir_output+'/Different_time_points_' + str(pheno_list)+'_density.pdf',
               facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
    plt.close()
    
def split_list(a_list):
    third = len(a_list)//3
    return a_list[:third], a_list[third:]

def split_list_half(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def prepare_data_for_plot(dir_input, pheno_name, RorL,type):
    df = pd.read_csv(dir_input+pheno_name+'_ventile_5_'+RorL+'.csv')
    y1 = df[type]
    return y1


def plot_density_time_dif(dir_input, type, pheno_list,name):
    ### Density plots
    l_legends= pheno_list
    xlabel_name='t0 - t1'
    title = type

    fig, axs = plt.subplots(2,2,figsize=(figsize_1,figsize_2))
    plt.rcParams['font.size'] = font_size
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    sns.set_context("paper", rc={"font.size":font_size,"axes.titlesize":axes_titlesize,"axes.labelsize":axes_labelsize})
    for phenotype in pheno_list:
        df_y = prepare_data_for_plot(dir_input, phenotype, 'right', type)
        fig = sns.kdeplot(df_y, shade=True, alpha=0.003,linewidth = 1.5)#, color="r")
    plt.legend(l_legends, loc='best')
    plt.title('Right eye')
    plt.xlabel(xlabel_name)

    plt.subplot(1, 2, 2) # index 2
    sns.set_context("paper", rc={"font.size":font_size,"axes.titlesize":axes_titlesize,"axes.labelsize":axes_labelsize})
    for phenotype in pheno_list:
        df_y = prepare_data_for_plot(dir_input, phenotype, 'left', type)
        fig = sns.kdeplot(df_y, shade=True, alpha=0.003,linewidth = 1.5)#, color="r")
    plt.legend(l_legends, loc='best')
    plt.title('Left eye')
    plt.xlabel(xlabel_name)
    plt.tight_layout()
    plt.savefig(dir_input+'/Different_time_points_' + str(name)+'_density.pdf',
               facecolor=fig.get_facecolor(), edgecolor='none')
    plt.show()
    plt.close()
