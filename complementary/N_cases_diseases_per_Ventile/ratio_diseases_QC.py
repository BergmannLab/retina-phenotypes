#!/usr/bin/env python
# coding: utf-8

######## Ratio of cases controls per diseases for different ventiles

# First version: 22/07/2022 
# Last modification:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



list_file_names= ['ventiles0', 'ventiles1', 'ventiles2', 'ventiles3', 'ventiles4', 'ventiles5', 
                'ventiles6', 'ventiles7', 'ventiles8', 'ventiles9', 'ventiles10', 'ventiles11',
                'ventiles12', 'ventiles13', 'ventiles14', 'ventiles15','ventiles16', 'ventiles17', 
                'ventiles18', 'ventiles19']

lista =['age_death',
       'age_stroke_00', 'age_stroke_10', 'age_stroke_20', 'age_stroke_30',
       'age_heartattack_00', 'age_heartattack_10', 'age_heartattack_20', 'age_heartattack_30',
       'age_angina_00', 'age_angina_10', 'age_angina_20', 'age_angina_30',
       'age_DVT_00', 'age_DVT_10', 'age_DVT_20', 'age_DVT_30',
       'age_pulmonary_embolism_00', 'age_pulmonary_embolism_10', 'age_pulmonary_embolism_20', 'age_pulmonary_embolism_30',
       'age_diabetes_00', 'age_diabetes_10', 'age_diabetes_20', 'age_diabetes_30',
       'age_glaucoma_00', 'age_glaucoma_10', 'age_glaucoma_20', 'age_glaucoma_30',
       'age_cataract_00', 'age_cataract_10', 'age_cataract_20', 'age_cataract_30',
       'age_other_serious_eye_condition_00', 'age_other_serious_eye_condition_10', 
       'age_other_serious_eye_condition_20', 'age_other_serious_eye_condition_30']


save_files = "~/retina-phenotypes/complementary/N_cases_diseases_per_Ventile/"
read_files = "/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/diseases_cov/22_07_2022_diseases.csv"
num_of_ventiles=10


#### -Read diseases

### The following file contain all the diseases of relevance without filtering by QC
df_diseases = pd.read_csv(read_files, sep=',')
### Remark-> This file can be generated modifying the code in '/image_to_participant/main_create_csv_diseases_covariants.R'

#### -Read Ventiles
# Very ugly way :/
df_v0 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles0.txt", sep=',') 
df_v1 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles1.txt", sep=',') 
df_v2 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles2.txt", sep=',') 
df_v3 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles3.txt", sep=',') 
df_v4 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles4.txt", sep=',') 
df_v5 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles5.txt", sep=',') 
df_v6 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles6.txt", sep=',') 
df_v7 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles7.txt", sep=',') 
df_v8 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles8.txt", sep=',') 
df_v9 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles9.txt", sep=',') 
df_v10 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles10.txt", sep=',') 
df_v11 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles11.txt", sep=',') 
df_v12 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles12.txt", sep=',') 
df_v13 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles13.txt", sep=',') 
df_v14 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles14.txt", sep=',') 
df_v15 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles15.txt", sep=',') 
df_v16 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles16.txt", sep=',') 
df_v17 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles17.txt", sep=',') 
df_v18 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles18.txt", sep=',') 
df_v19 = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_ventiles19.txt", sep=',') 

################################## FUNCTIONS ######################################
#### -Create csv
def images_to_eids(file_name):
    df_after_QC = df_diseases
    #print(file_name)
    df_v = pd.read_csv("/HDD/data/ukbb/fundus/qc/ageCorrected_"+str(file_name)+".txt", sep=',') 

    df_v.columns = ['image']
    df_v.head(3)

    df_v= df_v['image'].str.split('_', expand=True)
    df_v.columns = ['eid', 'eye', 'date', 'instance']
    df_v = df_v[['eid', 'date']]
    # sorting by first name
    df_v.sort_values("eid", inplace = True)
    #print('Before drop duplicates ', df_v.shape[0])

    # dropping ALL duplicate values
    df_v.drop_duplicates(subset ="eid", keep = 'first', inplace = True)
    #print('After drop duplicates ', df_v.shape[0])
    df_v= df_v.astype({"eid": int})
    # Merge
    df_after_QC = pd.merge(df_v, df_after_QC, how='left', on='eid')
    #print('Merge ', df_after_QC.shape[0])
    return df_v, df_after_QC


def plot_diseases_ventiles(lista, df_merge_v0, file_name, N_total):
    print(file_name)
    aux=[]
    aux_ratio=[]
    
    for i in range(len(lista)):
        title=lista[i]
        #plt.figure()
        size_1 = len(df_merge_v0[title])- df_merge_v0[title].isna().sum()
        #print(title + " N=" + str(size_1))
        
        #print(str(size_1))
        #df_merge_v0[title].plot.hist(bins=100, rwidth=0.5)
        #plt.title(title + " N=" + str(size_1))
        #plt.xlabel(title)
        #plt.ylabel('Frequency')
        #plt.show()
        
        data = {'disease': title, 'N_'+file_name: str(size_1)}
        data_ratio = {'disease': title, 'ratio_'+file_name: str(size_1/N_total)}
        aux.append(data)
        aux_ratio.append(data_ratio)
        
    return pd.DataFrame(aux), pd.DataFrame(aux_ratio)


def diseases_ventiles(file_type, save_files):
    for i in range(num_of_ventiles):
        if i==0:
            df_disease_init = pd.read_csv(save_files+'/ventiles0'+str(file_type)+'.csv', sep=',')
            print('Check the len')
            print(len(df_disease_init))
        else:
            df_disease_vi = pd.read_csv(save_files+ 'ventiles'+str(i)+str(file_type)+'.csv', sep=',')
            df_merge = pd.merge(df_disease_vi, df_disease_init, how='inner', on='disease')
            print(len(df_disease_vi), len(df_disease_init), len(df_merge))
            df_disease_init = df_merge
    return df_merge


def bar_plot(df_used, title_name, set_index_column, figsize_a, figsize_b):
    #ax= df_N_cases.plot()
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),fancybox=True, shadow=True, ncol=5)
    ax = df_used.set_index(set_index_column).plot.bar(rot=90, title=title_name, figsize=(figsize_a,figsize_b), fontsize=12)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),fancybox=True, shadow=True, ncol=5)


def compute_N_diseases_all_ventiles():
    for j in range(len(list_file_names)):
        file_name = list_file_names[j]

        df_v0, df_merge = images_to_eids(file_name)
        N_total =df_merge.shape[0]
        df_ventile_disease, df_ventile_disease_ratio = plot_diseases_ventiles(lista, df_merge, file_name, N_total)

        df_ventile_disease.to_csv(save_files+str(file_name)+'_N_cases.csv', index=False)
        df_ventile_disease_ratio.to_csv(save_files+str(file_name)+'_ratios.csv', index=False)


    df_N_cases=diseases_ventiles('_N_cases', save_files)
    df_ratios=diseases_ventiles('_ratios', save_files)


    bar_plot(df_N_cases, 'Number of cases', 'disease', 25, 10)
    bar_plot(df_ratios, 'Ratios cases controls', 'disease', 25,10)
    #df_ratios.plot()


    #### Plots per disease (instances grouped):
    ## Group the different instances by diseases
    df_N_cases_split = df_N_cases.copy()
    df_ratios_split = df_ratios.copy()

    df_N_cases_split[['first', 'second', 'third', 'fourth', 'fifth', 'sixth']]= df_N_cases_split['disease'].str.split('_', expand=True)
    df_ratios_split[['first', 'second', 'third', 'fourth', 'fifth', 'sixth']]= df_N_cases_split['disease'].str.split('_', expand=True)

    df_N_cases_groupby=df_N_cases_split.groupby(by=["second"]).sum()
    df_ratios_groupby=df_ratios_split.groupby(by=["second"]).sum()


    df_N_cases_groupby['second']=df_N_cases_groupby.index
    df_ratios_groupby['second']=df_ratios_groupby.index


    bar_plot(df_N_cases_groupby, ' Number of cases (group)', 'second', 25, 10)
    bar_plot(df_ratios_groupby, 'Ratios cases controls (group)', 'second', 25, 10)
    return df_N_cases_groupby, df_ratios_groupby


### Quantile specific:
def quantile_specific_N_diseases(num):
    df_N_cases_groupby=pd.DataFrame([])
    df_ratios_groupby=pd.DataFrame([])
    
    df_N_cases_groupby, df_ratios_groupby = compute_N_diseases_all_ventiles()

    Quantile='N_ventiles'+str(num)
    Quantile_ratio='ratio_ventiles'+str(num)

    df_quantile2=pd.DataFrame([])
    df_quantile2['diseases']=df_N_cases_groupby['second']
    df_quantile2.reset_index(drop=True, inplace=True)
    df_quantile2[Quantile]=df_N_cases_groupby[Quantile].values

    df_quantile2_ratio=pd.DataFrame([])
    df_quantile2_ratio['diseases']=df_ratios_groupby['second']
    df_quantile2_ratio.reset_index(drop=True, inplace=True)
    df_quantile2_ratio[Quantile_ratio]=df_ratios_groupby[Quantile_ratio].values


    #df_quantile2.set_index('diseases').plot.bar(rot=90, title='Number ventile 2', figsize=(5,5), fontsize=12)
    #df_quantile2_ratio.set_index('diseases').plot.bar(rot=90, title='Ratios ventile 2', figsize=(5,5), fontsize=12)

    bar_plot(df_quantile2, ' Number of cases (group) ventile '+str(num) , 'diseases', 5, 5)
    bar_plot(df_quantile2_ratio, ' Ratio of cases-controls (group) ventile '+str(num) , 'diseases', 5, 5)
    print(df_quantile2_ratio, df_quantile2)

    return df_quantile2_ratio, df_quantile2





