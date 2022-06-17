# 17-06-2022 Sofia
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns


input_dir = '/HDD/data/UKBiob/GWAS/2022_06_08_all_phenotypes_LWNet_Decile3/PascalX/'
save_results = '/SSD/home/sofia/retina-phenotypes/complementary/GWAS_postprocessing/common_genes_path_snps/'
csv_name = 'intersections'
csv_genes_name = 'intersections_genes_name'
csv_name_diagonal = 'intersections_diagonal'
p_value_min = 5.7  # -math.log10(0.05/len(x))

filenames= ['geneScores_bifurcations','geneScores_mean_angle_tva', 'geneScores_mean_angle_taa', 'geneScores_eq_CRVE', 'geneScores_eq_CRAE', 'geneScores_ratio_CRAE_CRVE', 'geneScores_medianDiameter_all', 'geneScores_medianDiameter_artery', 'geneScores_medianDiameter_vein', 'geneScores_ratio_AV_medianDiameter', 'geneScores_D_median_std', 'geneScores_D_std_std', 'geneScores_D_mean_std', 'geneScores_std_intensity', 
'geneScores_VD_orig_all', 'geneScores_VD_orig_artery', 'geneScores_VD_orig_vein', 
'geneScores_slope', 'geneScores_slope_artery', 'geneScores_slope_vein']

filenames_all= ['geneScores_bifurcations','geneScores_mean_angle_tva', 'geneScores_mean_angle_taa', 'geneScores_eq_CRVE', 'geneScores_eq_CRAE', 'geneScores_ratio_CRAE_CRVE', 'geneScores_median_CRVE', 'geneScores_median_CRAE', 'geneScores_ratio_median_CRAE_CRVE', 'geneScores_medianDiameter_all', 'geneScores_medianDiameter_artery', 'geneScores_medianDiameter_vein', 'geneScores_ratio_AV_medianDiameter', 'geneScores_ratio_VA_medianDiameter', 'geneScores_D_median_std', 'geneScores_D_std_std', 'geneScores_D_mean_std', 'geneScores_std_intensity', 
'geneScores_VD_orig_all', 'geneScores_VD_orig_artery', 'geneScores_VD_orig_vein', 
'geneScores_slope', 'geneScores_slope_artery', 'geneScores_slope_vein',
'geneScores_DF_all','geneScores_DF_artery', 'geneScores_DF_vein', 'geneScores_ratio_AV_DF', 'geneScores_ratio_VA_DF']

def plot():
    df_pintar = pd.read_csv(save_results + csv_name + '.csv')
    df_pintar= df_pintar.drop(columns=['Unnamed: 0'])
    df_pintar.index= df_pintar.columns
    df_pintar=df_pintar.astype(int)
    #print(df_pintar.columns)

    plt.subplots(figsize=(20,15))
    sns.heatmap(df_pintar, annot=True)
    plt.savefig(save_results+'/Heatmap.pdf', edgecolor='none')



def compute_intersections_csv():
    l_aux = []

    for file in filenames:
        print('file', file)
        # Read csvs
        df = pd.read_csv(input_dir+file, delimiter='\t', names =['gen', 'p']) #, index_col=None, header=0)
        df['file_col']=file
        l_aux.append(df)

    # Concat all the csvs
    df_concat = pd.concat(l_aux)

    # From p to -log10(p)
    df_concat['-log10(p)'] = -np.log10(df_concat['p'])
    y = df_concat[df_concat['-log10(p)'] >= p_value_min]
    df_significant = y.sort_values('-log10(p)', ascending=False)
    #print(df_significant.head(5))

    ## Save the number of significant genes per phenotype
    df_guardar = pd.DataFrame(df_significant.groupby(by=['file_col'])['gen'].apply(list))
    df_guardar2 = pd.DataFrame(df_significant.groupby(by=['file_col'])['gen'].count())
    df_guardar_final=df_guardar.merge(df_guardar2, how='inner', on='file_col')
    df_guardar_final.to_csv(save_results + csv_name_diagonal + '.csv')
    df_save_shapes=pd.DataFrame([])
    df_save_intersections=pd.DataFrame([])


    i=0
    for file in filenames:
        i=i+1
        genes=df_significant[df_significant['file_col']==file]['gen']#.to_list()
        genes=genes.to_list()
        l_aux2 = []
        l_aux3 = []

        for j in range(len(filenames)):#-i):#-file: #Error
            other_file=filenames[j]
            df_intersection=df_significant[(df_significant['file_col']==other_file)&(df_significant['gen'].isin(genes))]
            # To save the intersection len
            save_shapes=df_intersection.shape[0]
            l_aux3.append(save_shapes)

            # To save the names of the genes in the intersection
            l_aux2.append(df_intersection['gen'])#.to_list())

        # To save the intersection len
        df = pd.DataFrame({file:l_aux3})
        df_save_shapes = pd.concat([df_save_shapes, df], axis=1)

        # To save the names of the genes in the intersection
        df2 = pd.DataFrame({file:l_aux2})
        df_save_intersections = pd.concat([df_save_intersections, df2], axis=1)

    # To save the intersection len  
    df_save_shapes = df_save_shapes.set_axis(df_save_shapes.columns, axis='index')

    df_save_shapes.to_csv(save_results + csv_name + '.csv')

    # To save the names of the genes in the intersection
    df_save_intersections = df_save_intersections.set_axis(df_save_intersections.columns, axis='index')
    df_save_intersections.to_csv(save_results + csv_genes_name + '.csv')



compute_intersections_csv()
plot()