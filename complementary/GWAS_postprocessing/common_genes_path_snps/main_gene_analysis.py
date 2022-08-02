# 17-06-2022 Sofia
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns


input_dir = '/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/gwas/2022_07_08_ventile5__FINAL/'
save_results = '/SSD/home/sofia/retina-phenotypes/complementary/GWAS_postprocessing/common_genes_path_snps/'
csv_name = 'intersections'
csv_name_all = 'intersections_all'
csv_genes_name = 'intersections_genes_name'
csv_name_diagonal = 'intersections_diagonal'
csv_name_count = 'genes_count'
p_value_min = 5.7  # -math.log10(0.05/len(x))

filenames= ["AVScore_all", "AVScore_longestFifth_all", "tau1_longestFifth_all", "tau1_longestFifth_artery", "tau1_longestFifth_vein", "tau1_all", "tau1_artery", "tau1_vein", "tau2_longestFifth_all", "tau2_longestFifth_artery", "tau2_longestFifth_vein", "tau2_all", "tau2_artery", "tau2_vein", "tau4_longestFifth_all", "tau4_longestFifth_artery", "tau4_longestFifth_vein", "tau4_all", "tau4_artery", "tau4_vein", "D_A_std_std", "D_V_std_std", "D_median_CVMe", "N_median_main_arteries", "N_median_main_veins", "arcLength_longestFifth_artery", "arcLength_longestFifth_vein", "arcLength_artery", "arcLength_vein", "bifurcations", "VD_orig_all", "VD_orig_artery", "VD_orig_vein", "slope", "slope_artery", "slope_vein", "mean_angle_taa", "mean_angle_tva", "medianCenter1_longestFifth_artery", "medianCenter1_longestFifth_vein", "medianCenter1_artery", "medianCenter1_vein", "medianCenter2_longestFifth_artery", "medianCenter2_longestFifth_vein", "medianCenter2_artery", "medianCenter2_vein", "medianDiameter_longestFifth_artery", "medianDiameter_longestFifth_vein", "eq_CRAE", "eq_CRVE", "median_CRAE", "median_CRVE", "ratio_CRAE_CRVE", "ratio_median_CRAE_CRVE", "ratio_AV_medianDiameter", "ratio_medianDiameter_longest", "ratio_DF_longest", "ratio_tau2_longest", "ratio_tau4_longest"]


def plot():
    df_pintar = pd.read_csv(save_results + csv_name + '.csv')
    df_pintar= df_pintar.drop(columns=['Unnamed: 0'])
    df_pintar.index= df_pintar.columns
    df_pintar=df_pintar.astype(int)
    #print(df_pintar.columns)

    plt.subplots(figsize=(40,35))
    sns.heatmap(df_pintar, annot=True, cmap="YlGnBu")
    plt.savefig(save_results+'/Heatmap_genes_intersection.pdf', edgecolor='none')



def compute_intersections_csv():
    l_aux = []

    for file in filenames:
        #print('file', file)
        # Read csvs
        df = pd.read_csv(input_dir+file+'__gene_scores', delimiter='\t', names =['gen', 'p']) #, index_col=None, header=0)
        df['file_col']=file
        l_aux.append(df)

    # Concat all the csvs
    df_concat = pd.concat(l_aux)

    # From p to -log10(p)
    df_concat['-log10(p)'] = -np.log10(df_concat['p'])
    y = df_concat[df_concat['-log10(p)'] >= p_value_min]
    df_significant = y.sort_values('-log10(p)', ascending=False)
    #print(df_significant.head(5))
    df_significant.to_csv(save_results + csv_name_all + '.csv')
    df_count=df_significant['gen'].value_counts().to_frame()
    df_count['ratio_N_pheno']=df_significant['gen'].value_counts().to_frame()/len(filenames)
    df_count.to_csv(save_results + csv_name_count + '.csv')

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