import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from datetime import datetime
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
import scipy.spatial as sp, scipy.cluster.hierarchy as hc



def plot(save_results, csv_name, DATE, num_ventile, size_a, size_b):
    df_pintar = pd.read_csv(save_results + csv_name + '.csv')
    #df_pintar= df_pintar.drop(columns=['Unnamed: 0'])
    df_pintar= df_pintar.set_index('Unnamed: 0')
    #df_pintar.index= df_pintar.columns
    df_pintar=df_pintar.astype(int)
    #print(df_pintar.columns)
    plt.subplots(figsize=(size_a,size_b))
    sns.heatmap(df_pintar, annot=True, cmap="YlGnBu")
    plt.savefig(save_results+'/'+str(DATE)+'_ventile'+str(num_ventile)+
                '_heatmap_genes_intersection.pdf', edgecolor='none')
    

def compute_intersections_csv(p_value_min, filenames, input_dir, save_results, csv_name_all, csv_name_count, csv_name_diagonal, csv_name, csv_genes_name):
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
    df_guardar_final.to_csv(save_results + csv_name_diagonal +'.csv')
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
            l_aux2.append(df_intersection['gen'].to_list())# {'index1': value1, 'index2':value2,...}, ignore_index=True)#.to_list())
            #print('antes', df_intersection['gen'].to_list(), 'despues')
            #print(df_intersection['gen'].values())

        # To save the intersection len
        df = pd.DataFrame({file:l_aux3})
        df_save_shapes = pd.concat([df_save_shapes, df], axis=1)

        # To save the names of the genes in the intersection
        df2 = pd.DataFrame({file:l_aux2})
        df_save_intersections = pd.concat([df_save_intersections, df2], axis=1)
        #print(df_save_intersections[1])

    # To save the intersection len  
    df_save_shapes = df_save_shapes.set_axis(df_save_shapes.columns, axis='index')

    df_save_shapes.to_csv(save_results + csv_name +'.csv')

    # To save the names of the genes in the intersection
    df_save_intersections = df_save_intersections.set_axis(df_save_intersections.columns, axis='index')
    df_save_intersections.to_csv(save_results + csv_genes_name + '.csv')
    
    return df_count, df_guardar_final, df_save_shapes, df_save_intersections


def table_names_numbers(save_results, csv_name, csv_genes_name, csv_both):
    ## Read numbers and names csv
    df_numbers = pd.read_csv(save_results + csv_name + '.csv')
    df_numbers= df_numbers.set_index('Unnamed: 0')

    df_names = pd.read_csv(save_results + csv_genes_name + '.csv')
    df_names= df_names.set_index('Unnamed: 0')
    
    np_upper = np.triu(df_numbers)
    df_upper=pd.DataFrame(np_upper)

    np_lower = np.tril(df_names, k=-1) #k=-1 to not take the diagonal
    df_lower=pd.DataFrame(np_lower)
    
    i_upper = np.triu_indices(16)
    np_final = np_lower
    np_final[i_upper] = np_upper[i_upper]
    df_c_magic = pd.DataFrame(np_lower)
    df_c_magic.columns = df_names.columns
    df_c_magic.index = df_names.columns
    df_c_magic.to_csv(save_results + csv_both +'.csv')
    
    return df_c_magic

def cluster_with_distances(save_results, csv_name, DATE, num_ventile):
    df_pintar = pd.read_csv(save_results + csv_name + '.csv')
    #df_pintar= df_pintar.drop(columns=['Unnamed: 0'])
    df_pintar= df_pintar.set_index('Unnamed: 0')
    #df_pintar.index= df_pintar.columns
    df_pintar=df_pintar.astype(int)
    #print(df_pintar.columns)

    df_aux = df_pintar
    l=[]
    for i in range(len(df_aux)):
        lista=df_aux.iloc[i]/df_aux.iloc[i][i]
        l.append(lista)
    df_ratios= pd.DataFrame(l)
    ### You need to: 
    ## - delete the nans, 
    ## - diagonal to 0, and
    ## - abs(linkage)

    ##### IT IS NOT SIMETRIC!
    np_upper = np.triu(df_ratios)
    df_upper=pd.DataFrame(np_upper)
    df_upper = df_upper.T+ df_upper ##diagonal values=2

    np_lower = np.tril(df_ratios)
    df_lower=pd.DataFrame(np_lower)
    df_lower = df_lower.T+ df_lower
    
    #set index and column names back (since the cluster is not done yet it is okay)

    #len(df_upper)
    df_upper.columns = df_ratios.columns
    df_upper.index= df_ratios.index

    df_lower.columns = df_ratios.columns
    df_lower.index= df_ratios.index

    df_upper_aux=df_upper.copy()
    df_lower_aux=df_lower.copy()
    
    #set diagonal values = 1

    df_upper.values[[np.arange(df_upper.shape[0])]*2] = 1
    df_lower.values[[np.arange(df_lower.shape[0])]*2] = 1
    
    # Check the values are correct: 
    Check_correct=False
    if Check_correct=='True':
        df_ratios - df_upper 
        #Correct Outcome: First triang half and diag = 0
        df_ratios - df_lower 
        #Correct Outcome: Second triang half and diag = 0
        df_upper_aux - df_upper #Correct Outcome: All 0 except diag = 1
        df_lower_aux - df_lower #Correct Outcome: All 0 except diag = 1
    
    ########## df_upper
    df_upper_ = 1 - abs(df_upper) 
    ## diagonal same value:
    df_upper_.values[[np.arange(df_upper_.shape[0])]*2] = 0

    linkage = hc.linkage(sp.distance.squareform(df_upper_), method='average')

    h = sns.clustermap(df_upper_, row_linkage=linkage, col_linkage=linkage, cmap=cm.get_cmap('viridis_r'), figsize=(10,10), vmin=0, vmax=1)
    h.ax_row_dendrogram.set_visible(False)
    #h.ax_heatmap.xaxis.tick_top()
    h.ax_heatmap.tick_params(axis='x', rotation=90)
    plt.savefig(save_results + str(DATE) + '_ventile'+ str(num_ventile) + '_upper_genes_cluster.jpg')
    
    ########## df_lower
    df_lower_ = 1 - abs(df_lower) 
    ## diagonal same value:
    df_upper_.values[[np.arange(df_upper_.shape[0])]*2] = 0

    linkage = hc.linkage(sp.distance.squareform(df_lower_), method='average')
    
    h = sns.clustermap(df_lower_, row_linkage=linkage, col_linkage=linkage, cmap=cm.get_cmap('viridis_r'), figsize=(10,10), vmin=0, vmax=1)
    h.ax_row_dendrogram.set_visible(False)
    #h.ax_heatmap.xaxis.tick_top()
    h.ax_heatmap.tick_params(axis='x', rotation=90)
    plt.savefig(save_results  + str(DATE) + '_ventile'+ str(num_ventile) + '_lower_genes_cluster.jpg')


def difference_diagonal(save_results, DATE):
    df_v2=pd.read_csv(save_results+ 'intersections_diagonal_main_v2.csv')
    df_v5=pd.read_csv(save_results+ 'intersections_diagonal_main_v5.csv')
    if len(df_v2) != len(df_v5):   
        print('Error, different lens in intersections_diagonal_main')
    #df_v5= df_v5[['file_col', 'gen_y']]
    #df_v5.set_index('file_col').plot.bar(rot=90, title='ventile5', figsize=(10,5), fontsize=12)
    df_v5.rename(columns = {'gen_y':'V5_gen_y'}, inplace = True)
    df_v2.rename(columns = {'gen_y':'V2_gen_y'}, inplace = True)
    df_aux= df_v2.merge(df_v5, on='file_col', how='inner')
    df_aux['dif_v5_minus_v2']=df_aux['V5_gen_y']-df_aux['V2_gen_y']
    #df_v2= df_v2[['file_col', 'gen_y']]
    df_aux.set_index('file_col').plot.bar(rot=90, title='v5 -v2', figsize=(10,5), fontsize=12)
    plt.savefig(save_results+'/'+str(DATE)+'_difference_ventiles_genes_intersection.pdf', edgecolor='none')
    sys.exit()

