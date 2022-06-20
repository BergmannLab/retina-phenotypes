import pandas as pd
import genes_sets_functions as f
import numpy as np
import matplotlib.pyplot as plt
#import venn

# [0]: pathway names
# [3]: pathway P values

input_dir = '/HDD/data/UKBiob/GWAS/2022_06_08_all_phenotypes_LWNet_Decile3/PascalX/'

#List_genes_files=[geneScores_DF_all, geneScores_D_median_std, geneScores_eq_CRVE, geneScores_ratio_AV_DF,
#                  geneScores_DF_artery, geneScores_D_std_std, geneScores_mean_angle_taa, 
#                  geneScores_ratio_AV_medianDiameter, geneScores_DF_vein, geneScores_VD_orig_all, geneScores_mean_angle_tva, 
#                  geneScores_ratio_CRAE_CRVE, geneScores_D_A_mean_std, geneScores_VD_orig_artery, geneScores_mean_intensity,
#                  geneScores_ratio_VA_DF, geneScores_D_A_median_std, geneScores_VD_orig_vein, geneScores_medianDiameter_all,
#                  geneScores_ratio_VA_medianDiameter, geneScores_D_A_std_std, geneScores_VD_small_all, 
#                  geneScores_medianDiameter_artery,  geneScores_ratio_median_CRAE_CRVE, geneScores_D_V_mean_std,   
#                  geneScores_VD_small_artery, geneScores_medianDiameter_vein, geneScores_slope, geneScores_D_V_median_std,
#                  geneScores_VD_small_vein, geneScores_median_CRAE, geneScores_slope_artery, geneScores_D_V_std_std,
#                  geneScores_bifurcations, geneScores_median_CRVE, geneScores_slope_vein, geneScores_D_mean_std,
#                  geneScores_eq_CRAE, geneScores_median_intensity, geneScores_std_intensity]


#######################################################################################################################
# Part 1: Select the parameters of interest
dir_median_diameter_genes = input_dir + 'geneScores_medianDiameter_all'
dir_median_diameter_artery_genes = input_dir + 'geneScores_medianDiameter_artery'
dir_median_diameter_vein_genes = input_dir + 'geneScores_medianDiameter_vein'

dir_median_tortuosity_genes = input_dir +'geneScores_DF_all'
dir_median_tortuosity_artery_genes = input_dir +'geneScores_DF_artery'
dir_median_tortuosity_vein_genes = input_dir +'geneScores_DF_vein'

dir_ratio_genes = input_dir +'geneScores_ratio_AV_medianDiameter'
dir_bifurcation_genes = input_dir +'geneScores_bifurcations'


#######################################################################################################################
# Part 2: Read the files
p_value_min = 5.7  # -math.log10(0.05/len(x))

top_hits_median_diameter_genes = f.create_df(dir_median_diameter_genes, p_value_min)
print("Top hits from median diameter: ", top_hits_median_diameter_genes)
top_hits_median_diameter_artery_genes = f.create_df(dir_median_diameter_artery_genes, p_value_min)
print("Top hits from median diameter artery: ", top_hits_median_diameter_artery_genes)
top_hits_median_diameter_vein_genes = f.create_df(dir_median_diameter_vein_genes, p_value_min)
print("Top hits from median diameter vein: ", top_hits_median_diameter_vein_genes)

top_hits_median_tortuosity_genes = f.create_df(dir_median_tortuosity_genes, p_value_min)
print("Top hits from median tortuosity: ", top_hits_median_tortuosity_genes)
top_hits_median_tortuosity_artery_genes = f.create_df(dir_median_tortuosity_artery_genes, p_value_min)
print("Top hits from median tortuosity artery: ", top_hits_median_tortuosity_artery_genes)
top_hits_median_tortuosity_vein_genes = f.create_df(dir_median_tortuosity_vein_genes, p_value_min)
print("Top hits from median tortuosity vein: ", top_hits_median_tortuosity_vein_genes)

top_hits_ratio_genes = f.create_df(dir_ratio_genes, p_value_min)
print("Top hits from ratio: ", top_hits_ratio_genes)

top_hits_bifurcation_genes = f.create_df(dir_bifurcation_genes, p_value_min)
print("Top hits from bifurcations: ", top_hits_bifurcation_genes)


dir_ratio_DF_genes = input_dir +'geneScores_ratio_AV_DF'
dir_tva = input_dir +'geneScores_mean_angle_tva'
dir_taa = input_dir +'geneScores_mean_angle_taa'
dir_CRVE = input_dir +'geneScores_eq_CRVE'
dir_CRAE = input_dir +'geneScores_eq_CRAE'
dir_VD = input_dir +'geneScores_VD_orig_all'
dir_slope = input_dir +'geneScores_slope'
dir_D_V = input_dir +'geneScores_D_median_std'
dir_Intensity = input_dir +'geneScores_std_intensity'

list_phenotypes = [top_hits_median_diameter_genes,
                   top_hits_median_diameter_artery_genes,  top_hits_median_diameter_vein_genes,
                   top_hits_median_tortuosity_genes,
                   top_hits_median_tortuosity_artery_genes, top_hits_median_tortuosity_vein_genes,
                   top_hits_ratio_genes, top_hits_bifurcation_genes]
names = ['diameter', 'diameter_artery', 'diameter_vein', 'tortuosity', 'tortuosity_artery', 'tortuosity_vein',
                   'ratio', 'bifurcations']
dfs = []
for i in range(len(list_phenotypes)):
    for j in range(len(list_phenotypes)):
        df_merge = pd.merge(list_phenotypes[i], list_phenotypes[j], how='inner', on=['gen'])
        print("intersection between ", names[i], " and ",  names[j], " : ",  len(df_merge), " : ", df_merge['gen'])
        dfs.append(len(df_merge))


                # Plot table

fig, axs = plt.subplots(2, 1)
clust_data = np.random.random((10, 3))
collabel = ("col 1", "col 2", "col 3")
axs[0].axis('tight')
axs[0].axis('off')
the_table = axs[0].table(cellText=clust_data, colLabels=collabel, loc='center')

# axs[1].plot(clust_data[:, 0], clust_data[:, 1])
# plt.show()
# fig, axs = plt.subplots(2, 1)
# clust_data = np.random.random((10, 3))
# collabel = ("col 1", "col 2", "col 3")
# axs[0].axis('tight')
# axs[0].axis('off')
# the_table = axs[0].table(cellText=clust_data, colLabels=collabel, loc='center')
#
# axs[1].plot(clust_data[:, 0], clust_data[:, 1])
plt.show()


print(1)




        