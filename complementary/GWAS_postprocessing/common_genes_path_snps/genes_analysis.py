import pandas as pd
import genes_sets_functions as f
import numpy as np
import matplotlib.pyplot as plt
import venn

# [0]: pathway names
# [3]: pathway P values

input_dir = '/Users/sortinve/Desktop/PascalOutput/PascalX/'

#######################################################################################################################
# Part 1: Select the parameters of interest
dir_median_diameter_genes = input_dir + 'geneScores_median_diameter'
dir_median_diameter_artery_genes = input_dir + 'geneScores_median_diameter_artery'
dir_median_diameter_vein_genes = input_dir + 'geneScores_median_diameter_vein'

dir_median_tortuosity_genes = input_dir +'geneScores_median_tortuosity'
dir_median_tortuosity_artery_genes = input_dir +'geneScores_median_tortuosity_artery'
dir_median_tortuosity_vein_genes = input_dir +'geneScores_median_tortuosity_vein'

dir_ratio_genes = input_dir +'geneScores_AV_median_diameter_ratio'
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
