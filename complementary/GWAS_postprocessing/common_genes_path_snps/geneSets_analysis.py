import pandas as pd
import genes_sets_functions as f
import math
import numpy as np
import venn
from matplotlib import pyplot as plt

# [0]: pathway names
# [3]: pathway P values

input_dir = '/Users/sortinve/Desktop/PascalOutput/PascalX/'

#######################################################################################################################
# Part 1: Select the parameters of interest
dir_median_diameter_path = input_dir + 'pathwayScores_median_diameter.txt'
dir_median_diameter_artery_path = input_dir + 'pathwayScores_median_diameter_artery.txt'
dir_median_diameter_vein_path = input_dir + 'pathwayScores_median_diameter_vein.txt'

dir_median_tortuosity_path = input_dir +'pathwayScores_median_tortuosity.txt'
dir_median_tortuosity_artery_path = input_dir +'pathwayScores_median_tortuosity_artery.txt'
dir_median_tortuosity_vein_path = input_dir +'pathwayScores_median_tortuosity_vein.txt'

dir_ratio_path = input_dir +'pathwayScores_AV_median_diameter_ratio.txt'
dir_bifurcation_path = input_dir +'pathwayScores_tau3.txt'


#######################################################################################################################
# Part 2: Read the files
p_value_min = 5.8

top_hits_median_diameter_path = f.create_df_path(dir_median_diameter_path, p_value_min)
print("Top hits from median diameter: ", top_hits_median_diameter_path)
top_hits_median_diameter_artery_path = f.create_df_path(dir_median_diameter_artery_path, p_value_min)
print("Top hits from median diameter artery: ", top_hits_median_diameter_artery_path)
top_hits_median_diameter_vein_path = f.create_df_path(dir_median_diameter_vein_path, p_value_min)
print("Top hits from median diameter vein: ", top_hits_median_diameter_vein_path)

top_hits_median_tortuosity_path = f.create_df_path(dir_median_tortuosity_path, p_value_min)
print("Top hits from median tortuosity: ", top_hits_median_tortuosity_path)
top_hits_median_tortuosity_artery_path = f.create_df_path(dir_median_tortuosity_artery_path, p_value_min)
print("Top hits from median tortuosity artery: ", top_hits_median_tortuosity_artery_path)
top_hits_median_tortuosity_vein_path = f.create_df_path(dir_median_tortuosity_vein_path, p_value_min)
print("Top hits from median tortuosity vein: ", top_hits_median_tortuosity_vein_path)

top_hits_ratio_path = f.create_df_path(dir_ratio_path, p_value_min)
print("Top hits from ratio: ", top_hits_ratio_path)

top_hits_bifurcation_path = f.create_df_path(dir_bifurcation_path, p_value_min)
print("Top hits from bifurcations: ", top_hits_bifurcation_path)

list_phenotypes = [top_hits_median_diameter_path,
                   top_hits_median_diameter_artery_path,  top_hits_median_diameter_vein_path,
                   top_hits_median_tortuosity_path,
                   top_hits_median_tortuosity_artery_path, top_hits_median_tortuosity_vein_path,
                   top_hits_ratio_path, top_hits_bifurcation_path]
names = ['diameter', 'diameter_artery', 'diameter_vein', 'tortuosity', 'tortuosity_artery', 'tortuosity_vein',
                   'ratio', 'bifurcations']

for i in range(len(list_phenotypes)):
    for j in range(len(list_phenotypes)):
        df_merge = pd.merge(list_phenotypes[i], list_phenotypes[j], how='inner', on=['pathway'])
        print("intersection between ", names[i], " and ",  names[j], " : ", len(df_merge), " : ", df_merge['pathway'])


print(1)
