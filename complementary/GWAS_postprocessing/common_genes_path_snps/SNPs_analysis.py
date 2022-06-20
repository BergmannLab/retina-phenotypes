import pandas as pd
import functions_genes_pathways as f
import math
import numpy as np
import venn
from matplotlib import pyplot as plt

input_dir = '/Users/sortinve/Desktop/PascalOutput/SNPs/'

#######################################################################################################################
# Part 1: Select the parameters of interest
dir_median_diameter_genes = input_dir + 'GWAS_results_pruned_median_diameter_R2_0.1_GBR_all.csv'
dir_median_diameter_artery_genes = input_dir + 'GWAS_results_pruned_R2_0.1_GBR_artery.csv'
dir_median_diameter_vein_genes = input_dir + 'GWAS_results_pruned_R2_0.1_GBR_vein.csv'

dir_median_tortuosity_genes = input_dir +'GWAS_results_pruned_median_tortuosity_R2_0.1_GBR_all.csv'
dir_median_tortuosity_artery_genes = input_dir +'GWAS_results_pruned_R2_0.1_GBRmedian_tortuosity_artery.csv'
dir_median_tortuosity_vein_genes = input_dir +'GWAS_results_pruned_R2_0.1_GBRmedian_tortuosity_vein.csv'

dir_ratio_genes = input_dir +'GWAS_results_pruned_AV_median_diameter_ratio_R2_0.1_GBR_all.csv'
dir_bifurcation_genes = input_dir +'GWAS_results_pruned_R2_0.1_bifurcations__topHits.csv_GBR_all.csv'


#######################################################################################################################
# Part 2: Read the files

top_hits_median_diameter_genes = pd.read_csv(dir_median_diameter_genes, delimiter=',')
print("Top hits from median diameter: ", top_hits_median_diameter_genes)
top_hits_median_diameter_artery_genes = pd.read_csv(dir_median_diameter_artery_genes, delimiter=',')
print("Top hits from median diameter artery: ", top_hits_median_diameter_artery_genes)
top_hits_median_diameter_vein_genes = pd.read_csv(dir_median_diameter_vein_genes, delimiter=',')
print("Top hits from median diameter vein: ", top_hits_median_diameter_vein_genes)
top_hits_median_tortuosity_genes = pd.read_csv(dir_median_tortuosity_genes, delimiter=',')
print("Top hits from median tortuosity: ", top_hits_median_tortuosity_genes)
top_hits_median_tortuosity_artery_genes = pd.read_csv(dir_median_tortuosity_artery_genes, delimiter=',')
print("Top hits from median tortuosity artery: ", top_hits_median_tortuosity_artery_genes)
top_hits_median_tortuosity_vein_genes = pd.read_csv(dir_median_tortuosity_vein_genes, delimiter=',')
print("Top hits from median tortuosity vein: ", top_hits_median_tortuosity_vein_genes)
top_hits_ratio_genes = pd.read_csv(dir_ratio_genes, delimiter=',')
print("Top hits from ratio: ", top_hits_ratio_genes)
top_hits_bifurcation_genes = pd.read_csv(dir_bifurcation_genes, delimiter=',')
print("Top hits from bifurcations: ", top_hits_bifurcation_genes)

list_phenotypes = [top_hits_median_diameter_genes,
                   top_hits_median_diameter_artery_genes,  top_hits_median_diameter_vein_genes,
                   top_hits_median_tortuosity_genes,
                   top_hits_median_tortuosity_artery_genes, top_hits_median_tortuosity_vein_genes,
                   top_hits_ratio_genes, top_hits_bifurcation_genes]
names = ['diameter', 'diameter_artery', 'diameter_vein', 'tortuosity', 'tortuosity_artery', 'tortuosity_vein',
                   'ratio', 'bifurcations']


for i in range(len(list_phenotypes)):
    for j in range(len(list_phenotypes)):
        df_merge = pd.merge(list_phenotypes[i], list_phenotypes[j], how='inner', on=['rsid'])
        print("intersection between ", names[i], " and ",  names[j], " : ",  len(df_merge), " : ", df_merge['rsid'] )


print(1)
