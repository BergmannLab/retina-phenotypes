# Created 17-06-2022
# Last modification: 16-08-2022

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import function_genes as f_g
import sys
from datetime import datetime

DATE = datetime.now().strftime("%Y-%m-%d")

num_ventile = 'Zekavat' #2
input_dir = sys.argv[1] #'/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/gwas/2022_08_03_ventile'+str(num_ventile)+'/'
save_results = sys.argv[2] # '/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/genes/'
phenotypes_type=sys.argv[3] #'main'


csv_name = 'intersections_'+phenotypes_type + '_v'+str(num_ventile) 
csv_name_all = 'intersections_all_'+phenotypes_type + '_v'+str(num_ventile) 
csv_genes_name = 'intersections_genes_name_'+phenotypes_type+ '_v'+ str(num_ventile) 
csv_name_diagonal = 'intersections_diagonal_'+phenotypes_type + '_v'+str(num_ventile) 
csv_name_count = 'genes_count_'+phenotypes_type + '_v'+str(num_ventile) 

csv_both = 'both_'+phenotypes_type + '_v'+str(num_ventile)

p_value_min = 5.7  # -math.log10(0.05/len(x))

if phenotypes_type=='suplementary':
    filenames= list(sys.argv[6].split(","))
    filenames_new= list(sys.argv[7].split(","))
    print(filenames)
    print(filenames_new)
    size_a=40
    size_b=40
elif phenotypes_type=='main':
    filenames= list(sys.argv[4].split(","))
    filenames_new= list(sys.argv[5].split(","))
    size_a=15
    size_b=15


df_count, df_guardar_final, df_save_shapes, df_save_intersections = f_g.compute_intersections_csv(p_value_min, filenames, input_dir, save_results, csv_name_all, csv_name_count, csv_name_diagonal, csv_name, csv_genes_name)

f_g.plot(save_results, csv_name, DATE, num_ventile, size_a, size_b, phenotypes_type, filenames, filenames_new)

df_both = f_g.table_names_numbers(save_results, csv_name, csv_genes_name, csv_both)

#f_g.cluster_with_distances(save_results, csv_name, DATE, num_ventile)

#f_g.difference_diagonal(save_results, DATE)
