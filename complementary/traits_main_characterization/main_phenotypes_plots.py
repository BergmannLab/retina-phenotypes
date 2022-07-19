import pandas as pd
import seaborn as sns
import os
import glob
import numpy as np
import matplotlib as plt
import functions_plot as f_plot

#all_phenotypes = ['bifurcations', 'median_CRAE', 'eq_CRAE', 'median_CRVE', 'eq_CRVE','ratio_median_CRAE_CRVE', 'ratio_CRAE_CRVE', 'medianDiameter_all', 'medianDiameter_artery', 'medianDiameter_vein', 'DF_all', 'DF_artery', 'DF_vein', 'ratio_AV_medianDiameter', 'ratio_VA_medianDiameter','ratio_AV_DF', 'ratio_VA_DF', 'mean_angle_taa', 'mean_angle_tva', 'D_median_std', 'D_mean_std', 'D_std_std', 'D_A_median_std', 'D_A_mean_std', 'D_A_std_std', 'D_V_median_std', 'D_V_mean_std', 'D_V_std_std', 'VD_orig_all', 'VD_orig_artery', 'VD_orig_vein', 'VD_small_all', 'VD_small_artery', 'VD_small_vein', 'slope', 'slope_artery', 'slope_vein', 'std_intensity', 'mean_intensity', 'median_intensity']

plot_histograms=False
plot_violin=True


phenofiles_dir='/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/participant_phenotype/'
phenofile_used_for_dist_plots="2022_07_08_ventile2_raw.csv"
save_dir='/SSD/home/sofia/retina-phenotypes/complementary/traits_main_characterization/results/'
main_phenotypes_seaborn = ["bifurcations", "VD_orig_all", "slope", "mean_angle_taa", "mean_angle_tva", "eq_CRAE", "eq_CRVE", "medianDiameter_all", "DF_all"]


#### Read file:
df_data_completo = pd.read_csv(phenofiles_dir + phenofile_used_for_dist_plots, sep=',')
print(df_data_completo.columns)
print("Sample size: ", len(df_data_completo))
# Reemplazar los -999 por np.NaN
# Comparar si borrando los NaNs disminuye mucho la muestra
#print("Sample size before delete the -999: ", len(df_data_completo))
#df_data_completo = df_data_completo[df_data_completo != -999]
#df_data_completo = df_data_completo.dropna()
#print("Sample size after delete the -999: ", len(df_data_completo))


################### Scatter plots + histograms in the diagonal : #############################
## Relevant documentation to create new versions: 
# https://seaborn.pydata.org/generated/seaborn.pairplot.html, https://stackoverflow.com/questions/71217352/seaborn-pairgrid-pairplot-two-data-set-with-different-transparency

### Multiple scatter plots with lines and histogram in the diagonal
#f_plot.multiple_scatter_plots(df_data_completo, main_phenotypes, save_dir)

### Multiple scatter plots with corr values in the first half and histogram in the diagonal
#f_plot.multiple_scatter_plots_2(df_data_completo, main_phenotypes, save_dir)

### Scatter plots with corr values in the first half and histogram in the diagonal
#f_plot.multiple_scatter_plots_3(df_data_completo, main_phenotypes_seaborn, save_dir)


################### Violin plots: #############################
# Relevant documents to create new versions: https://www.oreilly.com/library/view/python-data-science/9781491912126/ch04.html
if plot_violin==True:
    ###
    list_phenotypes = ["medianDiameter_all", "medianDiameter_artery", "medianDiameter_vein"]
    my_pal= {"medianDiameter_all": "gray", "medianDiameter_artery": "lightcoral", "medianDiameter_vein": 
             "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)

    list_phenotypes = ["mean_angle_taa",  "mean_angle_tva"]
    my_pal= {"mean_angle_taa": "lightcoral", "mean_angle_tva": "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    
    list_phenotypes = ["eq_CRAE", "eq_CRVE"]
    my_pal= {"eq_CRAE": "lightcoral", "eq_CRVE": "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    
    list_phenotypes = ["DF_all",  "DF_artery", "DF_vein"]
    my_pal= {"DF_all": "gray", "DF_artery": "lightcoral", "DF_vein": "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)

    list_phenotypes = ["ratio_AV_medianDiameter", "ratio_CRAE_CRVE",  "ratio_AV_DF"]
    my_pal= {"ratio_AV_medianDiameter": "g", "ratio_CRAE_CRVE": "b", "ratio_AV_DF": "m"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    
    list_phenotypes = ["VD_orig_all", "VD_orig_artery", "VD_orig_vein"]
    my_pal= {"VD_orig_all": "gray", "VD_orig_artery": "lightcoral", "VD_orig_vein": "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    
    list_phenotypes = ['slope', 'slope_artery', 'slope_vein']
    my_pal= {"slope": "gray", "slope_artery": "lightcoral", "slope_vein": "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    
    list_phenotypes = ['bifurcations', 'std_intensity', 'mean_intensity']
    my_pal="Set3"
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)

    list_phenotypes = ['D_median_CVMe', 'D_std_std', 'D_A_std_std', 'D_V_std_std']
    my_pal="Set3"
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    
    list_phenotypes = ['DF_all', 'tau2_all', 'tau3_all', 'tau4_all']
    my_pal="Set3"
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    
    list_phenotypes = ['ratio_medianDiameter_longest', 'ratio_DF_longest', 'ratio_tau2_longest', 
                       'ratio_tau3_longest', 'ratio_tau4_longest']
    my_pal="Set3"
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)

    list_phenotypes = ["N_median_main_arteries", "N_median_main_veins"]
    my_pal= {"N_median_main_arteries": "lightcoral", "N_median_main_veins": "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)

################### Histograms: #############################
# Relevant documents to create new versions: https://www.oreilly.com/library/view/python-data-science/9781491912126/ch04.html
elif plot_histograms==True:
    ### One multiple histograms plot
    list_phenotypes = ["bifurcations"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    ### Two multiple histograms plot
    list_phenotypes = ["mean_angle_taa",  "mean_angle_tva"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    ### Three multiple histograms plot
    list_phenotypes = ["medianDiameter_all",  "medianDiameter_artery", "medianDiameter_vein"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    list_phenotypes = ["DF_all",  "DF_artery", "DF_vein"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    #list_phenotypes = ['std_intensity', 'mean_intensity', 'median_intensity']
    #f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    #list_phenotypes = ['D_median_std', 'D_mean_std', 'D_std_std']
    #f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    #list_phenotypes = ['D_A_median_std', 'D_A_mean_std', 'D_A_std_std']
    #f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    #list_phenotypes = ['D_V_median_std', 'D_V_mean_std', 'D_V_std_std']
    #f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    list_phenotypes = [ "VD_orig_all", "VD_orig_artery", "VD_orig_vein"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    list_phenotypes = ['slope', 'slope_artery', 'slope_vein']
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    list_phenotypes = ["ratio_CRAE_CRVE", "eq_CRAE", "eq_CRVE"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    list_phenotypes = ["ratio_median_CRAE_CRVE", "median_CRAE", "median_CRVE"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    ### Four multiple histograms plot
    list_phenotypes = ["medianDiameter_all",  "medianDiameter_artery", "medianDiameter_vein", "ratio_AV_medianDiameter"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    list_phenotypes = ["DF_all",  "DF_artery", "DF_vein", "ratio_AV_DF"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    #list_phenotypes = ["ratio_AV_medianDiameter",  "ratio_VA_medianDiameter", "ratio_AV_DF", "ratio_VA_DF"]
    #f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)
