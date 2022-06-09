# Sofia 09-06-2022

import pandas as pd
import seaborn as sns
import os
import glob
import numpy as np
import matplotlib as plt
import functions_plot as f_plot

phenofiles_dir='/HDD/data/UKBiob/phenofiles/'
save_dir='/SSD/home/sofia/retina-phenotypes/complementary/traits_main_characterization/results/'
plot_histograms=False

all_phenotypes = ['bifurcations', 'median_CRAE', 'eq_CRAE', 'median_CRVE', 'eq_CRVE','ratio_median_CRAE_CRVE', 'ratio_CRAE_CRVE', 'medianDiameter_all', 'medianDiameter_artery', 'medianDiameter_vein', 'DF_all', 'DF_artery', 'DF_vein', 'ratio_AV_medianDiameter', 'ratio_VA_medianDiameter','ratio_AV_DF', 'ratio_VA_DF', 'mean_angle_taa', 'mean_angle_tva', 'D_median_std', 'D_mean_std', 'D_std_std', 'D_A_median_std', 'D_A_mean_std', 'D_A_std_std', 'D_V_median_std', 'D_V_mean_std', 'D_V_std_std', 'VD_orig_all', 'VD_orig_artery', 'VD_orig_vein', 'VD_small_all', 'VD_small_artery', 'VD_small_vein', 'slope', 'slope_artery', 'slope_vein', 'std_intensity', 'mean_intensity', 'median_intensity']

#### Read file:
df_data_completo = pd.read_csv(phenofiles_dir + "2022_06_08_all_phenotypes_LWNet_Decile3.csv", sep=' ')
print(df_data_completo.columns)

# Reemplazar los -999 por np.NaN
# Comparar si borrando los NaNs disminuye mucho la muestra
print("Sample size before delete the -999: ", len(df_data_completo))
df_data_completo = df_data_completo[df_data_completo != -999]
df_data_completo = df_data_completo.dropna()
print("Sample size after delete the -999: ", len(df_data_completo))

################### Histograms: #############################
if plot_histograms==True:
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

    list_phenotypes = ['std_intensity', 'mean_intensity', 'median_intensity']
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    list_phenotypes = ['D_median_std', 'D_mean_std', 'D_std_std']
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    list_phenotypes = ['D_A_median_std', 'D_A_mean_std', 'D_A_std_std']
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    list_phenotypes = ['D_V_median_std', 'D_V_mean_std', 'D_V_std_std']
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    list_phenotypes = [ "VD_orig_all", "VD_orig_artery", "VD_orig_vein"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    list_phenotypes = [ 'slope', 'slope_artery', 'slope_vein']
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

    list_phenotypes = ["ratio_AV_medianDiameter",  "ratio_VA_medianDiameter", "ratio_AV_DF", "ratio_VA_DF"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)


################### Scatter plots + histograms in the diagonal : #############################

main_phenotypes = ["bifurcations",  "mean_angle_taa", "mean_angle_tva", "eq_CRAE", "eq_CRVE", "medianDiameter_all", "DF_all"]


### Multiple scatter plots with lines and histogram in the diagonal
f_plot.multiple_scatter_plots(df_data_completo, main_phenotypes, save_dir)

### Multiple scatter plots with corr values in the first half and histogram in the diagonal
f_plot.multiple_scatter_plots_2(df_data_completo, main_phenotypes, save_dir)

### Scatter plots with corr values in the first half and histogram in the diagonal
f_plot.multiple_scatter_plots_3(df_data_completo, main_phenotypes, save_dir)
