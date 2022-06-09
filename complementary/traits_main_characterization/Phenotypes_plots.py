import pandas as pd
import seaborn as sns
import os
import glob
import numpy as np
import matplotlib as plt

# df_data_completo = pd.read_csv("/data_manual_and_DL_features.csv", sep=' ')
# # Reemplazar los -999 por np.NaN
# # Comparar si borrando los NaNs disminuye mucho la muestra
# sns_plot = sns.pairplot(df_data_completo[["0", "1", "2", "3"]], diag_kind="hist",  kind="reg")
# sns_plot.savefig('example_phenofile_seaborn.png')

phenofiles_dir='/HDD/data/UKBiob/phenofiles/'
save_dir='/SSD/home/sofia/retina-phenotypes/complementary/traits_main_characterization/results/'

df_data_completo = pd.read_csv(phenofiles_dir + "2022_06_08_all_phenotypes_LWNet_Decile3_qqnorm.csv", sep=' ')
print(df_data_completo.columns)
# Reemplazar los -999 por np.NaN
# Comparar si borrando los NaNs disminuye mucho la muestra
print("Sample size before delete the -999: ", len(df_data_completo))
df_data_completo = df_data_completo[df_data_completo != -999]
df_data_completo = df_data_completo.dropna()
print("Sample size after delete the -999: ", len(df_data_completo))



all_phenotypes =["bifurcations", "median_CRAE", "eq_CRAE", "median_CRVE", "eq_CRVE", "ratio_median_CRAE_CRVE", "ratio_CRAE_CRVE", 
"medianDiameter_artery", "medianDiameter_vein", "DF_artery", "DF_vein", "ratio_AV_medianDiameter", 
"ratio_VA_medianDiameter", "ratio_AV_DF", "ratio_VA_DF", "mean_angle_taa", "mean_angle_tva", "D_median_std", "D_mean_std", "D_std_std", "D_A_median_std", "D_A_mean_std", "D_A_std_std", "D_V_median_std", "D_V_mean_std", "D_V_std_std", "VD_orig_all","VD_orig_artery", "VD_orig_vein", "VD_small_all", "VD_small_artery", "VD_small_vein", "std_intensity", "mean_intensity", "median_intensity"]

main_phenotypes = ["bifurcations",  "mean_angle_taa", "mean_angle_tva", "eq_CRAE", "eq_CRVE", 
"medianDiameter_artery", "medianDiameter_vein", "DF_artery", "DF_vein", "ratio_AV_medianDiameter", "ratio_AV_DF", "VD_orig_all"]


### First type of image -  Simple version: Scatter plots with lines and histogram in the diagonal
sns_plot = sns.pairplot(df_data_completo[main_phenotypes], diag_kind="hist",  kind="reg",
                        plot_kws={'scatter_kws': {'alpha': 0.8, 's': 0.5}
                                    # ,'line_kws':{'color':'red'}
                                  })

sns.set(font_scale = 2)
sns.set_context("paper", rc={"axes.labelsize":28})
sns_plot.savefig(save_dir + 'example_type1_seaborn.png')


##### Second type of image: Scatter plots with corr values in the first half and histogram in the diagonal
# Function to calculate correlation coefficient between two arrays
def corr(x, y, **kwargs):
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    # Add the label to the plot
    ax = plt.pyplot.gca()
    ax.annotate(label, xy=(0.2, 0.95), size=20, xycoords=ax.transAxes)


# Create a pair grid instance
grid = sns.PairGrid(data=df_data_completo,
                    vars=main_phenotypes, size=4)

# Map the plots to the locations
grid = grid.map_upper(plt.pyplot.scatter)
grid = grid.map_upper(corr)
grid = grid.map_diag(plt.pyplot.hist, bins=10, edgecolor='k')
grid = grid.map_lower(sns.scatterplot)
grid.savefig(save_dir + 'example_type2_seaborn.png')


print(1)
