import pandas as pd
import seaborn as sns
import os, sys
import glob
import numpy as np
import matplotlib as plt
import functions_plot as f_plot

phenofiles_dir ='/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/participant_phenotype/' #sys.argv[1] 
phenofile_used_for_dist_plots = "2022_07_08_ventile2_raw.csv" # sys.argv[2]
save_dir ='/SSD/home/sofia/retina-phenotypes/complementary/traits_main_characterization/results/' #sys.argv[3]
main_phenotypes_rawname = list(sys.argv[4].split(",")) #["bifurcations", "VD_orig_all", "slope", "mean_angle_taa", "mean_angle_tva", "eq_CRAE", "eq_CRVE", "medianDiameter_all", "DF_all"]
plot_violin = sys.argv[5] #True
plot_histograms = sys.argv[6] # False
plot_seaborn = sys.argv[7] #True
main_phenotypes_new_names = list(sys.argv[8].split(","))

print(phenofiles_dir, phenofile_used_for_dist_plots, save_dir)

supplementary=False
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


#### Apply an renaming of the columns:
df_data_completo.rename(columns=dict(zip(main_phenotypes_rawname, main_phenotypes_new_names)), inplace=True)
#print(df_data_completo)

################### Violin plots: #############################
# Relevant documents to create new versions: https://www.oreilly.com/library/view/python-data-science/9781491912126/ch04.html
if plot_violin=='True':
    list_phenotypes = ["medianDiameter_artery", "medianDiameter_vein"]
    my_pal= {"medianDiameter_artery": "lightcoral", "medianDiameter_vein": "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)

    list_phenotypes = ["mean_angle_taa", "mean_angle_tva"]
    my_pal= {"mean_angle_taa": "lightcoral", "mean_angle_tva": "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    
    list_phenotypes = ["eq_CRAE", "eq_CRVE"]
    my_pal= {"eq_CRAE": "lightcoral", "eq_CRVE": "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    
    list_phenotypes = [ "DF_artery", "DF_vein"]
    my_pal= {"DF_artery": "lightcoral", "DF_vein": "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    ## correct for outlier
    df_data_completo_1 = df_data_completo.sort_values(by=['DF_artery'],  ascending=False)
    print(df_data_completo_1['DF_artery'])
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal, 1.2)
    
    list_phenotypes = [ "VD_artery", "VD_vein"]
    my_pal= {"VD_artery": "lightcoral", "VD_vein": "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    
    list_phenotypes = ['D_A_std', 'D_V_std']
    my_pal= {"D_A_std": "lightcoral", "D_V_std": "cornflowerblue"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    
    list_phenotypes = ["ratio_medianDiameter", "ratio_CRAE_CRVE", "ratio_DF"]
    my_pal= {"ratio_medianDiameter": "g", "ratio_CRAE_CRVE": "b", "ratio_DF": "m"}
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    ## correct for outlier
    df_data_completo_ = df_data_completo.sort_values(by=['ratio_medianDiameter'],  ascending=False)
    print(df_data_completo_['ratio_medianDiameter'].head(7))  
    df_data_completo__ = df_data_completo.sort_values(by=['ratio_DF'],  ascending=False)
    print(df_data_completo__['ratio_DF'].head(7))
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal, 4)
    
    list_phenotypes = ['ratio_VD']
    my_pal="Set3"
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    ## correct for outlier ## it looks more slim (?)
    #df_data_completo_2 = df_data_completo.sort_values(by=['ratio_VD'],  ascending=False)
    #print(df_data_completo_2['ratio_VD'].head(15))
    #f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal, 10)
    
    list_phenotypes = ['bifurcations']
    my_pal="Set3"
    f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)


    if supplementary==True:
        list_phenotypes = ["medianDiameter_all", "medianDiameter_artery", "medianDiameter_vein"]
        my_pal= {"medianDiameter_all": "gray", "medianDiameter_artery": "lightcoral", "medianDiameter_vein": "cornflowerblue"}
        f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
    
        list_phenotypes = ["DF_all", "DF_artery", "DF_vein"]
        my_pal= {"DF_all": "gray", "DF_artery": "lightcoral", "DF_vein": "cornflowerblue"}
        f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)

        list_phenotypes = ["VD_orig_all", "VD_orig_artery", "VD_orig_vein"]
        my_pal= {"VD_orig_all": "gray", "VD_orig_artery": "lightcoral", "VD_orig_vein": "cornflowerblue"}
        f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)
        
        list_phenotypes = ['slope', 'slope_artery', 'slope_vein']
        my_pal= {"slope": "gray", "slope_artery": "lightcoral", "slope_vein": "cornflowerblue"}
        f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)

        list_phenotypes = ['bifurcations', 'std_intensity']
        my_pal="Set3"
        f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)

        list_phenotypes = ['D_median_CVMe', 'D_A_std_std', 'D_V_std_std']
        my_pal="Set3"
        f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)

        list_phenotypes = ["N_median_main_arteries", "N_median_main_veins"]
        my_pal= {"N_median_main_arteries": "lightcoral", "N_median_main_veins": "cornflowerblue"}
        f_plot.violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal)

################### Histograms: #############################
# Relevant documents to create new versions: https://www.oreilly.com/library/view/python-data-science/9781491912126/ch04.html

if plot_histograms=='True':
    ### One multiple histograms plot
    list_phenotypes = ["bifurcations"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    ### Two multiple histograms plot
    list_phenotypes = ["mean_angle_taa",  "mean_angle_tva"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    
    list_phenotypes = ["medianDiameter_artery", "medianDiameter_vein"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)

    
    list_phenotypes = ["eq_CRAE", "eq_CRVE"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)
    
    list_phenotypes = [ "DF_artery", "DF_vein"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)
    
    list_phenotypes = [ "VD_artery", "VD_vein"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)
    
    list_phenotypes = ['D_A_std', 'D_V_std']
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)
    
    
    list_phenotypes = ['ratio_VD']
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)
    
    ### Three multiple histograms plot

    list_phenotypes = ["ratio_medianDiameter", "ratio_CRAE_CRVE", "ratio_DF"]
    f_plot.multiple_histograms(df_data_completo, list_phenotypes, save_dir)



################### Scatter plots + histograms in the diagonal : #############################
## Relevant documentation to create new versions: 
# https://seaborn.pydata.org/generated/seaborn.pairplot.html, https://stackoverflow.com/questions/71217352/seaborn-pairgrid-pairplot-two-data-set-with-different-transparency
if plot_seaborn=='True':
    ### Multiple scatter plots with lines and histogram in the diagonal
    f_plot.multiple_scatter_plots(df_data_completo, main_phenotypes_new_names, save_dir)

    ### Multiple scatter plots with corr values in the first half and histogram in the diagonal
    f_plot.multiple_scatter_plots_2(df_data_completo, main_phenotypes_new_names, save_dir)

    ### Scatter plots with corr values in the first half and histogram in the diagonal
    f_plot.multiple_scatter_plots_3(df_data_completo, main_phenotypes_new_names, save_dir)
