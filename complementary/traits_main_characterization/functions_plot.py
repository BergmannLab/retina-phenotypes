import pandas as pd
import seaborn as sns
import os
import glob
import numpy as np
import matplotlib as plt
from matplotlib import figure
import matplotlib.pyplot as plt_py

# Function to calculate correlation coefficient between two arrays
def corr(x, y, **kwargs):
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    # Add the label to the plot
    ax = plt.pyplot.gca()
    ax.annotate(label, xy=(0.2, 0.95), size=20, xycoords=ax.transAxes)


### Scatter plots with lines and histogram in the diagonal
def multiple_scatter_plots(df_data_completo, main_phenotypes, save_dir):
    sns_plot = sns.pairplot(df_data_completo[main_phenotypes], diag_kind="hist",  kind="reg",
                            plot_kws={'scatter_kws': {'alpha': 0.8, 's': 0.5}}) # ,'line_kws':{'color':'red'}})
    sns.set(font_scale = 2)
    sns.set_context("paper", rc={"axes.labelsize":28})
    sns_plot.savefig(save_dir + 'example_type1_seaborn.png')

                            
### Scatter plots with corr values in the first half and histogram in the diagonal
def multiple_scatter_plots_2(df_data_completo, main_phenotypes, save_dir):
    # Create a pair grid instance
    grid = sns.PairGrid(data=df_data_completo, vars=main_phenotypes, size=None)
    # Map the plots to the locations
    grid = grid.map_upper(plt.pyplot.scatter)
    grid = grid.map_upper(corr)
    grid = grid.map_diag(plt.pyplot.hist, bins=10, edgecolor='k')
    grid = grid.map_lower(sns.scatterplot)
    grid.savefig(save_dir + 'example_type2_seaborn.png')
    
### Scatter plots with corr values in the first half, histogram in the diagonal, and maps in the second half
def multiple_scatter_plots_3(df_data_completo, main_phenotypes, save_dir):    
    # Create a pair grid instance
    #grid = sns.pairplot(df_data_completo[main_phenotypes], diag_kind="hist",  kind="reg",
                           # plot_kws={'scatter_kws': {'alpha': 0.8, 's': 0.5}})
    grid = sns.set_context("paper", rc={"axes.labelsize":18})
    grid = sns.PairGrid(data=df_data_completo, vars=main_phenotypes)
    grid = grid.map_upper(sns.scatterplot, cmap="Blues_d")
    grid = grid.map_upper(corr)
    grid = grid.map_diag(plt.pyplot.hist, bins=10, edgecolor='k')
    #grid = grid.map_lower(sns.scatterplot, color=".1")
    grid = grid.map_lower(sns.scatterplot, alpha=0.2)
    grid = grid.map_lower(sns.kdeplot, cmap="Greys_d")
    #grid = grid.map_lower(sns.kdeplot, levels=5, color=".1")
    grid.savefig(save_dir + 'example_type3_seaborn.png')


def multiple_histograms(df_data_completo, list_phenotypes, save_dir):
    if len(list_phenotypes)==1:
        var1=list_phenotypes[0]

        sns.set_style("white")
        size_1 = len(df_data_completo[var1])- df_data_completo[var1].isna().sum()

        sns.histplot(df_data_completo[var1], color="dimgray", label= str(var1) +" (N=" + str(size_1) + ")"
                     , alpha=0.75)
        plt_py.legend()
        plt_py.savefig(save_dir + str(var1)+'_histograms.png')
        plt_py.close()
        
    elif len(list_phenotypes)==2:
        var1=list_phenotypes[0]
        var2=list_phenotypes[1]

        sns.set_style("white")
        size_1 = len(df_data_completo[var1])- df_data_completo[var1].isna().sum()
        size_2 = len(df_data_completo[var2])- df_data_completo[var2].isna().sum()

        sns.histplot(df_data_completo[var1], color="lightcoral", label= str(var1) +" (N=" + str(size_1) + ")"
                     , alpha=0.75)
        sns.histplot(df_data_completo[var2],color="cornflowerblue", label=str(var2) +" (N=" + str(size_2) + ")" 
                     , alpha=0.75)
        plt_py.legend()
        plt_py.savefig(save_dir + str(var1)+'_'+ str(var2)+'_histograms.png')
        plt_py.close()
    
    elif len(list_phenotypes)==3:
        var1=list_phenotypes[0]
        var2=list_phenotypes[1]
        var3=list_phenotypes[2]

        sns.set_style("white")
        size_1 = len(df_data_completo[var1])- df_data_completo[var1].isna().sum()
        size_2 = len(df_data_completo[var2])- df_data_completo[var2].isna().sum()
        size_3 = len(df_data_completo[var3])- df_data_completo[var3].isna().sum()

        sns.histplot(df_data_completo[var2],color="lightcoral", label=str(var2) +" (N=" + str(size_2) + ")" 
                     , alpha=0.75)
        sns.histplot(df_data_completo[var3], color="cornflowerblue", label=str(var3) +" (N=" + str(size_3) + ")" 
                     , alpha=0.75)
        sns.histplot(df_data_completo[var1], color="dimgray", label= str(var1) +" (N=" + str(size_1) + ")", 
                     alpha=0.75)
        plt_py.legend()
        plt_py.savefig(save_dir + str(var1)+'_'+ str(var2)+'_'+str(var3)+'_histograms.png')
        plt_py.close()
    
    elif len(list_phenotypes)==4:
        var1=list_phenotypes[0]
        var2=list_phenotypes[1]
        var3=list_phenotypes[2]
        var4=list_phenotypes[3]

        sns.set_style("white")
        size_1 = len(df_data_completo[var1])- df_data_completo[var1].isna().sum()
        size_2 = len(df_data_completo[var2])- df_data_completo[var2].isna().sum()
        size_3 = len(df_data_completo[var3])- df_data_completo[var3].isna().sum()
        size_4 = len(df_data_completo[var4])- df_data_completo[var4].isna().sum()

        sns.histplot(df_data_completo[var2],color="lightcoral", label=str(var2) +" (N=" + str(size_2) + ")" 
                     , alpha=0.75)
        sns.histplot(df_data_completo[var3], color="cornflowerblue", label=str(var3) +" (N=" + str(size_3) + ")" 
                     , alpha=0.75)
        sns.histplot(df_data_completo[var1], color="dimgray", label= str(var1) +" (N=" + str(size_1) + ")", 
                     alpha=0.75)
        sns.histplot(df_data_completo[var4], color="mediumpurple", label= str(var4) +" (N=" + str(size_4) + ")", 
                     alpha=0.75)
        plt_py.legend()
        plt_py.savefig(save_dir + str(var1)+'_'+ str(var2)+'_'+str(var3)+'_'+str(var4)+'_histograms.png')
        plt_py.close()