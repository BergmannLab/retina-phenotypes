import pandas as pd
import seaborn as sns
import os
import glob
from datetime import datetime
import numpy as np
import matplotlib as plt
from matplotlib import figure
import matplotlib.pyplot as plt_py


DATE = datetime.now().strftime("%Y-%m-%d")


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
    ####,dropna=True
    sns.set(font_scale = 2)
    sns.set_context("paper", rc={"axes.labelsize":28})
    sns_plot.savefig(save_dir + DATE +'_seaborn_type1.png')

                            
### Scatter plots with corr values in the first half and histogram in the diagonal
def multiple_scatter_plots_2(df_data_completo, main_phenotypes, save_dir):
    # Create a pair grid instance
    grid = sns.PairGrid(data=df_data_completo, vars=main_phenotypes, size=None)  ####,dropna=True
    # Map the plots to the locations
    grid = grid.map_upper(plt.pyplot.scatter)
    grid = grid.map_upper(corr)
    grid = grid.map_diag(plt.pyplot.hist, bins=15, edgecolor='k') 
    grid = grid.map_lower(sns.scatterplot)
    grid.savefig(save_dir + DATE +'_seaborn_type2.png')
    
### Scatter plots with corr values in the first half, histogram in the diagonal, and maps in the second half
def multiple_scatter_plots_3(df_data_completo, main_phenotypes, save_dir):    
    # Create a pair grid instance
    #grid = sns.pairplot(df_data_completo[main_phenotypes], diag_kind="hist",  kind="reg",
                           # plot_kws={'scatter_kws': {'alpha': 0.8, 's': 0.5}})
    grid = sns.set_context("paper", rc={"axes.labelsize":18})
    grid = sns.PairGrid(data=df_data_completo, vars=main_phenotypes)  ####,dropna=True
    grid = grid.map_upper(sns.scatterplot, cmap="Blues_d")
    grid = grid.map_upper(corr)
    grid = grid.map_diag(plt.pyplot.hist, bins=15, edgecolor='k')
    #grid = grid.map_lower(sns.scatterplot, color=".1")
    grid = grid.map_lower(sns.scatterplot, alpha=0.2)
    grid = grid.map_lower(sns.kdeplot, cmap="Greys_d")
    #grid = grid.map_lower(sns.kdeplot, levels=5, color=".1")
    grid.savefig(save_dir + DATE +'_seaborn_type3.png')

    
### Violin plot
def violin_plot(df_data_completo, list_phenotypes, save_dir, my_pal, outlier1=False):
    df_data_completo2 = df_data_completo
    df_data_completo=df_data_completo[list_phenotypes]
    df_data_completo = df_data_completo.melt(var_name='phenotypes', value_name='distribution')
    #var1=list_phenotypes[0]
    #sns.set(style="whitegrid")
    sns.set_style("white")
    #ax = sns.violinplot(x="phenotypes", y="distribution", data=df_data_completo, palette=my_pal, saturation=0.8)


    if len(list_phenotypes)==1:
        var1=list_phenotypes[0]
        size_1 = len(df_data_completo2[var1])- df_data_completo2[var1].isna().sum()
        if outlier1!=False:
            ax = sns.violinplot(x="phenotypes", y = df_data_completo2[df_data_completo2[var1]<outlier1][var1], data=df_data_completo, palette=my_pal, 
                            saturation=0.8)
        else:    
            ax = sns.violinplot(x="phenotypes", y="distribution", data=df_data_completo, palette=my_pal, saturation=0.8)
        ax.set_title( "N = " + str(size_1))
        
    if len(list_phenotypes)==2:
        var1=list_phenotypes[0]
        var2=list_phenotypes[1]
        if outlier1!=False:
            print(df_data_completo)
            df_data_completo = df_data_completo.loc[(df_data_completo['distribution'] <= outlier1)]
        size_1 = len(df_data_completo2[var1])- df_data_completo2[var1].isna().sum()
        size_2 = len(df_data_completo2[var2])- df_data_completo2[var2].isna().sum()

        ax = sns.violinplot(x="phenotypes", y="distribution", data=df_data_completo, palette=my_pal, 
                            saturation=0.8)
        ax.set_title( "N = " + str(size_1) + ", "  + str(size_2))

    elif len(list_phenotypes)==3:
        var1=list_phenotypes[0]
        var2=list_phenotypes[1]
        var3=list_phenotypes[2]
        if outlier1!=False:
            print(df_data_completo)
            df_data_completo = df_data_completo.loc[(df_data_completo['distribution'] <= outlier1)]

        size_1 = len(df_data_completo2[var1])- df_data_completo2[var1].isna().sum()
        size_2 = len(df_data_completo2[var2])- df_data_completo2[var2].isna().sum()
        size_3 = len(df_data_completo2[var3])- df_data_completo2[var3].isna().sum()
        ax = sns.violinplot(x="phenotypes", y="distribution", data=df_data_completo, palette=my_pal, 
                            saturation=0.8) 
        ax.set_title( "N = " + str(size_1) + ", "  + str(size_2) + ", " +  str(size_3))

    elif len(list_phenotypes)==4:
        var1=list_phenotypes[0]
        var2=list_phenotypes[1]
        var3=list_phenotypes[2]
        var4=list_phenotypes[3]

        size_1 = len(df_data_completo2[var1])- df_data_completo2[var1].isna().sum()
        size_2 = len(df_data_completo2[var2])- df_data_completo2[var2].isna().sum()
        size_3 = len(df_data_completo2[var3])- df_data_completo2[var3].isna().sum()
        size_4 = len(df_data_completo2[var4])- df_data_completo2[var4].isna().sum()
        ax = sns.violinplot(x="phenotypes", y="distribution", data=df_data_completo, palette=my_pal, 
                            saturation=0.8) 
        ax.set_title( "N = " + str(size_1) + ", "  + str(size_2) + ", " +  str(size_3)+ ", " +  str(size_4))

    elif len(list_phenotypes)==5:
        var1=list_phenotypes[0]
        var2=list_phenotypes[1]
        var3=list_phenotypes[2]
        var4=list_phenotypes[3]
        var5=list_phenotypes[4]

        size_1 = len(df_data_completo2[var1])- df_data_completo2[var1].isna().sum()
        size_2 = len(df_data_completo2[var2])- df_data_completo2[var2].isna().sum()
        size_3 = len(df_data_completo2[var3])- df_data_completo2[var3].isna().sum()
        size_4 = len(df_data_completo2[var4])- df_data_completo2[var4].isna().sum()
        size_5 = len(df_data_completo2[var5])- df_data_completo2[var5].isna().sum()
        ax = sns.violinplot(x="phenotypes", y="distribution", data=df_data_completo, palette=my_pal, 
                            saturation=0.8) 
        ax.set_title( "N = " + str(size_1) + ", "  + str(size_2) + ", " +  str(size_3)
                     + ", " +  str(size_4)+ ", " +  str(size_5))

    if outlier1!=False:
        plt_py.savefig(save_dir + DATE + '_'.join(list_phenotypes) +'_violinplot_outliers_mod.png')
    else:
        plt_py.savefig(save_dir + DATE + '_'.join(list_phenotypes) +'_violinplot.png')
    plt_py.close()



### Histograms
def multiple_histograms(df_data_completo, list_phenotypes, save_dir):
    if len(list_phenotypes)==1:
        var1=list_phenotypes[0]

        sns.set_style("white")
        size_1 = len(df_data_completo[var1])- df_data_completo[var1].isna().sum()

        sns.histplot(df_data_completo[var1], color="dimgray", label= str(var1) +" (N=" + str(size_1) + ")"
                     , alpha=0.75)
        plt_py.legend()
        plt_py.savefig(save_dir + DATE +'_'.join(list_phenotypes) +'_histograms.png')
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
        plt_py.savefig(save_dir + DATE +'_'.join(list_phenotypes)+'_histograms.png')
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
        plt_py.savefig(save_dir + DATE +'_'.join(list_phenotypes)+'_histograms.png')
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
        plt_py.savefig(save_dir + DATE + '_'.join(list_phenotypes)+'_histograms.png')
        plt_py.close()