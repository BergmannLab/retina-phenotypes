import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import matplotlib.gridspec as gridspec


DATE = datetime.now().strftime("%Y-%m-%d")

def corr(x, y, **kwargs):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
    """
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    # Add the label to the plot
    ax = plt.pyplot.gca()
    ax.annotate(label, xy=(0.2, 0.95), size=20, xycoords=ax.transAxes)


def corr_heatmap(df_pheno_dise, output_dir, save_fig=False):
    """_summary_

    Args:
        df_pheno_dise (_type_): _description_
        output_dir (_type_): _description_
        save_fig (bool, optional): _description_. Defaults to False.
    """
    matrix_total = df_pheno_dise.corr().round(1)
    plt.figure(figsize=(15, 15), dpi = 900)
    fig = sns.heatmap(matrix_total, annot=True, vmax=1, vmin=-1, center=0, annot_kws={'size': 3}, cmap='vlag')
    plt.show()
    if save_fig:
        plt.savefig(output_dir + str(DATE)+'_corr_heatmap.jpg', facecolor='white', bbox_inches='tight', pad_inches=0.1, dpi=150)


def hist_diseases_plot(df_diseases, list_diseases, output_dir, save_fig):
    """_summary_

    Args:
        df_diseases (_type_): _description_
        list_diseases (_type_): _description_
        output_dir (_type_): _description_
        save_fig (_type_): _description_
    """
    df_diseases_red = df_diseases[list_diseases]

    fig = plt.figure(figsize = (12,10))
    ax = fig.gca()
    df_diseases_red.hist(ax = ax)
    if save_fig:
        plt.savefig(output_dir + str(DATE)+'_MLRdiseases_hist.jpg', facecolor='white', bbox_inches='tight', pad_inches=0.1, dpi=150)


def format_axes(fig):
    """_summary_

    Args:
        fig (_type_): _description_
    """
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


def plot_pvalues_signif_values(df_log10p):
    """_summary_

    Args:
        df_log10p (_type_): _description_

    Returns:
        _type_: _description_
    """
    Bonf_thresh_linear = -np.log10(0.05 / (df_log10p.shape[0] * df_log10p.shape[1]))
    Bonf_thresh2_linear = -np.log10(0.001 / (df_log10p.shape[0] * df_log10p.shape[1]))

    linear_log10p_copy = df_log10p.copy()
    linear_log10p_copy2 = df_log10p.copy()
    linear_log10p_copy3 = df_log10p.copy()

    linear_log10p_copy = (linear_log10p_copy>Bonf_thresh_linear).replace({True:'*', False:''})
    linear_log10p_copy2 = (linear_log10p_copy2>Bonf_thresh2_linear).replace({True:'*', False:''})
    linear_log10p_copy3 =linear_log10p_copy+linear_log10p_copy2

    return linear_log10p_copy, linear_log10p_copy2,linear_log10p_copy3


def base_parameters(font_size_val):
    """_summary_

    Args:
        font_size_val (_type_): _description_
    """
    plt.rcParams['figure.constrained_layout.use'] = True
    #ax1.yaxis.set_ticks_position('right')
    plt.rcParams['font.size'] = font_size_val


def figure_betas_pval_linear_log(betas_linear, linear_log10p_copy3, betas_logistic, log_log10p_copy3, figsize_val=(18, 10), height_ratios_val=None, font_size_val='12'):
    if height_ratios_val is None:
        height_ratios_val = [0.7, 1.4]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize_val, gridspec_kw={'height_ratios': height_ratios_val})

    base_parameters(font_size_val)

    sns.heatmap(betas_linear.T, 
                annot=linear_log10p_copy3.T,
                cbar=True,
                fmt="", annot_kws={'weight': 'bold'},
                vmin=-abs(betas_linear.T).max().max(),
                vmax=abs(betas_linear.T).max().max(),
                cmap='seismic', alpha=1.0, cbar_kws={'label': 'Standardised \u03B2'},
                ax=ax1)

    sns.heatmap(betas_logistic.T, 
                annot=log_log10p_copy3.T,
                cbar=True,
                fmt="", annot_kws={'weight': 'bold'},
                vmin=-abs(betas_logistic.T).max().max(),
                vmax=abs(betas_logistic.T).max().max(),
                cmap='seismic', alpha=1.0, cbar_kws={'label': 'Standardised \u03B2'},
                ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    #fig.tight_layout()  # Adjust subplot spacing

    return fig, (ax1, ax2)


def figure_betas_pval_linear_log_old(betas_linear, linear_log10p_copy3, betas_logistic, log_log10p_copy3, figsize_val=(18, 10), height_ratios_val = None, font_size_val='12'):
    """_summary_

    Args:
        betas_linear (_type_): _description_
        linear_log10p_copy3 (_type_): _description_
        betas_logistic (_type_): _description_
        log_log10p_copy3 (_type_): _description_
        figsize_val (tuple, optional): _description_. Defaults to (18, 10).
        height_ratios_val (_type_, optional): _description_. Defaults to None.
        font_size_val (str, optional): _description_. Defaults to '12'.
    """
    if height_ratios_val is None:
        height_ratios_val = [0.7, 1.4]
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize_val, gridspec_kw={'height_ratios': height_ratios_val})

    base_parameters(font_size_val)

    fig1 = sns.heatmap(betas_linear.T, 
                annot=linear_log10p_copy3.T, #(log10p>Bonf_thresh).replace({True:'*', False:''}), 
                cbar=True, #If not False
                fmt="", annot_kws={'weight': 'bold'}, 
                vmin=-abs(betas_linear.T).max().max(), ## combined
                vmax=abs(betas_linear.T).max().max(), 
                cmap='seismic',alpha=1.0, cbar_kws={'label': 'Standardised \u03B2'},
                ax=ax1)
    #fig1.set_xticklabels(ax1.get_xticklabels(), rotation = 45, ha='right', visible=False)

    fig2 = sns.heatmap(betas_logistic.T, 
                annot=log_log10p_copy3.T, #(log10p>Bonf_thresh).replace({True:'*', False:''}), 
                cbar=True, #False
                fmt="", annot_kws={'weight': 'bold'}, 
                vmin=-abs(betas_logistic.T).max().max(), 
                vmax=abs(betas_logistic.T).max().max(), 
                cmap='seismic', alpha=1.0, cbar_kws={'label': 'Standardised \u03B2'},
                ax=ax2)
    fig2.set_xticklabels(ax2.get_xticklabels(), rotation = 45, ha='right')
    #plt.ylabel('Logistic regresion')


def figure_betas_pval(betas, log10p_copy3, figsize_val=(18, 10), font_size_val='12'):
    """_summary_

    Args:
        betas (_type_): _description_
        log10p_copy3 (_type_): _description_
        figsize_val (tuple, optional): _description_. Defaults to (18, 10).
        font_size_val (str, optional): _description_. Defaults to '12'.
    """
    fig, ax = plt.subplots(figsize=figsize_val)
    base_parameters(font_size_val)
    fig = sns.heatmap(betas.T, 
                annot=log10p_copy3.T, #(log10p>Bonf_thresh).replace({True:'*', False:''}), 
                cbar=True, #If not False
                fmt="", annot_kws={'weight': 'bold'}, 
                vmin=-abs(betas.T).max().max(), 
                vmax=abs(betas.T).max().max(), 
                cmap='seismic',alpha=1.0, cbar_kws={'label': 'Standardised \u03B2'},
                ax = ax)
    fig.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha='right')


def figure_betas_pval_cox_linear_log(betas_linear, linear_log10p_copy3, betas_logistic, log_log10p_copy3, df_cox_hazar_ratio, log10p_copy3_cox, figsize_val=(10, 10), height_ratios_val = None, font_size_val='12'):
    """_summary_

    Args:
        betas_linear (_type_): _description_
        linear_log10p_copy3 (_type_): _description_
        betas_logistic (_type_): _description_
        log_log10p_copy3 (_type_): _description_
        df_cox_hazar_ratio (_type_): _description_
        log10p_copy3_cox (_type_): _description_
        figsize_val (tuple, optional): _description_. Defaults to (10, 10).
        height_ratios_val (_type_, optional): _description_. Defaults to None.
        font_size_val (str, optional): _description_. Defaults to '12'.
    """
    if height_ratios_val is None:
        height_ratios_val = [2.9, 2.24, 2.8]

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize_val, gridspec_kw={'height_ratios': height_ratios_val})

    base_parameters(font_size_val)

    fig1 = sns.heatmap(betas_linear.T, 
                annot=linear_log10p_copy3.T, 
                cbar=True, #If not False
                fmt="", annot_kws={'weight': 'bold'}, 
                vmin=-abs(betas_linear).max().max(), ## combined
                vmax=abs(betas_linear).max().max(), 
                cmap='seismic',alpha=1.0, cbar_kws={'label': 'Standardised \u03B2'},
                    ax=ax1)
    fig1.set_xticklabels(ax1.get_xticklabels(), rotation = 45, ha='right', visible=False)
    plt.ylabel('Linear regresion')

    #ax2.yaxis.set_ticks_position('right')
    fig2 = sns.heatmap(betas_logistic.T, 
                annot=log_log10p_copy3.T, #(log10p>Bonf_thresh).replace({True:'*', False:''}), 
                cbar=True, #False
                fmt="", annot_kws={'weight': 'bold'}, 
                vmin=-abs(betas_logistic).max().max(), 
                vmax=abs(betas_logistic).max().max(), 
                cmap='seismic', alpha=1.0, cbar_kws={'label': 'Standardised \u03B2'},
                    ax=ax2)
    fig2.set_xticklabels(ax2.get_xticklabels(), rotation = 45, ha='right', visible=False)
    plt.ylabel('Logistic regresion')

    #ax2.yaxis.set_ticks_position('right')
    fig3 = sns.heatmap(df_cox_hazar_ratio.T, 
                annot=log10p_copy3_cox.T, #(log10p>Bonf_thresh).replace({True:'*', False:''}), 
                cbar=True, #False
                fmt="", annot_kws={'weight': 'bold'}, 
                vmin=(df_cox_hazar_ratio).min().min(), 
                vmax=(df_cox_hazar_ratio).max().max(), 
                center=1.0,
                cmap='seismic', alpha=1.0, cbar_kws={'label': 'Hazard ratio '}, ## PuOr_r , RdBu_r, BrBG_r, twilight_shifted, Spectral_r, PRGn_r
                    ax=ax3)
    fig3.set_xticklabels(ax3.get_xticklabels(), rotation = 45, ha='right')
    #plt.xlabel('Vascular IDPs')
    #plt.ylabel('Cox model')
    #format_axes(fig)