import pandas as pd
import numpy as np
from statsmodels.formula.api import ols, logit
from datetime import datetime

DATE = datetime.now().strftime("%Y-%m-%d")

def read_diseases_csv_info(pheno_info_file):
    """_summary_

    Args:
        pheno_info_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    inf = pd.read_csv(pheno_info_file)
    #inf.drop(inf[(inf['name']=='vascular_heart_problems_00')].index, inplace=True)

    # Lists of phenotypes
    inf = inf[inf['name_LR'].notnull()]
    list_diseases = inf['name_LR'].values

    list_diseases_bin = inf.loc[inf['dtype']=='bin', 'name_LR'].values # binary phenotypes
    list_diseases_con = inf.loc[inf['dtype']=='con', 'name_LR'].values # continuous phenotypes
    list_diseases_cat = inf.loc[inf['dtype']=='cat', 'name_LR'].values # categorical phenotypes

    return inf, list_diseases, list_diseases_bin, list_diseases_con, list_diseases_cat


def info_subsplit_LR_cox(inf):
    """_summary_

    Args:
        inf (_type_): _description_

    Returns:
        _type_: _description_
    """
    inf_LR=inf[inf['main_supl']=='main']
    # inf_LR=inf[inf['name_LR'].notnull()]
    inf_cox=inf[inf['main_supl']=='hr']

    ##Take list for the old and new names
    list_diseases_LR = inf_LR['name_LR'].values
    list_diseases_LR_new = inf_LR['final_name'].values + ' '+ inf_LR['round_N_cases_LR_cox'].values

    list_diseases_cox = inf_cox['name_LR'].values
    list_diseases_cox_new = inf_cox['final_name'].values + ' '+ inf_cox['round_N_cases_LR_cox'].values

    return list_diseases_LR, list_diseases_LR_new, list_diseases_cox, list_diseases_cox_new


def info_subsplit_linear_log(inf):
    """_summary_

    Args:
        inf (_type_): _description_

    Returns:
        _type_: _description_
    """

    ## To spit in Linear and logistic regressions (note that categorical is not used)
    inf_LinearR_bin = inf[inf['dtype']=='con']
    inf_LinearR_bin = inf_LinearR_bin[inf_LinearR_bin['main_supl']=='main']
    list_diseases_LinearR = inf_LinearR_bin['name_LR'].values
    list_diseases_LinearR_new = inf_LinearR_bin['final_name'].values + ' '+ inf_LinearR_bin['round_N_cases_LR_cox'].values

    inf_LogisticR_bin = inf[inf['dtype']=='bin']
    inf_LogisticR_bin = inf_LogisticR_bin[inf_LogisticR_bin['main_supl']=='main']

    list_diseases_LogistR = inf_LogisticR_bin['name_LR'].values
    list_diseases_LogistR_new = inf_LogisticR_bin['final_name'].values  + ' '+ inf_LogisticR_bin['round_N_cases_LR_cox'].values

    inf_LogisticR_bin_con = inf[inf['dtype']=='bin_con']
    inf_LogisticR_bin_con = inf_LogisticR_bin_con[inf_LogisticR_bin_con['main_supl']=='hr'] 

    list_diseases_LogistR_both = inf_LogisticR_bin_con['name_LR'].values #+ inf_LogisticR_bin_con['name_cox'].values
    list_diseases_LogistR_both_new = inf_LogisticR_bin_con['final_name'].values  + ' '+ inf_LogisticR_bin_con['round_N_cases_LR_cox'].values

    return list_diseases_LinearR, list_diseases_LinearR_new, list_diseases_LogistR, list_diseases_LogistR_new, list_diseases_LogistR_both, list_diseases_LogistR_both_new


def filter_rename_col_index(df_, l_pehos_old, l_phenos_new, l_diseases_old, l_diseases_new, filtered_by):
    """_summary_

    Args:
        df_ (_type_): _description_
        l_pehos_old (_type_): _description_
        l_phenos_new (_type_): _description_
        l_diseases_old (_type_): _description_
        l_diseases_new (_type_): _description_
        filtered_by (_type_): _description_

    Returns:
        _type_: _description_
    """
    if filtered_by==True:
        df_ =df_.loc[l_pehos_old]
    else:
        df_ = df_.T
        df_ =df_.loc[l_diseases_old]
        df_ = df_.T

    df_.rename(index=dict(zip(l_pehos_old, l_phenos_new)), inplace=True)
    #print(df_)
    df_.rename(columns=dict(zip(l_diseases_old, l_diseases_new)), inplace=True)
    return df_


def read_diseases_files(diseases_file, pheno_info_file):
    """_summary_

    Args:
        diseases_file (_type_): _description_
        pheno_info_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_diseases=pd.read_csv(diseases_file, sep=',')
    inf, list_diseases, list_diseases_bin, list_diseases_con, list_diseases_cat = read_diseases_csv_info(pheno_info_file)
    return df_diseases, inf, list_diseases, list_diseases_bin, list_diseases_con, list_diseases_cat


def read_pheno(pheno_file):
    """_summary_

    Args:
        pheno_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_pheno=pd.read_csv(pheno_file, sep=',')

    ### To double-check: Replace -999 by np.nan
    df_pheno = df_pheno.replace(-999.00, np.nan)
    #print(df_pheno.isna().sum())

    ## Rename to eid to make the merge
    df_pheno.rename(columns={'Unnamed: 0': 'eid'}, inplace=True)

    return df_pheno


def merge_pheno_diseases(df_pheno, df_diseases):
    """_summary_

    Args:
        df_pheno (_type_): _description_
        df_diseases (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_pheno_dise=df_pheno.merge(df_diseases, how='left', on='eid')
    print('Len BEFORE merge: pheno size , diseases size: ', len(df_pheno), len(df_diseases))
    print('Len AFTER merge: ', len(df_pheno_dise))
    return df_pheno_dise


def filtrate_col(df_pheno_dise, list_phenos_diseases):
    """_summary_

    Args:
        df_pheno_dise (_type_): _description_
        list_phenos_diseases (_type_): _description_

    Returns:
        _type_: _description_
    """
    return df_pheno_dise[list_phenos_diseases]


def compute_pval_betas_LR(df_pheno_dise, inf, list_diseases, list_phenotypes):
    """_summary_

    Args:
        df_pheno_dise (_type_): _description_
        inf (_type_): _description_
        list_diseases (_type_): _description_
        list_phenotypes (_type_): _description_

    Returns:
        _type_: _description_
    """
    betas = pd.DataFrame(columns=list_diseases, index=list_phenotypes)
    log10p = pd.DataFrame(columns=list_diseases, index=list_phenotypes)

    for out in list_diseases:
        for reg in list_phenotypes:
            print(out, reg)
            ### checking the min and max values
            #print(df_pheno_dise[out].min(), df_pheno_dise[out].max())
            
            ## Logistic: to calculate odds ratios it would be simply e^beta (or np.exp(beta)),
            #beta being the estimate found in the logistic regression. 
            #And the SE would be odds-ratio times the found SE (np.exp(beta)*se)
            #(you can get the SE with results.bse the same you get betas with results.params)
            
            # OLS regression for categorical/ordinal and continuous outcomes
            if (inf.loc[inf['name_LR']==out, 'dtype'].values[0]=='cat') | (inf.loc[inf['name_LR']==out, 'dtype'].values[0]=='con'):
                model = ols(formula=out+'~'+reg, data=df_pheno_dise)
            # Logistic regression for binary outcomes
            elif inf.loc[inf['name_LR']==out, 'dtype'].values[0]=='bin':
                model = logit(formula=out+'~'+reg, data=df_pheno_dise)
                # results = model.fit(method='bfgs')
            elif inf.loc[inf['name_LR']==out, 'dtype'].values[0]=='bin_con':
                model = logit(formula=out+'~'+reg, data=df_pheno_dise)
            results = model.fit()
            betas.loc[reg, out] = results.params[reg]
            log10p.loc[reg, out] = -np.log10(results.pvalues[reg])
    
    betas = betas.astype('float64')

    return betas, log10p


def column_startwith_replace_nan_by_0(df_pheno_dise):
    """_summary_

    Args:
        df_pheno_dise (_type_): _description_

    Returns:
        _type_: _description_
    """
    for col in df_pheno_dise.columns:
        # Make variables that start with "age_" binary (0 in NaN, 1 if not NaN)
        if col.startswith('age_'):
            df_pheno_dise.loc[df_pheno_dise[col].notna(), col] = 1
            df_pheno_dise.loc[df_pheno_dise[col].isna(), col] = 0

        elif col.startswith('date_'):
            df_pheno_dise.loc[df_pheno_dise[col].notna(), col] = 1
            df_pheno_dise.loc[df_pheno_dise[col].isna(), col] = 0

        # Same for variables starting with "eye_" 
        elif col.startswith('eye_'):
            df_pheno_dise.loc[df_pheno_dise[col].notna(), col] = 1
            df_pheno_dise.loc[df_pheno_dise[col].isna(), col] = 0
    return df_pheno_dise


def std_contin_col(df_pheno_dise, list_diseases_con):
    """_summary_

    Args:
        df_pheno_dise (_type_): _description_
        list_diseases_con (_type_): _description_

    Returns:
        _type_: _description_
    """
    for var in list_diseases_con:
        mu = np.mean(df_pheno_dise[var])
        sig = np.std(df_pheno_dise[var])
        df_pheno_dise[var] = (df_pheno_dise[var]-mu)/sig
    return df_pheno_dise


def N_of_nans_and_nonans(df_,disease):
    """_summary_

    Args:
        df_ (_type_): _description_
        disease (_type_): _description_
    """
    print('- ', disease,': ', 
          len(df_[disease]) - df_[disease].isna().sum(),
          (df_[disease].isna().sum()/len(df_[disease])).round(4))#, '\n')


def col_to_numeric(df_pheno_dise):
    """_summary_

    Args:
        df_pheno_dise (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Get the data types of all columns in the DataFrame
    col_types = df_pheno_dise.dtypes

    # Convert columns that are not already numeric to numeric data type
    for col_name, col_type in col_types.items():
        if col_type not in ['int64', 'float64']: # Check if the column is not already numeric
            try:
                df_pheno_dise[col_name] = pd.to_numeric(df_pheno_dise[col_name])
            except ValueError: # Handle the case when the column cannot be converted to numeric
                print(f"Column '{col_name}' cannot be converted to numeric")

    # Print the summary information about the DataFrame
    print(df_pheno_dise.info())

    return df_pheno_dise


def cox_specific(betas, df_cox):
    """_summary_

    Args:
        betas (_type_): _description_
        df_cox (_type_): _description_

    Returns:
        _type_: _description_
    """
    ### add cox:
    df_cox = pd.DataFrame(df_cox, index = list(betas.index))
    ## Separate in pval and hazar ratio dfs:
    df_cox_pvalues = df_cox.loc[:,df_cox.columns.str.endswith('_pval')]
    df_cox_hazar_ratio = df_cox.loc[:,df_cox.columns.str.endswith('_hr')]
    ## Create column -log10(p) 
    df_cox_log10p = -np.log(df_cox_pvalues)

    return df_cox_log10p, df_cox_hazar_ratio


def save_betas_pval(betas, log10p, output_dir, What_type_phenotype):
    """_summary_

    Args:
        betas (_type_): _description_
        log10p (_type_): _description_
        output_dir (_type_): _description_
        What_type_phenotype (_type_): _description_
    """
    if What_type_phenotype == 'main':
        betas.to_csv(output_dir+'reg_betas_.csv')
        log10p.to_csv(output_dir+'reg_log10p_.csv')
    else:
        betas.to_csv(output_dir+'reg_betas_sup.csv')
        log10p.to_csv(output_dir+'reg_log10p_sup.csv')