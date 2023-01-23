#!/usr/bin/env python
# coding: utf-8

########### Diseases MLR and corr with phenotypes 

# Last modification: 22/11/2022

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from statsmodels.formula.api import ols, logit
from datetime import datetime
import sys
import seaborn as sns


DATE = datetime.now().strftime("%Y-%m-%d")

What_type_phenotype=sys.argv[1]
output_dir = sys.argv[2]
diseases_file = sys.argv[2] + sys.argv[3] 
pheno_file =  sys.argv[4] + sys.argv[5]

file_info_name='pheno_diseases_info.csv'
pheno_info_file = '~/retina-phenotypes/complementary/'+str(file_info_name)

display_info=True

if What_type_phenotype == 'main':
    list_phenotypes=list(sys.argv[6].split(","))
    list_phenotypes_new=list(sys.argv[8].split(","))
    
elif What_type_phenotype == 'suplementary':
    list_phenotypes=list(sys.argv[7].split(","))
    list_phenotypes_new=list(sys.argv[9].split(","))

else:
    print('Error, should be main or suplementary!')
    #sys.stop()


#print(list_phenotypes, list_phenotypes_new)
####################### 1 - Read diseases:
df_diseases=pd.read_csv(diseases_file, sep=',')

# #### Read diseases info

inf = pd.read_csv(pheno_info_file)
#inf.drop(inf[(inf['name']=='vascular_heart_problems_00')].index, inplace=True)


# Lists of phenotypes
inf = inf[inf['name_LR'].notnull()]
list_diseases = inf['name_LR'].values

list_diseases_bin = inf.loc[inf['dtype']=='bin', 'name_LR'].values 
print(list_diseases_bin)
print(len(list_diseases_bin))
print(inf.loc[inf['dtype']=='bin_con', 'name_LR'].values) # binary phenotypes

list_diseases_con = inf.loc[inf['dtype']=='con', 'name_LR'].values # continuous phenotypes
list_diseases_cat = inf.loc[inf['dtype']=='cat', 'name_LR'].values # categorical phenotypes

print(list_diseases_con)
# ##### Number of cases and controls per disease


def N_of_nans_and_nonans(df_,disease):
    print('- ', disease,': ', 
          len(df_[disease]) - df_[disease].isna().sum(),
          (df_[disease].isna().sum()/len(df_[disease])).round(4))#, '\n')
                         
if display_info==True:
    print('Number of nans and ratio nans/no_nans:')
    for disease_name in list_diseases:
        N_of_nans_and_nonans(df_diseases,disease_name)
        #plt.hist(df_diseases[disease_name])
        #plt.title(disease_name)
        #plt.show()
                         
df_diseases_red = df_diseases[list_diseases]
                         
fig = plt.figure(figsize = (12,10))
ax = fig.gca()
df_diseases_red.hist(ax = ax) 
plt.savefig(output_dir + str(DATE)+'_MLRdiseases_hist.jpg', facecolor='white', bbox_inches='tight', pad_inches=0.1, dpi=150)


####################### 2 - Phenotypes:
### Read phenos
df_pheno=pd.read_csv(pheno_file, sep=',')

### Just to double-check: Replace -999 by np.nan
df_pheno = df_pheno.replace(-999.00, np.nan)
#print(df_pheno.isna().sum())

## Rename to eid to make the merge
df_pheno.rename(columns={'Unnamed: 0': 'eid'}, inplace=True)

####################### 3 - Merge diseases and phenotpyes
### Delete nans
#df_phenotypes=df_pheno.dropna()
#print('Delete subjects with nans, len before and after :',len(df_pheno), len(df_phenotypes))

df_pheno_dise=df_pheno.merge(df_diseases, how='left', on='eid')
print('Len BEFORE merge: pheno size , diseases size: ', len(df_pheno), len(df_diseases))
print('Len AFTER merge: ', len(df_pheno_dise))
#print(df_pheno_dise.columns)


# ## Filtrate only by the phenotypes and diseases of interest

list_phenos_diseases = list_phenotypes+list(list_diseases)

df_pheno_dise=df_pheno_dise[list_phenos_diseases]



####################### 4 - Correlation

def corr(x, y, **kwargs):
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    # Add the label to the plot
    ax = plt.pyplot.gca()
    ax.annotate(label, xy=(0.2, 0.95), size=20, xycoords=ax.transAxes)



matrix_total = df_pheno_dise.corr().round(1)
plt.figure(figsize=(10, 10), dpi = 900) 
sns.heatmap(matrix_total, annot=True, vmax=1, vmin=-1, center=0, annot_kws={'size': 3}, cmap='vlag')

####################### 5 - MLR

### Double check the numbers:
#print(df_pheno_dise_cov['age_cataract_00'].value_counts(), df_pheno_dise_cov['age_macular_deg_diag'].value_counts())


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


##### Standardise continuous diseasesa and continuous phenotypes if not z-scored

for var in list_diseases_con:
    mu = np.mean(df_pheno_dise[var])
    sig = np.std(df_pheno_dise[var])
    df_pheno_dise[var] = (df_pheno_dise[var]-mu)/sig


# ##### Check the type of variables and convert objects to numeric:

### convert to type numeric the columns that are not
df_pheno_dise['date_reported_atherosclerosis'] = pd.to_numeric(df_pheno_dise['date_reported_atherosclerosis'])
df_pheno_dise['date_disorders_arteries_arterioles'] = pd.to_numeric(df_pheno_dise['date_disorders_arteries_arterioles'])

#if file_info_name=='pheno_info_sup.csv':
df_pheno_dise['date_AD'] = pd.to_numeric(df_pheno_dise['date_AD']) 
#df_pheno_dise['date_death'] = pd.to_numeric(df_pheno_dise['date_death']) 
print(df_pheno_dise.info())


#########  Linear/logistic regression

#list_phenos_diseases, list_phenotypes, list(list_diseases)

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

if What_type_phenotype == 'main':
    betas.to_csv(output_dir+'reg_betas_.csv')
    log10p.to_csv(output_dir+'reg_log10p_.csv')

else:
    betas.to_csv(output_dir+'reg_betas_sup.csv')
    log10p.to_csv(output_dir+'reg_log10p_sup.csv')

# Regression heatmaps
## NB: infinite -log10(p) are arbitrarily replaced by a fixed value ('inf_val') for visualisation
#inf_val = 310

# betas = betas.astype('float64') # in case betas was coded as object type


# ##Change the name of the columns and index in beta and log10:
# def rename_col_index(df_, l_diseases_old, l_diseases_new, l_pehos_old, l_phenos_new):
#     #df_.columns = l_diseases_new
#     df_.rename(columns=dict(zip(l_diseases_old, l_diseases_new)), inplace=True)
#     df_.rename(index=dict(zip(l_pehos_old, l_phenos_new)), inplace=True)
#     return df_

# print(list(inf['name']))
# print(list(inf['final_name']))
# betas = rename_col_index(betas, list(inf['name']), list(inf['final_name']), list_phenotypes, list_phenotypes_new)
# log10p = rename_col_index(log10p, list(inf['name']), list(inf['final_name']), list_phenotypes, list_phenotypes_new)


# ## This colours by beta and annotates Bonferroni-significant models with an asterisk
# Bonf_thresh = -np.log10(0.05 / (log10p.shape[0] * log10p.shape[1]))
# Bonf_thresh2 = -np.log10(0.001 / (log10p.shape[0] * log10p.shape[1]))

# log10p_copy = log10p.copy()
# log10p_copy2 = log10p.copy()
# log10p_copy3 = log10p.copy()

# log10p_copy= (log10p_copy>Bonf_thresh).replace({True:'*', False:''})
# log10p_copy2= (log10p_copy2>Bonf_thresh2).replace({True:'*', False:''})
# log10p_copy3 =log10p_copy+log10p_copy2


# if file_info_name=='pheno_info_sup.csv':
#     fig = plt.figure(figsize=(13, 10))
#     ax2 = plt.axes()
#     ax2.yaxis.set_ticks_position('right')
#     ax=sns.heatmap(betas, 
#                 annot=log10p_copy3, #(log10p>Bonf_thresh).replace({True:'*', False:''}), 
#                 cbar=False,
#                 fmt="", annot_kws={'weight': 'bold'}, 
#                 vmin=-abs(betas).max().max(), 
#                 vmax=abs(betas).max().max(), 
#                 cmap='viridis', cbar_kws={'label': 'Standardised \u03B2'})
#     ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)  
#     plt.savefig(output_dir+ str(DATE)+'_MLR_sup.jpg', facecolor='white', bbox_inches='tight', pad_inches=0.1, dpi=150)

# else:
#     fig = plt.figure(figsize=(7, 6))
#     ax2 = plt.axes()
#     ax2.yaxis.set_ticks_position('right')
#     ax=sns.heatmap(betas, 
#                 annot=log10p_copy3, #(log10p>Bonf_thresh).replace({True:'*', False:''}), 
#                 cbar=False,
#                 fmt="", annot_kws={'weight': 'bold'}, 
#                 vmin=-abs(betas).max().max(), 
#                 vmax=abs(betas).max().max(), 
#                 cmap='viridis', cbar_kws={'label': 'Standardised \u03B2'})
#     ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
#     plt.savefig(output_dir+ str(DATE)+'_MLR_.jpg', facecolor='white', bbox_inches='tight', pad_inches=0.1, dpi=150)



#### cases and controls

# list_value=[]
# for i in log10p_copy3.columns:
#     #print(i, df_pheno_dise[i].value_counts())
#     data={
#         'i': i,
#         'value_counts': df_pheno_dise[i].value_counts()
#     }
#     list_value.append(data)
# df_count_val = pd.DataFrame(list_value)
# print(df_count_val)
# df_count_val.to_csv(output_dir+ str(DATE)+'_N_CASES_MLR_.csv', sep=',', index=False)

