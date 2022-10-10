#!/usr/bin/env python
# coding: utf-8

########### Diseases MLR and corr with phenotypes 

# Last modification: 09/08/2022

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from statsmodels.formula.api import ols, logit
from datetime import datetime
import sys

DATE = datetime.now().strftime("%Y-%m-%d")

ventile='ventile'+str(sys.argv[1])
What_type_phenotype=sys.argv[2]
output_dir = sys.argv[3]
diseases_file = sys.argv[3] + sys.argv[4] 
pheno_file =  sys.argv[5] + sys.argv[6]
#print('hiiiiiiiiiiiiii ',sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6] )

if What_type_phenotype == 'main':
    file_info_name='pheno_info_main.csv'
    
elif What_type_phenotype == 'suplementary':
    file_info_name='pheno_info_sup.csv'

else:
    print('Error, should be main or suplementary!')
    sys.stop()
    
pheno_info_file = '~/retina-phenotypes/complementary/disease_association/'+str(file_info_name)

display_info=True



if file_info_name=='pheno_info_sup.csv':
    list_phenotypes=list(sys.argv[7].split(","))
    list_phenotypes_new=list(sys.argv[9].split(","))
    
if file_info_name=='pheno_info_main.csv':
    list_phenotypes=list(sys.argv[8].split(","))
    list_phenotypes_new=list(sys.argv[10].split(","))


####################### 1 - Read diseases:
df_diseases=pd.read_csv(diseases_file, sep=',')


# #### Read diseases info

inf = pd.read_csv(pheno_info_file)
#inf.drop(inf[(inf['name']=='vascular_heart_problems_00')].index, inplace=True)


# Lists of phenotypes
list_diseases = inf['name'].values

list_diseases_bin = inf.loc[inf['dtype']=='bin', 'name'].values # binary phenotypes
list_diseases_con = inf.loc[inf['dtype']=='con', 'name'].values # continuous phenotypes
list_diseases_cat = inf.loc[inf['dtype']=='cat', 'name'].values # categorical phenotypes


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
plt.savefig(output_dir + str(DATE)+'_'+ ventile +'_MLRdiseases_hist.jpg', facecolor='white', bbox_inches='tight', pad_inches=0.1, dpi=150)


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


import seaborn as sns

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

if file_info_name=='pheno_info_sup.csv':

    df_pheno_dise['date_AD'] = pd.to_numeric(df_pheno_dise['date_AD']) 
    df_pheno_dise['date_death'] = pd.to_numeric(df_pheno_dise['date_death']) 
    #print(df_pheno_dise.info())


#########  Linear/logistic regression

#list_phenos_diseases, list_phenotypes, list(list_diseases)

betas = pd.DataFrame(columns=list_diseases, index=list_phenotypes)
log10p = pd.DataFrame(columns=list_diseases, index=list_phenotypes)

for out in list_diseases:
    for reg in list_phenotypes:
        #print(out, reg)
        ### checking the min and max values
        #print(df_pheno_dise[out].min(), df_pheno_dise[out].max())
        
        # OLS regression for categorical/ordinal and continuous outcomes
        if (inf.loc[inf['name']==out, 'dtype'].values[0]=='cat') | (inf.loc[inf['name']==out, 'dtype'].values[0]=='con'):
            model = ols(formula=out+'~'+reg, data=df_pheno_dise)
        # Logistic regression for binary outcomes
        elif inf.loc[inf['name']==out, 'dtype'].values[0]=='bin':
            model = logit(formula=out+'~'+reg, data=df_pheno_dise)
            # results = model.fit(method='bfgs')
        results = model.fit()
        betas.loc[reg, out] = results.params[reg]
        log10p.loc[reg, out] = -np.log10(results.pvalues[reg])


betas.to_csv(output_dir+'reg_betas_'+ventile+'.csv')
log10p.to_csv(output_dir+'reg_log10p_'+ventile+'.csv')


# Regression heatmaps
## NB: infinite -log10(p) are arbitrarily replaced by a fixed value ('inf_val') for visualisation
#inf_val = 310

betas = betas.astype('float64') # in case betas was coded as object type

## This colours by -log10(p) and annotates betas
## NOT VERY READABLE: DELETE??
#fig = plt.figure(figsize=(20, 16))
#sns.heatmap(log10p.replace(np.inf, inf_val), annot=betas.round(2), cmap='Blues', vmin=0, cbar_kws={'label': '-log10(p)'})
#plt.close()


##Change the name of the columns and index in beta and log10:
def rename_col_index(df_, l_diseases_old, l_diseases_new, l_pehos_old, l_phenos_new):
    df_.rename(columns=dict(zip(l_diseases_old, l_diseases_new)), inplace=True)
    df_.rename(index=dict(zip(l_pehos_old, l_phenos_new)), inplace=True)
    return df_


betas = rename_col_index(betas, list(inf['name']), list(inf['final_name']), list_phenotypes, list_phenotypes_new)
log10p = rename_col_index(log10p, list(inf['name']), list(inf['final_name']), list_phenotypes, list_phenotypes_new)


## This colours by beta and annotates Bonferroni-significant models with an asterisk
Bonf_thresh = -np.log10(0.05 / (log10p.shape[0] * log10p.shape[1]))


if file_info_name=='pheno_info_sup.csv':
    fig = plt.figure(figsize=(10, 10))
    imagen=sns.heatmap(betas, annot=(log10p>Bonf_thresh).replace({True:'*', False:''}), fmt="", annot_kws={'weight': 'bold'}, vmin=-abs(betas).max().max(), vmax=abs(betas).max().max(), cmap='seismic', cbar_kws={'label': 'Standardised \u03B2'})
    plt.savefig(output_dir+ str(DATE)+'_MLR_'+ventile+'_sup.jpg', facecolor='white', bbox_inches='tight', pad_inches=0.1, dpi=150)

else:
    fig = plt.figure(figsize=(7, 5))
    imagen=sns.heatmap(betas, annot=(log10p>Bonf_thresh).replace({True:'*', False:''}), fmt="", annot_kws={'weight': 'bold'}, vmin=-abs(betas).max().max(), vmax=abs(betas).max().max(), cmap='seismic', cbar_kws={'label': 'Standardised \u03B2'})
    plt.savefig(output_dir+ str(DATE)+'_MLR_'+ventile+'.jpg', facecolor='white', bbox_inches='tight', pad_inches=0.1, dpi=150)
#plt.close()


#### cases and controls

#for i in list_diseases:
#    print(i, df_pheno_dise[i].value_counts())
     



