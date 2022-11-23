#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
import sys,os

from scipy.stats import zscore

from matplotlib import pyplot as plt


# In[2]:


# RUN_PATH = "/NVME/decrypted/scratch/multitrait/UK_BIOBANK_PREPRINT/"
# PHENO_PATH = RUN_PATH+"participant_phenotype/"
# ID = "2022_11_23_covar_fix"
# CASE='qqnorm' #qqnorm or z
# MAIN_TRAITS='tau1_artery,tau1_vein,ratio_AV_DF,D_A_std,D_V_std,bifurcations,VD_orig_artery,VD_orig_vein,ratio_VD,mean_angle_taa,mean_angle_tva,eq_CRAE,eq_CRVE,ratio_CRAE_CRVE,medianDiameter_artery,medianDiameter_vein,ratio_AV_medianDiameter'.split(',')
# MAIN_LABELS='A tortuosity,V tortuosity,ratio tortuosity,A std diameter,V std diameter,bifurcations,A vascular density,V vascular density,ratio vascular density,A temporal angle,V temporal angle,A central retinal eq,V central retinal eq,ratio central retinal eq,A median diameter,V median diameter,ratio median diameter'.split(',')

RUN_PATH = sys.argv[1]
PHENO_PATH = RUN_PATH+"participant_phenotype/"
ID = sys.argv[2]
OUTID = ID+"_corrected"
CASE=sys.argv[3] #qqnorm or z
MAIN_TRAITS=sys.argv[4].split(',')
MAIN_LABELS=sys.argv[5].split(',')

OUTID = ID+"_"+CASE+"_corrected"

traitsfile = PHENO_PATH+ID+'_raw.csv'
covarsfile = RUN_PATH+'diseases_cov/'+ID+"_diseases_cov.csv"


# In[3]:


# rbINT

# for the following code block, the corresponding MIT License

#The MIT License (MIT)
#
#Copyright (c) 2016 Edward Mountjoy
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

def rank_INT(series, c=3.0/8, stochastic=False):
    """ Perform rank-based inverse normal transformation on pandas series.
        If stochastic is True ties are given rank randomly, otherwise ties will
        share the same value. NaN values are ignored.
        Args:
            param1 (pandas.Series):   Series of values to transform
            param2 (Optional[float]): Constand parameter (Bloms constant)
            param3 (Optional[bool]):  Whether to randomise rank of ties
        
        Returns:
            pandas.Series
    """

    # Check input
    assert(isinstance(series, pd.Series))
    assert(isinstance(c, float))
    assert(isinstance(stochastic, bool))
    
    # Print input name
    print(series.name)

    # Set seed
    np.random.seed(123)

    # Take original series indexes
    orig_idx = series.index

    # Drop NaNs
    series = series.loc[~pd.isnull(series)]

    # Get ranks
    if stochastic == True:
        # Shuffle by index
        series = series.loc[np.random.permutation(series.index)]
        # Get rank, ties are determined by their position in the series (hence
        # why we randomised the series)
        rank = ss.rankdata(series, method="ordinal")
    else:
        # Get rank, ties are averaged
        rank = ss.rankdata(series, method="average")

    # Convert numpy array back to series
    rank = pd.Series(rank, index=series.index)

    # Convert rank to normal distribution
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))

    return transformed.reindex(orig_idx, fill_value=np.NaN)

def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return ss.norm.ppf(x)


# In[4]:


traits = pd.read_csv(traitsfile, index_col=0)


# In[5]:


# apply rank-based inverse normal transform

if CASE=='qqnorm':

    from multiprocessing import Pool
    from scipy import stats as ss

    #creating rank-based INT phenofile
    def apply_rank_INT(col):
        return rank_INT(traits[col])

    cols=traits.columns
    pool=Pool()
    out=list(pool.map(apply_rank_INT, cols))
    pool.close()
    for idx,i in enumerate(cols):
        traits[i] = out[idx]

    # z-scoring, to enforce mean=0, std=1
    traits = traits.apply(zscore, nan_policy='omit')
    
else: # raw
    pass


# In[6]:


# plot and save raw histograms

traits_hist = traits.copy()

cols = []
for idx,i in enumerate(traits_hist.columns):
    cols.append(traits_hist.columns[idx] + " N=" + str(len(traits_hist) - traits_hist[i].isna().sum()))
    
traits_hist.columns = cols

traits_hist.hist(figsize=(40,40), bins=100);
plt.tight_layout()
plt.savefig(PHENO_PATH+ID+"_"+CASE+"_distribution.pdf")


# In[7]:


# load covar

covar = pd.read_csv(covarsfile, index_col='eid', parse_dates=['date_AD'])
instance = covar['instance']

covar['age_center_both_2'] = covar['age_center_both'].pow(2)
covar['sex_by_age'] = covar['age_center_both'] * covar['sex']
covar['sex_by_age_2'] = covar['age_center_both_2'] * covar['sex']

# covars['age2'] = covars['age'].pow(2)
# # covars['age3'] = covars['age'].pow(3)

covar['PC1_squared'] = covar['PC1'].pow(2)
# covars['PC1_cube'] = covars['PC1'].pow(3)

covar['PC5_squared'] = covar['PC5'].pow(2)
# covars['PC5_cube'] = covars['PC5'].pow(3)

# covars['spherical_power'] = covars[['spherical_power_R','spherical_power_L']].mean(axis=1)
# covars['cylindrical_power'] = covars[['cylindrical_power_R','cylindrical_power_L']].mean(axis=1)

# covars['spherical_power_2'] = covars['spherical_power'].pow(2)
# # covars['spherical_power_3'] = covars['spherical_power'].pow(3)

# covars['cylindrical_power_2'] = covars['cylindrical_power'].pow(2)
# covars['cylindrical_power_3'] = covars['cylindrical_power'].pow(3)


# In[8]:


len(traits), len(covar)


# In[9]:


# filter covar

# usecovars = ['sex','age_center','age_center_2','spherical_power','spherical_power_2','cylindrical_power','cylindrical_power_2','PC1', 'PC1_squared']
# usecovars = usecovars + ["PC"+str(i) for i in range(2,5)]
# usecovars = usecovars + ['PC5', 'PC5_squared']
# usecovars = usecovars + ["PC"+str(i) for i in range(6,41)] # dropping PC21-40, none are individually significant

usecovars = ['sex','age_center_both','age_center_both_2','sex_by_age','sex_by_age_2','spherical_power_both','spherical_power_both_2','cylindrical_power_both','cylindrical_power_both_2','instance','PC1']
usecovars = usecovars + ["PC"+str(i) for i in range(2,5)]
usecovars = usecovars + ['PC5']
usecovars = usecovars + ["PC"+str(i) for i in range(6,21)] # dropping PC21-40, none are individually significant

step1_covars = usecovars # we now perform only one step
# step1_covars = ['sex','age_center','age_center_2','spherical_power','spherical_power_2','cylindrical_power','cylindrical_power_2'] + ["PC"+str(i) for i in range(1,11)]

covar_full = covar.copy()
covar = covar[usecovars]


# In[10]:


#z-scoring of raw traits, and covar

traits = traits.apply(zscore, nan_policy='omit')
# traits.dropna(inplace=True)

covar_z = covar.apply(zscore, nan_policy='omit')

# participants_common = [i for i in covar_z.index if i in traits.index]
# covar = covar_z.loc[participants_common]

# covar_z.dropna(inplace=True) # TEMPORARY, to remove missing spherical values


# In[11]:


# covar clustermap
h = sns.clustermap(data=covar_z.corr(), cmap='viridis')
dgram=h.dendrogram_col.dendrogram
D = np.array(dgram['dcoord'])
I = np.array(dgram['icoord'])
plt.savefig(PHENO_PATH+ID+"_covar_distribution.pdf")


# In[12]:


# adding categorical variables

usecategorical = ['assessment_centre_both','genotype_measurement_batch','instance']

covar_z['assessment_centre_both'] = covar_full['assessment_centre_both']
covar_z['genotype_measurement_batch'] = covar_full['genotype_measurement_batch']
covar_z['instance'] = covar_full['instance']
covar_z.loc[covar_z['genotype_measurement_batch']<0, 'genotype_measurement_batch'] = 0
covar_z.loc[covar_z['genotype_measurement_batch']>0, 'genotype_measurement_batch'] = 1


# In[13]:


# equalize participants

index_intersect = traits.index.intersection(covar_z.index)
print(len(index_intersect))
traits = traits.loc[index_intersect]
covar = covar.loc[index_intersect]
covar_z = covar_z.loc[index_intersect]
instance = instance.loc[index_intersect]


# In[14]:


# PCA (not used finally, but I'm still removing some nans)

from sklearn.decomposition import PCA

npcs = covar_z.shape[1] #15
pca = PCA(n_components=npcs)

# temporary fix of some covariates being na
covar_z_nona = covar_z.dropna()
# traits_nona = traits.loc[covar_z_nona.index]
traits_nona = traits.loc[covar_z_nona.index]

pcs = pd.DataFrame(pca.fit_transform(covar_z_nona), index=covar_z_nona.index, columns=['covPC'+str(i) for i in range(1,npcs+1)])


# In[15]:


# check size match

len(traits),len(covar_z), len(pcs), len(covar_z_nona), len(traits_nona)


# # Regressing out covariate effects

# In[16]:


# def mlr2(X, y):
    
#     import pandas as pd
#     import numpy as np
#     from sklearn import datasets, linear_model
#     from sklearn.linear_model import LinearRegression
#     import statsmodels.api as sm
#     from scipy import stats

#     nani = np.argwhere(~np.isnan(y.values))
#     nani = [i[0] for i in nani]
    
# #     print(X.shape, len(y), len(nani))
    
#     X = X.iloc[nani]
#     y = list(y.iloc[nani])

# #     print(X.shape, len(y))
    
#     X2 = sm.add_constant(X)
#     est = sm.OLS(y, X2)
#     est2 = est.fit()
# #     print(est2.summary())
    
#     return est2


# In[17]:


# Visualization model
# * contains only linear covariates
# * adjusts only main traits

from statsmodels.formula.api import ols

viz_traits_corrected = traits_nona[MAIN_TRAITS].copy()

for i,trait in enumerate(MAIN_TRAITS):


    regdf = covar_z_nona.copy()
    regdf['response'] = traits_nona[trait]
    
    print(trait)
    regdf['assessment_centre_both'].replace(11024.0, 0, inplace=True)
    regdf.dropna(inplace=True)
    # print(len(regdf))
    
    # potential two-step
    # fit = ols('response ~ age_center_both', data=regdf).fit()
    # regdf['response'] = regdf['response'] - fit.params[1] * regdf['age_center_both']

    fit = ols('response ~  sex + age_center_both + sex_by_age + spherical_power_both + cylindrical_power_both + PC1 + PC2 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20 + C(assessment_centre_both) + C(genotype_measurement_batch) + C(instance)', data=regdf).fit() 
    # fit = ols('response ~ C(assessment_centre_both) + C(genotype_measurement_batch) + age_center_both + age_center_both_2 + sex + sex_by_age + sex_by_age_2 + spherical_power_both + spherical_power_both_2 + cylindrical_power_both + cylindrical_power_both_2 + PC1 + PC2 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20', data=regdf).fit() 
    # fit = ols('response ~ age_center_both + sex + sex_by_age + spherical_power_both + cylindrical_power_both + PC1 + PC2 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20', data=regdf).fit() 

    
    viz_traits_corrected[trait] = fit.resid

    if i == 0:
        covar_pval = pd.DataFrame(index = MAIN_TRAITS, columns=fit.pvalues.index[1:], dtype=float)
        covar_pval.loc[trait] = fit.pvalues[1:].values
        covar_coefs = pd.DataFrame(index = MAIN_TRAITS, columns=fit.pvalues.index[1:], dtype=float)
        covar_coefs.loc[trait] = fit.params[1:].values
    else:
        covar_pval.loc[trait] = fit.pvalues[1:].values
        covar_coefs.loc[trait] = fit.params[1:].values
        
covar_pval[covar_pval == 0] = 8.984227e-308


# In[18]:


# significance df

significant = covar_pval < 0.05 #/ len(covar_pval)

asterisks = significant.applymap(str)
asterisks = asterisks.replace('False', "")
asterisks = asterisks.replace('True', "*")


# In[19]:


label_dict = {'C(assessment_centre_both)[T.11014.0]': 'Sheffield AC', 'C(assessment_centre_both)[T.11016.0]': 'Liverpool AC', 'C(assessment_centre_both)[T.11018.0]': 'Hounslow AC', 'C(assessment_centre_both)[T.11020.0]': 'Croydon AC', 'C(assessment_centre_both)[T.11021.0]': 'Birmingham AC',  'C(assessment_centre_both)[T.11022.0]': 'Swansea AC', 'C(genotype_measurement_batch)[T.1]':'SNP array', 'C(instance)[T.1.0]':"instance", 'age_center_both':'age', 'sex_by_age':'sex * age', 'spherical_power_both':'spherical power', 'cylindrical_power_both':'cylindrical power'}

replaced_list = [x if x not in label_dict else label_dict[x] for x in covar_coefs.columns]
covar_coefs.columns=replaced_list


# In[20]:


if CASE == 'z':

    cmap = sns.color_palette("viridis", as_cmap=True)

    plt.figure(figsize=(14,6))
    sns.heatmap(covar_coefs.loc[MAIN_TRAITS], annot=asterisks.loc[MAIN_TRAITS], fmt='', cmap=cmap, yticklabels=MAIN_LABELS, cbar_kws={"aspect": 40, 'label':"Standardized effect size"});
    plt.title("Covariate effects on main traits",fontweight='bold')
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')
    plt.savefig(RUN_PATH+"/figures/covariate_effects_on_main_traits_" + ID + ".pdf")

else:
    print("Skipped covariate effect heatmap")


# In[21]:


# significance df

significant = covar_pval < 0.5

asterisks = significant.applymap(str)
asterisks = asterisks.replace('False', "")
asterisks = asterisks.replace('True', "*")

cmap = sns.color_palette("viridis", as_cmap=True)

plt.figure(figsize=(14,10))
sns.heatmap(covar_coefs.loc[MAIN_TRAITS], annot=asterisks.loc[MAIN_TRAITS], fmt='', cmap=cmap, yticklabels=MAIN_LABELS);
plt.title("Better to remove than keep",fontweight='bold')
plt.tight_layout()
plt.savefig(PHENO_PATH+ID+"_"+CASE+"_covar_effects_larger_than_05.pdf")


# In[ ]:





# In[ ]:





# In[22]:


# Visualization model - check correction

from statsmodels.formula.api import ols

for i,trait in enumerate(MAIN_TRAITS):


    regdf = covar_z_nona.copy()
    regdf['response'] = viz_traits_corrected[trait]
    
    print(trait)
    regdf['assessment_centre_both'].replace(11024.0, 0, inplace=True)
    regdf.dropna(inplace=True)
    # print(len(regdf))
    
    # potential two-step
    # fit = ols('response ~ age_center_both', data=regdf).fit()
    # regdf['response'] = regdf['response'] - fit.params[1] * regdf['age_center_both']

    fit = ols('response ~ age_center_both + sex + sex_by_age + spherical_power_both + cylindrical_power_both + PC1 + PC2 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20 + C(assessment_centre_both) + C(genotype_measurement_batch) + C(instance)', data=regdf).fit() 
    # fit = ols('response ~ C(assessment_centre_both) + C(genotype_measurement_batch) + age_center_both + age_center_both_2 + sex + sex_by_age + sex_by_age_2 + spherical_power_both + spherical_power_both_2 + cylindrical_power_both + cylindrical_power_both_2 + PC1 + PC2 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20', data=regdf).fit() 
    # fit = ols('response ~ age_center_both + sex + sex_by_age + spherical_power_both + cylindrical_power_both + PC1 + PC2 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20', data=regdf).fit() 

    if i == 0:
        covar_pval = pd.DataFrame(index = MAIN_TRAITS, columns=fit.pvalues.index[1:], dtype=float)
        covar_pval.loc[trait] = fit.pvalues[1:].values
        covar_coefs = pd.DataFrame(index = MAIN_TRAITS, columns=fit.pvalues.index[1:], dtype=float)
        covar_coefs.loc[trait] = fit.params[1:].values
    else:
        covar_pval.loc[trait] = fit.pvalues[1:].values
        covar_coefs.loc[trait] = fit.params[1:].values
        
covar_pval[covar_pval == 0] = 8.984227e-308


# In[23]:


# significance df

significant = covar_pval < 0.05 #/ len(covar_pval)

asterisks = significant.applymap(str)
asterisks = asterisks.replace('False', "")
asterisks = asterisks.replace('True', "*")

cmap = sns.color_palette("viridis", as_cmap=True)

plt.figure(figsize=(14,8))
sns.heatmap(covar_coefs.loc[MAIN_TRAITS], annot=asterisks.loc[MAIN_TRAITS], fmt='', cmap=cmap, yticklabels=MAIN_LABELS, cbar_kws={"shrink": .5}, vmin=-0.1,vmax=0.1);
plt.title("Covariate effects on main traits",fontweight='bold')
plt.tight_layout()
plt.xticks(rotation=45, ha='right')


# In[24]:


# Full model
# * contains squared covariates

from statsmodels.formula.api import ols

traits_res = traits_nona.copy()

for i,trait in enumerate(traits_res.columns):

    regdf = covar_z_nona.copy()
    regdf['response'] = traits_nona[trait]
    
    print(trait)
    
    regdf['assessment_centre_both'].replace(11024.0, 0, inplace=True)
    regdf.dropna(inplace=True)
    
    fit = ols('response ~ C(assessment_centre_both) + C(genotype_measurement_batch) + C(instance) + age_center_both + age_center_both_2 + sex + sex_by_age + sex_by_age_2 + spherical_power_both + spherical_power_both_2 + cylindrical_power_both + cylindrical_power_both_2 + PC1 + PC2 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20', data=regdf).fit() 
    fit = ols('response ~ age_center_both + age_center_both_2 + sex + sex_by_age + sex_by_age_2 + spherical_power_both + spherical_power_both_2 + cylindrical_power_both + cylindrical_power_both_2 + PC1 + PC2 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20', data=regdf).fit() 

    
    traits_res[trait] = fit.resid

    if i == 0:
        covar_pval = pd.DataFrame(index = MAIN_TRAITS, columns=fit.pvalues.index[1:], dtype=float)
        covar_pval.loc[trait] = fit.pvalues[1:].values
        covar_coefs = pd.DataFrame(index = MAIN_TRAITS, columns=fit.pvalues.index[1:], dtype=float)
        covar_coefs.loc[trait] = fit.params[1:].values
    else:
        covar_pval.loc[trait] = fit.pvalues[1:].values
        covar_coefs.loc[trait] = fit.params[1:].values
        
covar_pval[covar_pval == 0] = 8.984227e-308


# In[25]:


# Check residuals have no more associations

from statsmodels.formula.api import ols


for i,trait in enumerate(traits_res.columns):

    regdf = covar_z_nona.copy()
    regdf['response'] = traits_res[trait]
    
    print(trait)
    
    regdf['assessment_centre_both'].replace(11024.0, 0, inplace=True)
    regdf.dropna(inplace=True)
    
    fit = ols('response ~ C(assessment_centre_both) + C(genotype_measurement_batch) + C(instance) + age_center_both + age_center_both_2 + sex + sex_by_age + sex_by_age_2 + spherical_power_both + spherical_power_both_2 + cylindrical_power_both + cylindrical_power_both_2 + PC1 + PC2 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20', data=regdf).fit() 
    fit = ols('response ~ age_center_both + age_center_both_2 + sex + sex_by_age + sex_by_age_2 + spherical_power_both + spherical_power_both_2 + cylindrical_power_both + cylindrical_power_both_2 + PC1 + PC2 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20', data=regdf).fit() 

    
    if i == 0:
        covar_pval = pd.DataFrame(index = MAIN_TRAITS, columns=fit.pvalues.index[1:], dtype=float)
        covar_pval.loc[trait] = fit.pvalues[1:].values
        covar_coefs = pd.DataFrame(index = MAIN_TRAITS, columns=fit.pvalues.index[1:], dtype=float)
        covar_coefs.loc[trait] = fit.params[1:].values
    else:
        covar_pval.loc[trait] = fit.pvalues[1:].values
        covar_coefs.loc[trait] = fit.params[1:].values
        
covar_pval[covar_pval == 0] = 8.984227e-308


# In[26]:


# significance df

significant = covar_pval < 0.05 #/ len(covar_pval)

asterisks = significant.applymap(str)
asterisks = asterisks.replace('False', "")
asterisks = asterisks.replace('True', "*")

cmap = sns.color_palette("viridis", as_cmap=True)

plt.figure(figsize=(14,40))
sns.heatmap(covar_coefs, annot=asterisks, fmt='', cmap=cmap, cbar_kws={"shrink": .5}, vmin=-0.1,vmax=0.1);
plt.title("Effects on residuals (control)", fontweight='bold')
plt.tight_layout()
plt.xticks(rotation=45, ha='right')

plt.savefig(PHENO_PATH+ID+"_"+CASE+"_effects_on_residuals.pdf")


# In[28]:


# write corrected, plot corrected clustermap in case raw, not qqnorm

if CASE == 'qqnorm':
    tmp = pd.read_csv(traitsfile, index_col=0)
    out = pd.DataFrame(index=tmp.index, columns=tmp.columns)
    out.update(traits_res)
    out = out.astype(str)
    out = out.replace('nan', '-999')
    out.to_csv(PHENO_PATH+ID+"_qqnorm.csv", index=False, sep=" ")
    
    # storing covars for GWAS
    
    # dummy coding of assessment center categorical varaibles for bgenie
    covar_z_out = covar_z.copy()
    covar_z_out['assessment_centre_both'].replace(11024.0, 0, inplace=True)
    
    for ac in ['11014.0','11016.0','11018.0', '11020.0', '11021.0', '11022.0']: #'11024.0' (most common one) set as reference
        covar_z_out['11024.0-'+ac] = [1  if str(i)==ac else 0 for i in covar_z_out['assessment_centre_both']]    
    
    covar_z_out.drop('assessment_centre_both', axis=1, inplace=True)
    
    covar_z_out = covar_z_out.astype(str)
    covar_z_out.replace('nan', '-999', inplace=True)
    covar_z_out.to_csv(PHENO_PATH+ID+"_covar.csv", index=False, sep=" ")

    
else:
    traits_res.to_csv(PHENO_PATH+OUTID+".csv")
    
    h = sns.clustermap(data=traits_res.corr(), figsize=(40,40), cmap='viridis')
    dgram=h.dendrogram_col.dendrogram
    D = np.array(dgram['dcoord'])
    I = np.array(dgram['icoord'])
    plt.savefig(PHENO_PATH+OUTID+"_clustermap.pdf")

