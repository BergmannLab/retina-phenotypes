#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
import sys,os

from scipy.stats import zscore

from matplotlib import pyplot as plt


# In[38]:


# RUN_PATH = "/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/"
# PHENO_PATH = RUN_PATH+"participant_phenotype/"
# ID = "2022_07_08_ventile2"
# OUTID = ID+"_corrected.two.step.FINAL"
# CASE='qqnorm' #qqnorm or raw

RUN_PATH = sys.argv[1]
PHENO_PATH = RUN_PATH+"participant_phenotype/"
ID = sys.argv[2]
OUTID = ID+"_corrected.two.step.FINAL"
CASE=sys.argv[3] #qqnorm or raw

traitsfile = PHENO_PATH+ID+'_raw.csv'
covarsfile = RUN_PATH+'diseases_cov/'+ID+"_diseases_cov.csv"


# In[39]:


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


# In[40]:


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


# In[41]:


# load covar

covar = pd.read_csv(covarsfile, index_col=0, parse_dates=['date_AD'])
instance = covar['instance']
        
# covars['age'] = age_center
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


# In[42]:


len(traits), len(covar)


# In[43]:


traits.index[0:10]


# In[46]:


traits.isna().sum().min()


# In[44]:


covar.index[0:10]


# In[8]:


# filter covar

# usecovars = ['sex','age_center','age_center_2','spherical_power','spherical_power_2','cylindrical_power','cylindrical_power_2','PC1', 'PC1_squared']
# usecovars = usecovars + ["PC"+str(i) for i in range(2,5)]
# usecovars = usecovars + ['PC5', 'PC5_squared']
# usecovars = usecovars + ["PC"+str(i) for i in range(6,41)] # dropping PC21-40, none are individually significant

usecovars = ['sex','age_center','age_center_2','spherical_power','spherical_power_2','cylindrical_power','cylindrical_power_2','PC1']
usecovars = usecovars + ["PC"+str(i) for i in range(2,5)]
usecovars = usecovars + ['PC5']
usecovars = usecovars + ["PC"+str(i) for i in range(6,21)] # dropping PC21-40, none are individually significant

step1_covars = ['sex','age_center','age_center_2','spherical_power','spherical_power_2','cylindrical_power','cylindrical_power_2'] + ["PC"+str(i) for i in range(1,11)]

covar = covar[usecovars]


# In[9]:


#z-scoring of raw traits, and covar

traits = traits.apply(zscore, nan_policy='omit')
# traits.dropna(inplace=True)

covar_z = covar.apply(zscore, nan_policy='omit')

# participants_common = [i for i in covar_z.index if i in traits.index]
# covar = covar_z.loc[participants_common]

# covar_z.dropna(inplace=True) # TEMPORARY, to remove missing spherical values


# In[10]:


# covar clustermap
h = sns.clustermap(data=covar_z.corr(), cmap='viridis')
dgram=h.dendrogram_col.dendrogram
D = np.array(dgram['dcoord'])
I = np.array(dgram['icoord'])
plt.savefig(PHENO_PATH+ID+"_correction_covar_distribution.pdf")


# In[11]:


# equalize participants

index_intersect = traits.index.intersection(covar_z.index)
#print(len(index_intersect))
traits = traits.loc[index_intersect]
covar = covar.loc[index_intersect]
covar_z = covar_z.loc[index_intersect]
instance = instance.loc[index_intersect]


# In[12]:


# PCA

from sklearn.decomposition import PCA

npcs = covar_z.shape[1] #15
pca = PCA(n_components=npcs)

# temporary fix of some covariates being na
covar_z_nona = covar_z.dropna()
# traits_nona = traits.loc[covar_z_nona.index]
traits_nona = traits.loc[covar_z_nona.index]

pcs = pd.DataFrame(pca.fit_transform(covar_z_nona), index=covar_z_nona.index, columns=['covPC'+str(i) for i in range(1,npcs+1)])


# In[13]:


# check size match

print(len(traits),len(covar_z), len(pcs), len(covar_z_nona), len(traits_nona))


# # Regressing out covariate effects

# In[14]:


def mlr2(X, y):
    
    import pandas as pd
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    from scipy import stats

    nani = np.argwhere(~np.isnan(y.values))
    nani = [i[0] for i in nani]
    
#     print(X.shape, len(y), len(nani))
    
    X = X.iloc[nani]
    y = list(y.iloc[nani])

#     print(X.shape, len(y))
    
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
#     print(est2.summary())
    
    return est2


# In[15]:


# regression, step 1

reg_covar = covar_z_nona[step1_covars].copy()
reg_response = traits_nona.copy()

covar_pval = pd.DataFrame(columns = reg_response.columns, index=reg_covar.columns)
covar_effects = pd.DataFrame(columns = reg_response.columns, index=reg_covar.columns)
reg_corrected = reg_response.copy()

for i in reg_response.columns:
    notna = reg_response[i].notna()
    
    regr = mlr2(reg_covar.loc[notna], reg_response[i].loc[notna])
    covar_pval[i] = regr.pvalues[1:]
    covar_effects[i] = regr.params[1:]

    
covar_pval[covar_pval == 0] = 8.984227e-308


# In[16]:


# correct each phenotype (first step, we correct all, because important)

significant_effects=covar_effects.copy()
# significant_effects[np.invert(significant)] = 0
# significant_effects = significant_effects.transpose()

for i in reg_response.columns:
#     print(i)
    notna = reg_response[i].notna()
    
    reg_corrected[i] = reg_response[i] - list(reg_covar.dot(pd.Series(significant_effects[i])))
#     print(np.corrcoef(reg_corrected[i].loc[notna],reg_response[i].loc[notna]))


# In[17]:


# p-values to logscale, remove infinites

covar_pval.replace(0.0,np.min(covar_pval[covar_pval != 0.0]), inplace=True)
covar_pval = -covar_pval.applymap(np.log10)
covar_pval = covar_pval.transpose()
# covar_pval = covar_pval.loc[sorted(covar_pval.index)]

covar_effects = covar_effects.transpose()


# In[18]:


# significance df

result_shape = covar_pval.shape
significant = covar_pval > -np.log10(0.05 / result_shape[0] / result_shape[1])

asterisks = significant.applymap(str)
asterisks = asterisks.replace('False', "")
asterisks = asterisks.replace('True', "*")


# In[19]:


selected_traits=['DF_all','tau2_all','tau3_all','tau4_all','mean_angle_taa','mean_angle_tva','VD_orig_all','eq_CRAE','eq_CRVE','ratio_CRAE_CRVE','D_std_std','slope']


# In[20]:


cmap = sns.color_palette("viridis", as_cmap=True)

plt.figure(figsize=(8,30))
sns.heatmap(covar_pval, annot=asterisks, fmt='', cmap=cmap);
plt.title("P-values",fontweight='bold')
plt.tight_layout()
plt.savefig(PHENO_PATH+ID+"_correction_step_1_pval.pdf")


# In[21]:


cmap = sns.color_palette("viridis", as_cmap=True)

plt.figure(figsize=(8,30))
# sns.heatmap(np.abs(covar_effects), annot=asterisks, fmt='', cmap=cmap);
# sns.heatmap(covar_effects.drop(columns=['sex']), annot=asterisks.drop(columns=['sex']), fmt='', cmap=cmap);
sns.heatmap(covar_effects, annot=asterisks, fmt='', cmap=cmap);
plt.title("Standardized betas", fontweight='bold')
plt.tight_layout()
plt.savefig(PHENO_PATH+ID+"_correction_step_1_beta.pdf")


# In[22]:


# regression, check correction

reg_covar = covar_z_nona[step1_covars].copy()
# reg_covar = covar_z_nona[['age_center','cylindrical_power','spherical_power']].copy()
# reg_covar = covar_z_nona.drop(['sex','age_center','cylindrical_power','spherical_power'], axis=1)

reg_response = reg_corrected.copy()

covar_pval = pd.DataFrame(columns = reg_response.columns, index=reg_covar.columns)
covar_effects = pd.DataFrame(columns = reg_response.columns, index=reg_covar.columns)
reg_corrected = reg_response.copy()

for i in reg_response.columns:
    notna = reg_response[i].notna()

    regr = mlr2(reg_covar.loc[notna], reg_response[i].loc[notna])

    covar_pval[i] = regr.pvalues[1:]
    covar_effects[i] = regr.params[1:]


# In[23]:


# p-values to logscale, remove infinites
covar_pval = -covar_pval.applymap(np.log10)
covar_pval.replace(np.inf,np.max(covar_pval[covar_pval != np.inf]), inplace=True)
covar_pval = covar_pval.transpose()
# covar_pval = covar_pval.loc[sorted(covar_pval.index)]

covar_effects = covar_effects.transpose()


# In[24]:


# significance df
result_shape = covar_pval.shape
significant = covar_pval > -np.log10(0.05 / result_shape[0] / result_shape[1])

asterisks = significant.applymap(str)
asterisks = asterisks.replace('False', "")
asterisks = asterisks.replace('True', "*")


# In[25]:


cmap = sns.color_palette("viridis", as_cmap=True)

plt.figure(figsize=(7,30))
sns.heatmap(covar_pval, annot=asterisks, fmt='', cmap=cmap, vmax=1.3);
plt.title("P-values",fontweight='bold')
plt.tight_layout()
plt.savefig(PHENO_PATH+ID+"_correction_step_1_control.pdf")


# In[26]:


# regression, step 2

reg_covar = covar_z_nona.drop(step1_covars, axis=1)

reg_response = reg_corrected.copy()

covar_pval = pd.DataFrame(columns = reg_response.columns, index=reg_covar.columns)
covar_effects = pd.DataFrame(columns = reg_response.columns, index=reg_covar.columns)
reg_corrected = reg_response.copy()

for i in reg_response.columns:
    notna = reg_response[i].notna()

    regr = mlr2(reg_covar.loc[notna], reg_response[i].loc[notna])

    covar_pval[i] = regr.pvalues[1:]
    covar_effects[i] = regr.params[1:]
    
covar_pval = covar_pval.transpose()
covar_effects = covar_effects.transpose()


# In[27]:


# p-values to logscale, remove infinites
covar_pval = -covar_pval.applymap(np.log10)
covar_pval.replace(np.inf,np.max(covar_pval[covar_pval != np.inf]), inplace=True)


# In[28]:


# significance df
result_shape = covar_pval.shape
significant = covar_pval > -np.log10(0.05 / result_shape[0] / result_shape[1])

asterisks = significant.applymap(str)
asterisks = asterisks.replace('False', "")
asterisks = asterisks.replace('True', "*")


# In[29]:


# correct each phenotype (2nd step, we correct only significant)

significant_effects=covar_effects.copy()
significant_effects[np.invert(significant)] = 0

for i in reg_response.columns:
#     print(i)
    notna = reg_response[i].notna()
    
    reg_corrected[i] = reg_response[i] - list(reg_covar.dot(pd.Series(significant_effects.loc[i,:])))
#     print(np.corrcoef(reg_corrected[i].loc[notna],reg_response[i].loc[notna]))


# In[30]:


cmap = sns.color_palette("viridis", as_cmap=True)

plt.figure(figsize=(20,30))
sns.heatmap(covar_pval, annot=asterisks, fmt='', cmap=cmap);
plt.title("P-values",fontweight='bold')
plt.tight_layout()
plt.savefig(PHENO_PATH+ID+"_correction_step_2_pval.pdf")


# In[31]:


cmap = sns.color_palette("viridis", as_cmap=True)

plt.figure(figsize=(20,30))
# sns.heatmap(np.abs(covar_effects), annot=asterisks, fmt='', cmap=cmap);
# sns.heatmap(covar_effects.drop(columns=['sex']), annot=asterisks.drop(columns=['sex']), fmt='', cmap=cmap);
sns.heatmap(covar_effects, annot=asterisks, fmt='', cmap=cmap);
plt.title("Standardized betas", fontweight='bold')
plt.tight_layout()
plt.savefig(PHENO_PATH+ID+"_correction_step_2_beta.pdf")


# In[32]:


# regression, step 2 control

reg_covar = covar_z_nona
reg_response = reg_corrected.copy()

covar_pval = pd.DataFrame(columns = reg_response.columns, index=reg_covar.columns)
covar_effects = pd.DataFrame(columns = reg_response.columns, index=reg_covar.columns)
reg_corrected = reg_response.copy()

for i in reg_response.columns:
    notna = reg_response[i].notna()

    regr = mlr2(reg_covar.loc[notna], reg_response[i].loc[notna])

    covar_pval[i] = regr.pvalues[1:]
    covar_effects[i] = regr.params[1:]
    
covar_pval = covar_pval.transpose()
covar_effects = covar_effects.transpose()


# In[33]:


# p-values to logscale, remove infinites
covar_pval = -covar_pval.applymap(np.log10)
covar_pval.replace(np.inf,np.max(covar_pval[covar_pval != np.inf]), inplace=True)


# In[34]:


# significance df
result_shape = covar_pval.shape
significant = covar_pval > -np.log10(0.05 / result_shape[0] / result_shape[1])

asterisks = significant.applymap(str)
asterisks = asterisks.replace('False', "")
asterisks = asterisks.replace('True', "*")


# In[35]:


cmap = sns.color_palette("viridis", as_cmap=True)

plt.figure(figsize=(30,30))
# sns.heatmap(np.abs(covar_effects), annot=asterisks, fmt='', cmap=cmap);
# sns.heatmap(covar_effects.drop(columns=['sex']), annot=asterisks.drop(columns=['sex']), fmt='', cmap=cmap);
sns.heatmap(covar_effects, annot=asterisks, fmt='', cmap=cmap, vmin=-0.05, vmax=0.05);
plt.title("Standardized betas", fontweight='bold')
plt.tight_layout()
plt.savefig(PHENO_PATH+ID+"_correction_step_2_control.pdf")


# In[36]:


# plot corrected

traits_hist = reg_corrected.copy()

cols = []
for idx,i in enumerate(traits_hist.columns):
    cols.append(traits_hist.columns[idx] + " N=" + str(len(traits_hist) - traits_hist[i].isna().sum()))
    
traits_hist.columns = cols

traits_hist.hist(figsize=(40,40), bins=100);
plt.tight_layout()
plt.savefig(PHENO_PATH+ID+"_"+CASE+"_corrected_distribution.pdf")


# In[37]:


# write corrected, plot corrected clustermap in case raw, not qqnorm

tmp = pd.read_csv(traitsfile, index_col=0)
out = pd.DataFrame(index=tmp.index, columns=tmp.columns)
out.update(reg_corrected)
out = out.astype(str)
out = out.replace('nan', '-999')

if CASE == 'qqnorm':
    out.to_csv(PHENO_PATH+OUTID+"_qqnorm.csv", index=False, sep=" ")
else:
    reg_corrected.to_csv(PHENO_PATH+OUTID+".csv")
    
    h = sns.clustermap(data=reg_corrected.corr(), figsize=(40,40), cmap='viridis')
    dgram=h.dendrogram_col.dendrogram
    D = np.array(dgram['dcoord'])
    I = np.array(dgram['icoord'])
    plt.savefig(PHENO_PATH+OUTID+"_clustermap.pdf")
