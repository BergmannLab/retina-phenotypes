#!/usr/bin/env python
# coding: utf-8

# In[15]:


# fct for multiple linear regression

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


# In[56]:


# dirs

outdir = "/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO" + "/qc"


# In[71]:


# load data

import sys,os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

VD=pd.read_csv("/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/image_phenotype/2022-07-06_vascular_density.csv", index_col=0)
age_center=pd.read_csv("/NVME/decrypted/ukbb/labels/1_data_extraction/ukb34181.csv", usecols=['eid','21003-0.0','21003-1.0'], index_col=0)
age_center.columns=['0','1']
print(age_center, VD)

image_age=[]
for i in VD.index:
    eid,lr,instance,residual=i.split("_")

    image_age.append(age_center[instance].loc[int(eid)])
#slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_agecorr['age'],df_agecorr['PBV_corrected'])

VD['age_center'] = image_age
VD['age_center_2'] = [i**2 for i in image_age]

print(VD.isna().sum())


# In[73]:


# nans are assumed zero vD

# VD.replace(np.nan, 0, inplace=True)
VD = VD.dropna()


# In[14]:


#import seaborn as sns

#sns.kdeplot('age_center', 'VD_orig_all', data=VD)


# In[74]:


# linear regression

regr=mlr2(VD[['age_center', 'age_center_2']], VD['VD_orig_all'])
betas=regr.params[1:]


# In[75]:


# correction for age and age-squared

corr = VD['VD_orig_all'] - list(VD[['age_center', 'age_center_2']].dot(pd.Series(betas)))
VD['corr'] = corr


# In[76]:


# Ventiles

ventiles = [0] + list(np.quantile(corr, np.arange(1,20)/20))


# In[ ]:





# In[77]:


for i,vent in enumerate(ventiles):
    keep = VD[VD['corr'] >= vent].index
    print(len(keep))
    
    with open(outdir + "/agecorrected_ventiles" + str(i) + ".txt", 'w') as f:
        for img in keep:
            f.write(img + '\n')
        


# In[63]:


plt.scatter(VD['age_center'], VD['VD_orig_all'])


# In[50]:


plt.scatter(VD['age_center'], corr)


# In[37]:


plt.hist(corr * 100, 100)


# In[ ]:




