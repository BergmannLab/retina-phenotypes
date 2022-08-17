#!/usr/bin/env python
# coding: utf-8

# # Compute gcorr diseases and phenotypes (ldscr)
# ###### Created 03/08/2022
# ###### Last modification 17/08/2022

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import glob, os
from os import listdir
from os.path import isfile, join
import sys
from datetime import datetime

DATE = datetime.now().strftime("%Y-%m-%d")

ventile_num= sys.argv[1]
save_dir=sys.argv[2]
traits_phenos = list(sys.argv[3].split(","))
traits_phenos_new = list(sys.argv[4].split(","))
path = sys.argv[5] #'/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/gwas/2022_08_03_ventile5/main_phenos/gcorr_diseases/' #'/HDD/data/ukbb/disease_sumstats/files_modified/'

reduced_diseases_traits = {
'4079':'DBP ',
'4080':'SBP',    
'2976':'Age diabetes',
'3627':'Age angina',
'3894':'Age heart attack',
#'4689':'Age glaucoma',
'4700':'Age cataract'}

traits_reduced = list(reduced_diseases_traits.keys())


#irnt is the normalized
datafields_irnt = [ dat + "_irnt.gwas.imputed_v3.both_sexes.tsv.sumstats.gz" for dat in traits_reduced] 
datafields_pheno = [ dat + "__munged.sumstats.gz" for dat in traits_phenos]

traits_col_index = traits_phenos + traits_reduced
traits_names = datafields_pheno + datafields_irnt


# filter the files names containing 2 traits
def read_ldsr(traits_files, traits_col_index):
    df_cov=pd.DataFrame(columns =traits_col_index, index=traits_col_index)
    df_corr=pd.DataFrame(columns =traits_col_index, index=traits_col_index)
    #2976_irnt.gwas.imputed_v3.both_sexes.tsv.sumstats.gz_4700_irnt.gwas.imputed_v3.both_sexes.tsv.sumstats.gz.log
    #2976_irnt.gwas.imputed_v3.both_sexes.tsv.sumstats.gz_D_A_std__munged.sumstats.gz.log

    for i  in range(len(traits_files)):
        for j in range(len(traits_files)):
            h2 = []
            file_both_name = traits_files[i]+'_'+ traits_files[j]+'.log'
            dir_traitsfile = path+file_both_name
            #print(dir_traitsfile)
            with open(dir_traitsfile) as fp:
                #print(fp)
                #print(traits_files[i],traits_files[j])
                Lines = fp.readlines()
                for line in Lines:
                    #print(line)
                    split = line.split()
                    if('gencov:' in split):
                        df_cov.iloc[i][j] = float(split[ split.index('gencov:') +1 ])
                        df_cov.iloc[j][i] = float(split[ split.index('gencov:') +1 ])
                        #print(split)
                    if('Correlation:' in split):
                        #print(line)
                        #print(split)
                        df_corr.iloc[i][j] = float(split[ split.index('Correlation:') +1 ]) 
                        df_corr.iloc[j][i] = float(split[ split.index('Correlation:') +1 ])
                        #print(array2)
                        #print(split )
                        #print( df_corr.iloc[i][j], float(split[ split.index('Correlation:') +1 ]) )
                        #print( df_corr.iloc[j][i], float(split[ split.index('Correlation:') +1 ]))
    return df_cov, df_corr

df_cov, df_corr = read_ldsr(traits_names, traits_col_index)
#df_corr


df_corr = df_corr.astype(float)
df_reducida = df_corr[traits_reduced]
df_reducida= df_reducida.drop(index=traits_reduced) 

df_reducida.rename(columns=dict(zip(list(reduced_diseases_traits.keys()), list(reduced_diseases_traits.values()))), inplace=True)
df_reducida.rename(index=dict(zip(traits_phenos, traits_phenos_new)), inplace=True)

df_reducida= df_reducida.round(2)
#df_reducida


df = df_reducida

rcolors = plt.cm.Greys(np.full(len(df.index), 0.15))
ccolors = plt.cm.Greys(np.full(len(df.columns), 0.15))


fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')


#create table
table = ax.table(cellText=df.values, 
                 rowColours=rcolors,
                 colLabels=df.columns, 
                 rowLabels=df.index,
                 colColours=ccolors,
                 rowLoc='center',
                 colLoc='center',
                 cellLoc='center',
                 cellColours=plt.cm.coolwarm(df.values, alpha=0.2),
                 loc='center',
                 fontsize=16,
                 colWidths=[0.12 for x in df_reducida.columns])
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(3.7, 3.5) # make table a little bit larger
fig.tight_layout()
#plt.show()
fig.savefig(save_dir+str(DATE)+'_'+'ventile'+str(ventile_num)+'_diseases_gcorr.pdf', bbox_inches='tight',
            dpi=150)


### replace nan by 0
def replace_nans_by_zeros(df_corr):
    df_corr = df_corr.replace(np.nan, 0)
    return df_corr

