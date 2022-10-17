
# # Compute gcorr diseases and phenotypes (ldscr)
# ###### Created 03/08/2022
# ###### Last modification 17/10/2022

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import glob, os
from os import listdir
from os.path import isfile, join
from datetime import datetime
DATE = datetime.now().strftime("%Y-%m-%d")

ventile_num= sys.argv[1]
save_dir= sys.argv[2] 
traits_phenos = list(sys.argv[3].split(","))
traits_phenos_new = list(sys.argv[4].split(",")) #list(sys.argv[4].split(","))
path = sys.argv[5] #'/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/gcorr_diseases/2022_08_03_ventile5/'

high_med_conf= True #To select the diseases that have good confidence



if high_med_conf:
    diseases_traits = {
        '4079':'DBP',
        '4080':'SBP',
        #'102':'PR',
        '30760':'HDL cholesterol',
        '1558':'Alcohol intake freq',
        '21021':'Pulse wave ASI',
        '30780':'LDL direct',
        '30870':'Triglycerides'
        }

else:
    diseases_traits = {
        '4079':'DBP',
        '4080':'SBP',
        #'102':'PR',
        '1558':'Alcohol intake freq',
        '21021':'Pulse wave arterial stiffness',
        #'40000':'Date death',
        #'42020':'Date AD',
        '30760':'HDL cholesterol',
        '30780':'LDL direct',
        '30870':'Triglycerides',
        '2976':'Age diabetes',
        '3627':'Age angina',
        '3894':'Age heart attack',
        '4012':'Age DVT',
        '4056':'Age stroke',
        '40007':'Age death',
        #'4689':'Age glaucoma',
        '4700':'Age cataract',
        #'5408':'Amblyopia',
        '5610_1':'Presbyopia 1',
        '5610_2':'Presbyopia 2',
        '5610_3':'Presbyopia 3',
        '5832_3':'Hypermetropia 3',
        '5843_1':'Myopia 1',
        '5843_2':'Myopia 2',
        '5843_3':'Myopia 3',
        '5855_1':'Astigmatism 1',
        '5855_2':'Astigmatism 2',
        '5855_3':'Astigmatism 3',   
        #'5890':'Diabetes eye',
        #'5945':'Eye other',
        '1717':'Skin colour',
        '1747_1':'Hair colour 1',
        '1747_2':'Hair colour 2',
        '1747_3':'Hair colour 3',
        '1747_4':'Hair colour 4',
        '1747_5':'Hair colour 5',
        '1747_6':'Hair colour 6'
        #'4022':'Age pulmonary embolism',
        #'131380':'Circulatory sys dis',
        #'131390':'Other art dis'
        }


traits_all = list(diseases_traits.keys())
traits_phenos_new = list(diseases_traits.values())


# path This we can read from config
def try_to_compute_all(traits_all, traits_phenos, path):
    #save_path = path
    l_diseases_all=[]
    for trait in traits_all:
        #print(trait)
        for file in os.listdir(path):
            #print(file, '\n')
            if file.startswith(trait):
                if file.endswith('.tsv'):
                    #print('PHENO', trait)
                    #print(file, '\n')
                    df_ss = pd.read_csv(path + file,  nrows=1, sep='\t')
                    #print(df_ss['N'].iloc[0])
                    print(file)
                    data={
                        'pheno':  trait,
                        'file':  file
                        #,'N': df_ss['N'].iloc[0],
                        }
                    l_diseases_all.append(data)

    df_diseases_all =pd.DataFrame(l_diseases_all)
    #file_name_end = '_irnt.gwas.imputed_v3.both_sexes.tsv'

    l_traits_file=[]
    for trait in traits_phenos:
        file_pheno= trait + '__munged.sumstats.gz'
        l_traits_file.append(file_pheno)

    traits_files = l_traits_file + list(df_diseases_all['file'])
    #traits_names = traits_phenos + list(df_diseases_all['pheno'])
    print(len(traits_files), len(l_traits_file), len(list(df_diseases_all['file'])))
    return df_diseases_all



traits_reduced = list(diseases_traits.keys())


df_diseases_all = try_to_compute_all(traits_all, traits_phenos, path)

#datafields_irnt = [ dat + "_irnt.gwas.imputed_v3.both_sexes.tsv.sumstats.gz" for dat in traits_reduced]
datafields_irnt = [ dat for dat in df_diseases_all['file']]
datafields_pheno = [ dat + "__munged.sumstats.gz" for dat in traits_phenos]
diseasess_tra_aux = [ dat for dat in df_diseases_all['pheno']]

traits_col_index = traits_phenos + diseasess_tra_aux
traits_names = datafields_pheno + datafields_irnt


# filter the files names containing 2 traits
def read_ldsr(traits_files, traits_col_index, path):
    df_cov=pd.DataFrame(columns =traits_col_index, index=traits_col_index)
    df_corr=pd.DataFrame(columns =traits_col_index, index=traits_col_index)
    df_std=pd.DataFrame(columns =traits_col_index, index=traits_col_index)
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
                        df_cov.iloc[i][j] = round(float(split[ split.index('gencov:') +1 ]),3)
                        df_cov.iloc[j][i] = round(float(split[ split.index('gencov:') +1 ]),3)
                        #print(split)
                    if('Correlation:' in split):
                        #print(line)
                        #print(split)
                        df_corr.iloc[i][j] = round(float(split[ split.index('Correlation:') +1 ]),3)
                        df_corr.iloc[j][i] = round(float(split[ split.index('Correlation:') +1 ]),3)
                        #print(array2)
                        #print(split )
                        #print( df_corr.iloc[i][j], float(split[ split.index('Correlation:') +1 ]) )
                        #print( df_corr.iloc[j][i], float(split[ split.index('Correlation:') +1 ]))
                        df_std.iloc[i][j] = split[3]
                        df_std.iloc[j][i] = split[3]
    return df_cov, df_corr, df_std

df_cov, df_corr, df_std2 = read_ldsr(traits_names, traits_col_index, path)     


def rename_col_index(df, l_diseases_old, l_diseases_new, l_phenos_old, l_phenos_new):
    df.rename(columns=dict(zip(l_diseases_old, l_diseases_new)), inplace=True)
    df.rename(index=dict(zip(l_diseases_old, l_diseases_new)), inplace=True)
    return df

def detele_col_index(df, l_cols_delete, l_rows_delete, l_diseases_new, l_phenos_old, l_phenos_new):
    df=df.drop(columns=l_cols_delete)
    df=df.drop(index=l_rows_delete)
    df= rename_col_index(df, l_rows_delete, l_diseases_new, l_phenos_old, l_phenos_new)
    
    return df


df_corr_simpl= detele_col_index(df_corr, traits_phenos, traits_reduced, list(diseases_traits.values()),traits_phenos, traits_phenos_new)
df_std_simpl = detele_col_index(df_std2,  traits_phenos, traits_reduced, list(diseases_traits.values()), traits_phenos, traits_phenos_new)
df_corr_simpl


### Round the std values
for col in df_std_simpl.columns:
    df_std_simpl[col] = df_std_simpl[col].str.replace("(", "", regex=True)
    df_std_simpl[col] = df_std_simpl[col].str.replace(")", "", regex=True)

df_std_simpl = df_std_simpl.astype(float)
df_std_simpl = df_std_simpl.round(3)
df_std_simpl


#print(df_reducida.columns, df_std.columns)
df = df_corr_simpl.astype(str) + ' (' + df_std_simpl.astype(str)+ ')'

df_corr_simpl_aux = df_corr_simpl.copy()
df_corr_simpl_aux = df_corr_simpl_aux.astype(float)

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
                 cellColours=plt.cm.coolwarm(df_corr_simpl.values, alpha=0.2),
                 loc='center',
                 fontsize=16,
                 colWidths=[0.15 for x in df.columns])
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(3.7, 3.5) # make table a little bit larger
fig.tight_layout()
#plt.show()
fig.savefig(path+str(DATE)+'_'+'ventile'+str(ventile_num)+'_diseases_gcorr.pdf', bbox_inches='tight',dpi=250)

print('todo ok')
