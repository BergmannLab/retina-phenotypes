import glob
import os
import pandas as pd
from datetime import datetime

phenotypes_directory='/NVME/decrypted/ukbb/fundus/phenotypes/lwnet_QC2/'
DATE = datetime.now().strftime("%Y-%m-%d")

os.chdir(phenotypes_directory)

i=0
for file in glob.glob("*.csv"): 
    print('files ', file)
    if i==0:
        df_data = pd.read_csv(phenotypes_directory + file, sep=',')
        df_data.rename(columns={ df_data.columns[0]: "image" }, inplace = True)
        print('Initial len(df_data) ', len(df_data))
    # Avoid aria_phenotypes and CRAE, CRVE since they are alreay in ratios:
    elif (file=='2022-06-07_aria_phenotypes.csv') or (file=='2022-06-07_CRAE.csv') or (file=='2022-06-07_CRVE.csv'):
        print('Not included')
        continue
    else:
        df_new = pd.read_csv(phenotypes_directory + file, sep=',')
        print('len(df_new) ', len(df_new))
        df_new.rename(columns={df_new.columns[0]: "image" }, inplace = True)
        df_data=df_data.merge(df_new, how='inner', on='image')
        print('Others len(df_data) ', len(df_data))
    i=i+1

print('Final len(df_data) ', len(df_data)) 
df_data.to_csv(phenotypes_directory + DATE + "_all_phenotypes.csv", sep=',', index=False)