import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys 

dir_diseases_cov= sys.argv[1]
file_diseases_cov= sys.argv[2]
describe_file = sys.argv[3]
df_diseases_cov=pd.read_csv(dir_diseases_cov+file_diseases_cov)
list_diseases_cov = list(df_diseases_cov.columns)
print(list_diseases_cov)

### Merge to filter by QC:
phenofiles_dir =  sys.argv[4] #'/NVME/decrypted/scratch/multitrait/UK_BIOBANK_ZERO/participant_phenotype/'
phenofile_used_for_dist_plots = sys.argv[5]

df_data = pd.read_csv(phenofiles_dir + phenofile_used_for_dist_plots, sep=',')
df_data = df_data.rename(columns={'0': 'eid'})
print(len(df_data))


### Split in lsits
old_list_baselines =['eid', 'age_recruitment', 'age_center_00', 'age_center_10','sex', 'ethnic_background_00', 'ethnic_background_10', 'DBP_00','SBP_00','DBP_01','SBP_01','PR_00','PR_01','DBP_10','SBP_10','DBP_11','SBP_11','PR_10','PR_11', 'age_current_smoker_00', 'age_current_smoker_both', 'age_current_smoker_10', 'BMI_00','BMI_10', 'BMI_both'] 

list_baselines =['eid', 'age_recruitment', 'sex', 'age_current_smoker_both', 'BMI_both'] 

list_diseases =['age_diabetes_both','age_angina_both','age_heartattack_both','age_DVT_both','age_stroke_both','age_pulmonary_embolism_both','age_death']
list_ocular = ['age_glaucoma_both','age_cataract_both','eye_diabetes_both','age_other_serious_eye_condition_both']

MAIN_LABELS='mean_angle_taa,mean_angle_tva,tau1_vein,tau1_artery,ratio_AV_DF,eq_CRAE,ratio_CRAE_CRVE,D_A_std,D_V_std,eq_CRVE,ratio_VD,VD_orig_artery,bifurcations,VD_orig_vein,medianDiameter_artery,medianDiameter_vein,ratio_AV_medianDiameter'
#MAIN_LABELS='mean_angle_taa,mean_angle_tva'
main_phenotypes_old_names = list(MAIN_LABELS.split(","))

all_list = list_baselines +list_diseases+list_ocular
df_diseases_cov = df_diseases_cov[all_list]

# Create an empty DataFrame to store all the descriptions
df_combined = pd.DataFrame()

#####
for pheno in main_phenotypes_old_names:
    df_data_completo = df_data.copy()
    df_data_completo = df_data_completo[['eid', pheno]]

    df_data_completo = df_data_completo.dropna()
    print(len(df_data_completo))

    # Merge data
    df_merge = df_data_completo.merge(df_diseases_cov, how='inner', on='eid')
    print(len(df_merge))

    desc3 = df_merge.describe()
    desc3 = desc3.round(2)
    print(desc3)
    if pheno=='mean_angle_taa':
        df_combined = desc3.copy()
        df_combined.drop('eid', axis=1, inplace=True)
    else:
        df_combined = pd.concat([df_combined, desc3], axis=1)
        df_combined.drop('eid', axis=1, inplace=True)

df_combined.T.to_csv(dir_diseases_cov + describe_file)
