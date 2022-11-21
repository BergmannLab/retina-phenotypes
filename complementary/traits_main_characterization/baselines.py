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

desc3 = df_diseases_cov.describe()
desc3 = desc3.round(3)
print(desc3)
desc3.to_csv(dir_diseases_cov + describe_file)


list_main = ['age_center', 'sex', 'ethnic_background', 'DBP', 'SBP', 'PR','age_current_smoker', 
'alcohol_intake_frequency', 'alcohol_intake_frequency', 'pulse_wave_arterial_stiffness_index',
'N_cigarettes_curr_daily', 'Pack_year_smok', 'HDL_cholesterol', 'LDL_direct', 'Triglycerides', 'HbA1c']

list_vascular_diseases = ['age_diabetes', 'age_angina', 'age_heartattack', 'age_DVT',
'age_stroke']
list_severe_ocular_diseases = ['age_glaucoma', 'age_cataract']

list_eyesight_ocular_diseases = ['eye_amblyopia', 'eye_presbyopia', 'eye_hypermetropia', 
'eye_myopia', 'eye_astigmatism', 'eye_diabetes', 'eye_myopia', 'age_other_serious_eye_condition']

list_all = list_main + list_vascular_diseases + list_severe_ocular_diseases + list_eyesight_ocular_diseases

list_not_to_analzye = ['eid','date_center_00', 'date_center_10', 'date_center_20', 
                        'date_death', 'date_AD', 'date_reported_atherosclerosis', 
                        'date_disorders_arteries_arterioles']

def histogram(value, column_i):
    plt.figure()
    plt.title(column_i)
    value.plot.hist(bins=100, rwidth=0.5)
    plt.show()
    #plt.ylabel('Frequency')
    #mu = value.mean().round(decimals=2)
    #std = value.std().round(decimals=2)

# for column_i in list_diseases_cov:
#     if column_i in list_not_to_analzye:
#         continue
#     else:
#         histogram(df_diseases_cov[column_i],column_i)


