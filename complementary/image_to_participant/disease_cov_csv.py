import pandas as pd
import numpy as np
import sys

dir_diseases_cov=sys.argv[1]
file_diseases_cov=sys.argv[2]
df_diseases_cov=pd.read_csv(dir_diseases_cov+file_diseases_cov)
list_diseases_cov = list(df_diseases_cov.columns)


list_interest= ['age_center', 'assessment_centre', 'ethnic_background', 'date_center', 'DBP', 'SBP',  'PR', 'age_current_smoker', 'alcohol_intake_frequency',
                'pulse_wave_arterial_stiffness_index', 'BMI', 'N_cigarettes_curr_daily', 'Pack_year_smok',  'HDL_cholesterol', 'LDL_direct', 'Triglycerides',
                'HbA1c', 'age_diabetes', 'age_angina', 'age_heartattack', 'age_DVT', 'age_stroke', 'age_glaucoma', 'age_cataract',
                 'eye_amblyopia', 'eye_presbyopia', 'eye_hypermetropia', 'eye_myopia', 'eye_astigmatism' ,'eye_diabetes', 'age_other_serious_eye_condition',
                 'age_pulmonary_embolism','spherical_power', 'cylindrical_power']


for disease_name in list_interest:
    #disease_name_aux = f'{disease_name}_aux'
    disease_name_both = f'{disease_name}_both'
    disease_name_00 = f'{disease_name}_00'
    disease_name_10 = f'{disease_name}_10'
    #df_diseases_cov[disease_name_both] = np.where(df_diseases_cov["instance"] ==  1, df_diseases_cov[disease_name_10], df_diseases_cov[disease_name_00])
    df_diseases_cov[disease_name_both] = np.where(df_diseases_cov["instance"] == 0, df_diseases_cov[disease_name_00], df_diseases_cov[disease_name_10])
    ## If nan in the instance => we do not take the covariant into account
    df_diseases_cov[disease_name_both] = np.where(df_diseases_cov["instance"].isnull(), np.nan, df_diseases_cov[disease_name_both])
    #df_diseases_cov = df_diseases_cov.drop(columns=disease_name_aux)
    print('df_diseases_cov["instance"].isna().sum()', df_diseases_cov["instance"].isnull().sum())
    print(disease_name_both, df_diseases_cov[disease_name_both].isnull().sum())

#print(df_diseases_cov['spherical_power_both'])

df_diseases_cov['age_center_both_2']=df_diseases_cov['age_center_both']*df_diseases_cov['age_center_both']
df_diseases_cov['spherical_power_both_2']=df_diseases_cov['spherical_power_both']*df_diseases_cov['spherical_power_both']
df_diseases_cov['cylindrical_power_both_2']=df_diseases_cov['cylindrical_power_both']*df_diseases_cov['cylindrical_power_both']
df_diseases_cov.to_csv(dir_diseases_cov+file_diseases_cov)