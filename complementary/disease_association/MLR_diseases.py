import sys
import function_diseases_associations as fda
import function_figures_associations as ffa

What_type_phenotype=sys.argv[1]
output_dir = sys.argv[2]
diseases_file = sys.argv[2] + sys.argv[3]
pheno_file =  sys.argv[4] + sys.argv[5]

file_info_name='pheno_diseases_info.csv'
pheno_info_file = f'~/retina-phenotypes/complementary/{file_info_name}'

display_info=True
save_additional_figs=False

if What_type_phenotype == 'main':
    list_phenotypes=list(sys.argv[6].split(","))
    list_phenotypes_new=list(sys.argv[8].split(","))

elif What_type_phenotype == 'suplementary':
    list_phenotypes=list(sys.argv[7].split(","))
    list_phenotypes_new=list(sys.argv[9].split(","))

else:
    print('Error, should be main or suplementary!')
    #sys.stop()

#print(list_phenotypes, list_phenotypes_new)

####################### 1 - Read files:


df_diseases, inf, list_diseases, list_diseases_bin, list_diseases_con, list_diseases_cat = fda.read_diseases_files(diseases_file, pheno_info_file)

# Number of cases and controls per disease
if display_info:
    print('Number of nans and ratio nans/no_nans:')
    for disease_name in list_diseases:
        fda.N_of_nans_and_nonans(df_diseases,disease_name)
        #plt.hist(df_diseases[disease_name])
        #plt.title(disease_name)
        #plt.show()

# plot histogram diseases
ffa.hist_diseases_plot(df_diseases, list_diseases, output_dir, save_additional_figs)

####################### 2 - Phenotypes:

df_pheno =  fda.read_pheno(pheno_file)

####################### 3 - Merge diseases and phenotpyes
df_pheno_dise = fda.merge_pheno_diseases(df_pheno, df_diseases)

# Filtrate only by the phenotypes and diseases of interest
df_pheno_dise = fda.filtrate_col(df_pheno_dise, list_phenotypes+list(list_diseases))

# Replace nans by 0's in cases/controls
df_pheno_dise = fda.column_startwith_replace_nan_by_0(df_pheno_dise)

# Standardise continuous diseasesa and continuous phenotypes if not z-scored
df_pheno_dise =  fda.std_contin_col(df_pheno_dise, list_diseases_con)

# Check the type of variables and convert objects to numeric:

### convert to type numeric the columns that are not
#df_pheno_dise['date_reported_atherosclerosis'] = pd.to_numeric(df_pheno_dise['date_reported_atherosclerosis']) # 'date_disorders_arteries_arterioles', 'date_AD', 'date_death',
df_pheno_dise = fda.col_to_numeric(df_pheno_dise)

####################### 4 - Correlation
ffa.corr_heatmap(df_pheno_dise, output_dir, save_additional_figs)

####################### 5 - MLR
#  Linear/logistic regression:
### - Note-> Avoid using sklearn -> Pvalues and betas: scikit-learn's LinearRegression doesn't calculate this information but if needed one can extend the class to do it
### - Using from statsmodels.formula.api import ols, logit
betas, log10p = fda.compute_pval_betas_LR(df_pheno_dise, inf, list_diseases, list_phenotypes)

####################### 6 - save betas, and p values

fda.save_betas_pval(betas, log10p, output_dir, What_type_phenotype)

print('diseases_file', diseases_file)
print('pheno_file', pheno_file)
print('output_dir', output_dir)