For genetic association between diseases and your traits you have to:
1- Go to https://github.com/mjbeyeler/server_scripts/
2- Clone it and in the folder /GWASss/ (branch Xscore if it is not in the master)
3- Modify if needed and run dowloand_ss.ipynb (if you have not yet the files downloaded, always check)
4- THE FOLLOWING DOES NOT WORK PROPERLY: In the same folder (once the download is done, ('/HDD/data/ukbb/disease_sumstats/VARIANTS/') ) 4a) first filter: df = df[df['low_confidence_variant']==False]
and then 4b) run prepare_ss.sh (this would prarare the files in the format needed for ldsr) . 
INSTEAD- USE THE LDSR FILES ALREADY PROVIDED BY NEALE  ('/HDD/data/ukbb/ldsc_neale/')
# Note, always make sure that this is working for all the files. 
# Files used:  irnt version of '4079':'DBP', '4080':'SBP', '30760':'HDL cholesterol', 
#   '1558':'Alcohol intake freq', '21021':'Pulse wave ASI', '30780':'LDL direct', and '30870':'Triglycerides'
5- Activate ldsr and compute ldsr simple_prepare.sh on these files
6- Move these .gz files to the folder where you have your traits (all the files, phenos and diseases should have the .gz)
7- Activate ldsr and run compute_gcorr.sh changing the directory.
8- in retina-phenotypes/complementary/GWAS_postprocessing/ldsr_correlation you can run gcorr_disases_ldsr.py

N*-> /HDD/data/ukbb/disease_sumstats/VARIANTS/ for Pascal and /HDD/data/ukbb/ldsc_neale/ for ldsr