For genetic association between diseases and your traits you have to:
1- Go to https://github.com/mjbeyeler/server_scripts/
2- Clone it and in the folder /GWASss/ (branch Xscore if it is not in the master)
3- Modify if needed and run dowloand_ss.ipynb (if you have not yet the files downloaded, always check)
4- In the same folder (once the download is done) run prepare_ss.sh (this would prarare the files in the format needed for ldsr)
# Note, always make sure that this is working for all the files
5- Compute ldsr prepare.sh on these files
6- Move these .gz files to the folder where you have your traits (all the files, phenos and diseases should have the .gz)
7- Run compute_gcorr.sh changing the directory.
8- in retina-phenotypes/complementary/GWAS_postprocessing/ldsr_correlation you can run fcorr_diseases_ldsr_output.ipynb