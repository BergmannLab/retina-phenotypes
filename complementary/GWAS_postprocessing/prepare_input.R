# INPUT: ldscInput file produce by hit_extract/hit_to_csv.R
# this script generates in put for LD Hub http://ldsc.broadinstitute.org/ldhub/
# (calcultion of SNP heritability and rg, i.e., genetic correlation with other traits)
# setwd("/Users/sortinve/Desktop/Vascular_shared_genetics_in_the_retina/GWAS/tVA/2022_02_14_tVA_ageCorrectedVentile5QC/")
# pheno_list <- c("tVA")

foo_prepare_input <- function(pheno_list) {
    for (pheno_name in pheno_list) {

      ldscInput <- read.table(paste(pheno_name,"__ldscInput.csv", sep=""), sep="\t",
                              header=T, stringsAsFactors= F)
      # Read the sample size of the phenotype
      sample_size <-read.table(paste("sample_sizes.txt", sep=""), sep="\t",
                                 header=T, stringsAsFactors= F)
      N <- sample_size[pheno_name][[1]]

      ldscInput['N']=N #48600 #62751 # add a column with sample size
      write.table(ldscInput, file=paste(pheno_name,"__ldscInput_withN.txt", sep="")
                  ,row.names = FALSE, quote=FALSE, sep='\t')
    }
}