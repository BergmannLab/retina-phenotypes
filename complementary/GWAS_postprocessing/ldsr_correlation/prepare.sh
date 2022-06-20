#!/bin/bash
cd /HDD/data/UKBiob/GWAS/2022_06_08_all_phenotypes_LWNet_Decile3/ldscr
for f in *.txt;
do
	nohup /SSD/home/sofia/ldsc/munge_sumstats.py --sumstats $f --out $f --merge-alleles /SSD/home/sofia/ldsc/eur_w_ld_chr/w_hm3.snplist &
done
