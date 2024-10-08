from PascalX import genome
from PascalX import xscorer
import sys
# EDIT THIS WITH PATH TO CONFIG FILE
sys.path.append('/NVME/scratch/olga/retina-phenotypes/pascalx/config/retina-disease')
import PascalX_xscorer_config as cfg
import datetime
import pickle
import pandas as pd

## Download genome annotation (comment after 1st run)
print('Downloading genome annotation')
D = genome.genome()
D.get_ensembl_annotation(cfg.genome_dir, genetype='protein_coding', version='GRCh38')
print('Download complete')

gwas_a_done = [] # list of done gwas_a traits to avoid duplicate pairs when list_a=list_b
for gwas_a in cfg.list_a:
    gwas_a_done.append(gwas_a)
    for gwas_b in cfg.list_b:
        for direc in cfg.direction:
            if gwas_b not in gwas_a_done: # avoid duplicate pairs
                # Print start time
                starttime = datetime.datetime.now()
                print('Start time:', starttime.strftime('%Y-%m-%d %H:%M:%S'))
                
                # Load xscorer
                print('Loading xscorer')
                if direc == 'coherence':
                    X = xscorer.zsum(MAF=cfg.maf, leftTail=False, gpu=cfg.gpu) # side to test = coherence
                elif direc == 'anti-coherence':
                    X = xscorer.zsum(MAF=cfg.maf, leftTail=True, gpu=cfg.gpu) # side to test = anti-coherence
                
                # Load reference panel
                print('Loading reference panel')
                X.load_refpanel(cfg.refpanel_dir, keepfile=cfg.keepfile, parallel=22)

                # Load genome annotation
                print('Loading genome annotation')
                X.load_genome(cfg.genome_dir)

                # Load GWAS summary statistics
                print(f'Loading GWAS A: {gwas_a}')
                # Get column indices based on column names
                filename_a = f'{cfg.gwas_a_dir}{cfg.pfx_a}{gwas_a}{cfg.sfx_a}'
                cols_a = pd.read_csv(filename_a, sep=cfg.del_a, nrows=0).columns.tolist()
                rscol_a = cols_a.index(cfg.rscol_a_name)
                pcol_a = cols_a.index(cfg.pcol_a_name)
                bcol_a = cols_a.index(cfg.bcol_a_name)
                a1col_a = cols_a.index(cfg.a1col_a_name)
                a2col_a = cols_a.index(cfg.a2col_a_name)
                X.load_GWAS(filename_a,name=gwas_a,rscol=rscol_a,pcol=pcol_a,bcol=bcol_a,a1col=a1col_a,a2col=a2col_a,delimiter=cfg.del_a,header=True)
                print(f'Loading GWAS B: {gwas_b}')
                if gwas_b in ['30750_irnt']: # exception for files ending with '.varorder.tsv'
                    filename_b = f'{cfg.gwas_b_dir}{cfg.pfx_b}{gwas_b}{cfg.sfx_b_varorder}'
                else:
                    filename_b = f'{cfg.gwas_b_dir}{cfg.pfx_b}{gwas_b}{cfg.sfx_b}'
                cols_b = pd.read_csv(filename_b, sep=cfg.del_b, nrows=0).columns.tolist()
                rscol_b = cols_b.index(cfg.rscol_b_name)
                pcol_b = cols_b.index(cfg.pcol_b_name)
                bcol_b = cols_b.index(cfg.bcol_b_name)
                a1col_b = cols_b.index(cfg.a1col_b_name)
                a2col_b = cols_b.index(cfg.a2col_b_name)
                X.load_GWAS(filename_b,name=gwas_b,rscol=rscol_b,pcol=pcol_b,bcol=bcol_b,a1col=a1col_b,a2col=a2col_b,delimiter=cfg.del_b,header=True)
                print('Loading complete')

                # Match alleles
                print('Matching alleles')
                X.matchAlleles(gwas_a, gwas_b)
                print('Matching done')

                # QQ normalise p-values
                print('Jointly QQ normalising p-values')
                X.jointlyRank(gwas_a, gwas_b)
                print('Joint ranking done')

                # Cross-scoring
                print(f'Cross-scoring (sample overlap correction = {cfg.overlap_corr})')
                if cfg.overlap_corr:
                    # Using LDSC intercept as correction factor
                    if gwas_b in ['30760_irnt', '30780_irnt', '30870_irnt', '30750_irnt']: # exception for wrongly named Neale files
                        ldsc_res_file = cfg.ldsc_res_fmt.replace('A', gwas_a).replace('B.ldsc.imputed_v3', gwas_b+'.imputed_v3.ldsc')
                    else:
                        ldsc_res_file = cfg.ldsc_res_fmt.replace('A', gwas_a).replace('B', gwas_b)
                    f = open(cfg.ldsc_res_dir+ldsc_res_file, 'r') # read .log file from LDSC
                    gcov_i = 0 # initialise index of line with Genetic Covariance
                    for i, line in enumerate(f.readlines()):
                        if gcov_i!=0:
                            if 'Intercept:' in line:
                                int = float(line.split(' ')[1])
                                break
                        if 'Genetic Covariance' in line:
                            gcov_i = i # index of Genetic Covariance line, entering loop above
                    f.close()
                    print(f'LDSC intercept = {int}')
                    R = X.score_all(E_A=gwas_a, E_B=gwas_b, parallel=cfg.n_cpu, nobar=True, pcorr=int)
                else:
                    R = X.score_all(E_A=gwas_a, E_B=gwas_b, parallel=cfg.n_cpu, nobar=True)
                print('Cross-scoring complete')

                # Save output as pickle and .csv
                print('Saving output')
                f = open(f'{cfg.out_dir}{gwas_a}__{gwas_b}__{direc}__xscorer_results.p', 'wb')
                pickle.dump(R, f)
                f.close()
                df = pd.DataFrame(R[0])
                df.rename(columns={0: 'gene_name', 1: 'pval', 2: 'n_snps', 3: 'coherence'}, inplace=True)
                df.to_csv(f'{cfg.out_dir}{gwas_a}__{gwas_b}__{direc}__xscorer_results.csv', index=False)

                print(f'Output saved for {gwas_a} and {gwas_b} {direc}')

                # Print end time
                endtime = datetime.datetime.now()
                print('End time:', endtime.strftime('%Y-%m-%d %H:%M:%S'), '\n')

print('END OF SCRIPT')

