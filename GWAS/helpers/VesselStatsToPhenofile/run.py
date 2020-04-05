import pandas as pd
import os
            
# read samples file: create a dataframe that contains uniquely the participant ids
def read_samples(sample_file):
    samples = pd.read_csv(sample_file, sep=" ")
    # drop test value at index 0: "0 0 0 D"
    samples.drop(samples.index[0], inplace=True) 
    # keep non-missing samples only
    samples = samples[samples['missing'] == 0]
    # drop column that are not needed (missing and sex)
    samples.drop(samples.columns[[1,2,3]], axis=1, inplace=True)
    # rename the one column that is needed: eid
    samples.rename(columns = {'ID_1':'eid'}, inplace = True)
    samples.set_index("eid", inplace = True)
    return samples

def calculate_average(stats_i, stats_collision):
    return (stats_i.values + stats_collision.values) / 2
    
def VesselStats_to_phenofile(output_dir, sample_file, stats_dir):
    
    # create an empty dataframe to hold stats
    # index = eid from sample file (to respect UKBB ordering)
    stats_phenotypes = read_samples(sample_file)
    # add stats columns to dataframe
    stats_phenotypes['max_diameter'] = None
    stats_phenotypes['min_diameter'] = None
    stats_phenotypes['median_diameter'] = None
    stats_phenotypes['median_tortuosity'] = None
    stats_phenotypes['std_tortuosity'] = None
    
    # import stats for each input file
    add_once=0
    replace=0
    not_in_sample=[]
    for i in os.listdir(stats_dir):
        if not i.endswith("stats.tsv"): 
            ###print("WARNING: " + i + " will be ignored (not a stats file)")
            continue # process stats files only
        # import file stats to dataframe
        stats_i = pd.read_csv(stats_dir+"/"+i, sep="\t").iloc[0]
        # remove first element (it is a quality measure, not needed now)
        stats_i = stats_i.drop(stats_i.index[0])
        eid_i = float(i[0:7]) # i.split("_")[0]
        try: # eid might not be present in stats_phenotypes
            collision = stats_phenotypes.loc[eid_i] 
            not_present = collision[0] == None
            if not_present: # simply add
                stats_phenotypes.loc[eid_i] = stats_i
                add_once = add_once+1
            else: # average stats from iteration (stats_i) and those present (collision)
                stats_phenotypes.loc[eid_i] = calculate_average(stats_i,collision)
                replace = replace+1
        except KeyError:
            eid_not_in_sample = str(eid_i)[:-2]
            ###print("WARNING: eid " + eid_not_in_sample + " will be omitted (not in sample file)")
            not_in_sample.append(eid_not_in_sample) 
        
    # replace NAs according to BGENIE convention
    valid = stats_phenotypes.iloc[:,1].notna().sum()
    print("\nPhenofile has been built:")
    print(str(valid) + " entries in total")
    print("(" + str(replace) + " were averages of two eyes)")
    print("(" + str(add_once-replace) + " were from single eyes)")
    print(str(len(not_in_sample)) + " were omitted (not in sample file)")
    stats_phenotypes.fillna(-999, inplace=True)
    # output
    output_file  = output_dir + "/VesselStatsPhenofile.csv"
    stats_phenotypes.to_csv(output_file, sep=' ', index = False)
    omitted_file = output_dir + "/NotInSampleFile.csv"
    with open(omitted_file, "w") as outfile:
        outfile.write("\n".join(not_in_sample))


def main():
    output_dir = os.sys.argv[1]
    sample_file = os.sys.argv[2]
    stats_dir = os.sys.argv[3]
    print("Turning Vessel Stats (ARIA) to GWAS phenofile")
    print("using sample file: " + sample_file + "\n")
    VesselStats_to_phenofile(output_dir, sample_file, stats_dir)
    print("DONE")
  
if __name__== "__main__":
    main()
