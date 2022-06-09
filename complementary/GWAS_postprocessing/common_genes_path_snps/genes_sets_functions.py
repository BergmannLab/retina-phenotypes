import pandas as pd
import numpy as np


def create_df(dir, p_value_min):
    x = pd.read_csv(dir, delimiter='\t', header=None, names=['gen', 'p'])
    x['-log10(p)'] = -np.log10(x['p'])
    y = x[x['-log10(p)'] >= p_value_min]
    return y.sort_values('-log10(p)')


def create_df_path(dir, p_value_min):
    x = pd.read_csv(dir, delimiter=' ', header=None, names=['pathway', 'algo', 'algo2', 'p'])
    x['-log10(p)'] = -np.log10(x['p'])
    y = x[x['-log10(p)'] >= p_value_min]
    return y.sort_values('-log10(p)')
