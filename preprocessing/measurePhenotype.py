import itertools
import os, sys
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import csv
import PIL
import math
import cv2
from skimage import data, io, filters
from datetime import datetime
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from scipy import stats
from multiprocessing import Pool
from PIL import Image
from matplotlib import cm

DATE = datetime.now().strftime("%Y-%m-%d")

## Command line argumetns
aria_measurements_dir = sys.argv[3]
qcFile = sys.argv[1]
phenotype_dir = sys.argv[2]
lwnet_dir = sys.argv[4] 
OD_POSITIONS = sys.argv[5]
df_OD = pd.read_csv(OD_POSITIONS, sep=',')
traits = sys.argv[6].split(',')

## Parameters for temporal angle
delta = sys.argv[7] 
R_0 = sys.argv[8] 
radius = [R_0, R_0+delta, R_0+2*delta, R_0+3*delta, R_0+4*delta, R_0+5*delta]
min_ta = sys.argv[9] 
max_ta = sys.argv[10] 
lower_accept = sys.argv[11] 
upper_accept = sys.argv[12] 
## Parameters for bifurcations
norm_acceptance = sys.argv[13] 
neighborhood_cte = sys.argv[14] 
## Parameter for N main vessels
limit_diameter_main = sys.argv[15] 
## Parameter for Vascular Density
mask_radius=sys.argv[16]  # works for UKBB, may be adapted in other datasets, though only used for PBV (percent annotated as blood vessels) phenotype


##### Trait input data main functions:

def get_data_unpivot(path):
    """
    :param path:
    :return:
    """
    # Note -> Use .read_fwf since *.tsv have diferent fixed-width formatted lines
    # df = pd.read_fwf(path, sep='\t', header=None)
    # Split by tab and expand columns
    with open(path) as fd:
        rd_2 = csv.reader(fd, delimiter='\t')
        df = pd.DataFrame(list(rd_2))
    # get index in order to get the each segment id
    df.reset_index(inplace=True)
    # unpivot dataframe to get all the segments coordinates in one column
    df_unpivot = pd.melt(df, id_vars=['index']).sort_values(by=['index', 'variable'])[['index', 'value']]
    # remove null created by fixed-width
    df_unpivot = df_unpivot[~df_unpivot['value'].isnull()].copy()
    # get another index to get a secod key in order to merge
    df_unpivot.reset_index(inplace=True)
    return df_unpivot


def read_data(imageID, diameter=False):
    """
    :return:
    """
    x = get_data_unpivot(f"{aria_measurements_dir}/{imageID}_all_center2Coordinates.tsv")
    y = get_data_unpivot(f"{aria_measurements_dir}/{imageID}_all_center1Coordinates.tsv")
    df_all_seg = pd.read_csv(f"{aria_measurements_dir}/{imageID}_all_segmentStats.tsv", sep='\t')
    df_all_seg.reset_index(inplace=True)
    if diameter:
        diameters = get_data_unpivot(f"{aria_measurements_dir}/{imageID}_all_rawDiameters.tsv")
        df_merge = pd.merge(
            x, y,
            how='outer',
            on=['index', 'level_0']).merge(
            diameters,
            how='outer',
            on=['index', 'level_0']).merge(
            df_all_seg[['index', "AVScore"]],
            how='outer',
            on='index',
            indicator=True).rename(columns={'value_x': 'X', 'value_y': 'Y', 'value': 'Diameter', 'AVScore': 'type'})
        df_merge['X'] = pd.to_numeric(df_merge['X'])
        df_merge['Y'] = pd.to_numeric(df_merge['Y'])
        df_merge['Diameter'] = pd.to_numeric(df_merge['Diameter'])

        return df_merge
    else:
        df_merge = pd.merge(
            x, y, how='outer', on=['index', 'level_0']).merge(
            df_all_seg[['index', "AVScore"]],
            how='outer',
            on='index',
            indicator=True).rename(columns={'value_x': 'X', 'value_y': 'Y', 'AVScore': 'type'})

    df_merge['X'] = pd.to_numeric(df_merge['X'])
    df_merge['Y'] = pd.to_numeric(df_merge['Y'])
    
    return df_merge


##### Measure main phenotypes functions: (main_* functions are those that return the measures of the phenotypes)

def main_tva_or_taa(imgname_and_filter: str and int) -> dict:
    """
    :param imgname_and_filter:
    :return:
    """
    try:
        imgname = imgname_and_filter[0]
        filter_type = imgname_and_filter[1]
        imageID = imgname.split(".")[0]
        df_vasculature = read_data(imageID, diameter=True)
        df_vasculature['type'] = np.sign(df_vasculature['type'])
        OD_position = df_OD[df_OD['image'] == imgname]
        return {
            'mean_angle' : None
            if OD_position.empty
            else compute_mean_angle(df_vasculature, OD_position, filter_type)
        }

    except Exception as e:
        print(e)
        return {'mean_angle': np.nan}


def main_CRAE_CRVE(imgname_and_filter: str and int) -> dict:
    """
    :param imgname_and_filter:
    :return:
    """
    try:
        imgname = imgname_and_filter[0]
        filter_type = imgname_and_filter[1]
        imageID = imgname.split(".")[0]
        df_vasculature = read_data(imageID, diameter=True)
        df_vasculature['type'] = np.sign(df_vasculature['type'])
        OD_position = df_OD[df_OD['image'] == imgname]
        if OD_position.empty:
            return {
            'median_CRE': None,
            'eq_CRE': None,
            'CRE': None
            }
        median_CRE, eq_CRE = compute_CRE(df_vasculature, OD_position, filter_type, imageID)
        return {
            'median_CRE': median_CRE,
            'eq_CRE': eq_CRE,
            'CRE': CRE}

    except Exception as e:
        print(imgname, e)
        return {'median_CRE': np.nan, 'eq_CRE': np.nan, 'CRE': np.nan} 


def main_N_main_vessels(imgname_and_filter: str and int) -> dict:
    """
    :param imgname_and_filter:
    :return:
    """
    try:
        imgname = imgname_and_filter[0]
        filter_type = imgname_and_filter[1]
        imageID = imgname.split(".")[0]
        df_vasculature = read_data(imageID, diameter=True)
        df_vasculature['type'] = np.sign(df_vasculature['type'])
        OD_position = df_OD[df_OD['image'] == imgname]
        if OD_position.empty:
            return {
                    'N_median_main': None,
                    #'N_std_main': None#,
                    #'N_CVMe_main_'+str(trait_name): None,
                    #'N_CVP_main_'+str(trait_name): None
                    }
        N_main_v, N_std_v = compute_N_main_v(df_vasculature, OD_position, filter_type, imageID)
        if N_main_v in [0, 1, '0', '1']:
            N_main_v=np.nan
            N_std_v=np.nan

        return {
                'N_median_main': N_main_v,
                #'N_std_main': N_std_v #,
                #'N_CVMe_main_'+str(trait_name): N_median_CVMe,
                #'N_CVP_main_'+str(trait_name): N_median_CVP
                }

    except Exception as e:
        print(imgname, e)
        return {'N_median_main': np.nan} 


def main_bifurcations(imgname: str) -> dict:
    """
    :param imgname:
    :return:
    """
    try:
        imageID = imgname.split(".")[0]
        df_vasculature = read_data(imageID)
        aux = df_vasculature.groupby('index')
        df_segments = pd.concat([aux.head(1), aux.tail(1)]).drop_duplicates().sort_values('index').reset_index(drop=True)
        df_segments['type'] = np.sign(df_segments['type'])
        df_segments.sort_values(by=['X'], inplace=True, ascending=False)
        return {'bifurcations': float(bifurcation_counter(df_segments, imageID))}

    except Exception as e:
        print(imgname, e)
        return {'bifurcations': np.nan}


def main_diameter_variability(imgname: str) -> dict:
    """
    :param imgname_and_filter:
    :return:
    """
    try:
        imageID = imgname.split(".")[0]
        df_vasculature = read_data(imageID, diameter=True)
        df_vasculature['type'] = np.sign(df_vasculature['type'])
        # std of the diameters per index, i.e. std of each segment
        #std_values_vessels=df_vasculature.groupby(['index'])['Diameter'].std()
        
        ## For D_median_CVMe: # median of the diameters per index, i.e. median of each segment
        median_values_vessels = df_vasculature.groupby(['index'])['Diameter'].median()
        Dev_median = np.sum(abs(median_values_vessels - 
                             median_values_vessels.median()))/(len(median_values_vessels)- 
                                                            median_values_vessels.isnull().sum())
        ## For D_A_std_std and D_V_std_std:
        type_std_values_vessels = df_vasculature.groupby(['type'])['Diameter'].std()
        ## For D_CVMe: # median of the diameters
        diameters_vessels = df_vasculature['Diameter']
        Dev = np.sum(abs(diameters_vessels - 
                             diameters_vessels.median()))/(len(diameters_vessels)- 
                                                            diameters_vessels.isnull().sum())
        ## For D_CVMe_A and  D_CVMe_V: 
        df_A = df_vasculature.query("type == 1")
        A_diameters_vessels = df_A['Diameter']
        Dev_A = np.sum(abs(A_diameters_vessels - A_diameters_vessels.median())) / (
                    len(A_diameters_vessels) - A_diameters_vessels.isnull().sum())

        df_V = df_vasculature.query("type == -1")
        V_diameters_vessels = df_V['Diameter']
        Dev_V = np.sum(abs(V_diameters_vessels - V_diameters_vessels.median())) / (
                    len(V_diameters_vessels) - V_diameters_vessels.isnull().sum())
        return {'D_median_CVMe': Dev_median/abs(median_values_vessels.median()), 
                'D_CVMe': Dev/abs(diameters_vessels.median()), 
                'D_CVMe_A': Dev_A/abs(A_diameters_vessels.median()), 
                'D_CVMe_V': Dev_V/abs(V_diameters_vessels.median()), 
                'D_std': df_vasculature['Diameter'].std(),
                #'D_median_CVP': median_values_vessels.std()/abs(median_values_vessels.mean()),
                #'D_std_median': std_values_vessels.median(), 
                #'D_std_std': std_values_vessels.std(),
                'D_A_std': type_std_values_vessels[1],
                'D_V_std': type_std_values_vessels[-1]
               }
    
    except Exception as e:
        print(imageID, e)
        return {'D_median_CVMe': np.nan, 
                'D_CVMe': np.nan,
                'D_CVMe_A': np.nan,
                'D_CVMe_V': np.nan,
                'D_std':  np.nan, 
                #'D_median_CVP': np.nan,
                #'D_std_median': np.nan,  
                #'D_std_std': np.nan,
                'D_A_std': np.nan, 
                'D_V_std':np.nan
               }


def main_aria_phenotypes(imgname):    # still need to modify it
    """
    :param imgname:
    :return:
    """
    imageID = imgname.split(".")[0]
    lengthQuints = [23.3, 44.3, 77.7, 135.8]
    all_medians = []
    artery_medians = []
    vein_medians = []
    try:  # because for any image passing QC, ARIA might have failed
        # df is segment stat file
        df = pd.read_csv(aria_measurements_dir + imageID + "_all_segmentStats.tsv", delimiter='\t')
        all_medians = df.median(axis=0).values
        artery_medians = df[df['AVScore'] > 0].median(axis=0).values
        vein_medians = df[df['AVScore'] < 0].median(axis=0).values
        # stats based on longest fifth
        try:
            quintStats_all = df[df['arcLength'] > lengthQuints[3]].median(axis=0).values
            quintStats_artery = df[(df['arcLength'] > lengthQuints[3]) & (df['AVScore'] > 0)].median(axis=0).values
            quintStats_vein = df[(df['arcLength'] > lengthQuints[3]) & (df['AVScore'] < 0)].median(axis=0).values
        except Exception as e:
            print(imgname, e)
            print("longest 5th failed")
            quintStats_all = [np.nan for _ in range(14)]
            quintStats_artery = quintStats_all
            quintStats_vein = quintStats_all
        df_im = pd.read_csv(aria_measurements_dir + imageID + "_all_imageStats.tsv", delimiter='\t')
        return np.concatenate((all_medians, artery_medians, vein_medians, quintStats_all, \
                               quintStats_artery, quintStats_vein, df_im['nVessels'].values), axis=None).tolist()
    except Exception as e:
        print(imgname, e)
        print("ARIA didn't have stats for img", imageID)
        return [np.nan for _ in range(84)]


def main_fractal_dimension(imgname: str) -> dict:
    """
    :param imgname:
    :return:
    """
    imageID = imgname.split(".")[0]
    try:
        img = Image.open(imageID + ".png")#"_bin_seg.png") Modified
        img_artery = replaceRGB(img, (255, 0, 0), (0, 0, 0))
        img_vein = replaceRGB(img, (0, 0, 255), (0, 0, 0))
        w, h = img.size
        box_sidelengths = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        N_boxes, N_boxes_artery, N_boxes_vein = [], [], []
        for i in box_sidelengths:
            w_i = round(w / i)
            h_i = round(h / i)
            img_i = img.resize((w_i, h_i), resample=PIL.Image.BILINEAR)
            img_i_artery = img_artery.resize((w_i, h_i), resample=PIL.Image.BILINEAR)
            img_i_vein = img_vein.resize((w_i, h_i), resample=PIL.Image.BILINEAR)
            N_boxes.append(np_nonBlack(np.asarray(img_i)))
            N_boxes_artery.append(np_nonBlack(np.asarray(img_i_artery)))
            N_boxes_vein.append(np_nonBlack(np.asarray(img_i_vein)))

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log([1 / i for i in box_sidelengths]),
                                                                       np.log(N_boxes))
        slope_artery, intercept, r_value, p_value, std_err = stats.linregress(np.log([1 / i for i in box_sidelengths]),
                                                                              np.log(N_boxes_artery))
        slope_vein, intercept, r_value, p_value, std_err = stats.linregress(np.log([1 / i for i in box_sidelengths]),
                                                                            np.log(N_boxes_vein))
        return {
            'FD_all': float(slope),
            'FD_artery': float(slope_artery),
            'FD_vein': float(slope_vein)
        }

    except Exception as e:
        print(imgname, e)
        return {'FD_all': np.nan, 'FD_artery': np.nan, 'FD_vein': np.nan }


def main_vascular_density(imgname: str) -> dict:
    """
    :param imgname:
    :return:
    """
    scale_factor = 100/660 # fraction smaller compared to original
    imageID = imgname.split(".")[0]
    try:
        img = cv2.imread(imageID + ".png")
        gray=np.maximum(img[:,:,0], img[:,:,2])
        gray=np.stack((gray,gray,gray),axis=2)
        #print(gray.shape)
        img_mskd = mask_image(img, to_gray=False)
        img_mskd_gray = mask_image(gray, to_gray=False)
        #print([round(i * scale_factor) for i in img_mskd.shape[0:2]])
        img_small = cv2.resize(img, [round(i * scale_factor) for i in img_mskd.shape[1::-1]])
        gray_small = cv2.resize(gray, [round(i * scale_factor) for i in img_mskd.shape[1::-1]])
        #plt.imsave("/SSD/home/michael/gray_small_"+imageID+".png", gray_small)
        #plt.imsave("/SSD/home/michael/gray_"+imageID+".png", gray)
        #plt.imsave("/SSD/home/michael/orig_"+imageID+".png", img)
        #plt.imsave("/SSD/home/michael/small_"+imageID+".png", img_small)
        #gray_small = np.maximum(img_small[:,:,0],img_small[:,:,2])
        #gray_small = np.stack((gray_small,gray_small,gray_small),axis=2)
        img_mskd_small = mask_image(img_small, to_gray=False, mask_radius=round(mask_radius*scale_factor))
        img_mskd_gray_small = mask_image(gray_small, to_gray=False, mask_radius=round(mask_radius*scale_factor))

        area = mask_radius**2 * np.pi
        area_small = (mask_radius * scale_factor)**2 * np.pi

        vd_orig_all = np.mean(img_mskd_gray) / 255
        vd_orig_artery = np.mean(img_mskd[:,:,2]) / 255
        vd_orig_vein = np.mean(img_mskd[:,:,0]) / 255
        
        vd_small_all = np.mean(img_mskd_gray_small) / 255
        vd_small_artery = np.mean(img_mskd_small[:,:,2]) / 255
        vd_small_vein = np.mean(img_mskd_small[:,:,0]) / 255
        return { 'VD_orig_all': vd_orig_all, 'VD_orig_artery': vd_orig_artery, 'VD_orig_vein': vd_orig_vein }
                 #'VD_small_all': vd_small_all, 'VD_small_artery': vd_small_artery, 'VD_small_vein': vd_small_vein }

    except Exception as e:
        print(imgname, e)
        return { 'VD_orig_all': np.nan, 'VD_orig_artery': np.nan, 'VD_orig_vein': np.nan }
                 #'VD_small_all': np.nan, 'VD_small_artery': np.nan, 'VD_small_vein': np.nan }


def main_baseline_traits(imgname: str) -> dict:
    """
    :param imgname_and_filter:
    :return:
    """
    try:
        imageID = imgname.split(".")[0]
        # Next step: Include only the pixels inside the mask? Save the masks from LWnet?
        img = io.imread(imageID + '.png')
        # print(img.shape)
        return {'std_intensity': np.std(img), 'mean_intensity': np.mean(img)}#, 'median_intensity': np.median(img)}

    except Exception as e:
        print(imgname, e)
        return {'std_intensity': np.nan, 'mean_intensity': np.nan}#, 'median_intensity': np.nan}


#### Intermediate measuring phenotypes functions

def compute_N_main_v(df_vasculature, OD_position, filter_type, imageID) -> tuple:
    """
    :param df_vasculature:
    :param OD_position:
    :param filter_type:
    :return:
    """
    Number_main_vessels = get_intersections_N_main_v(df_vasculature, OD_position, filter_type, imageID)
    #Dev_median = sum(abs(Number_main_vessels - Number_main_vessels.median()))/(len(Number_main_vessels)- Number_main_vessels.isnull().sum())
    return Number_main_vessels.median()[0], Number_main_vessels.std()[0]#, Dev_median/abs(Number_main_vessels.median()), Number_main_vessels.std()/abs(Number_main_vessels.mean())


def get_intersections_N_main_v(df_vasculature, OD_position, filter_type, imageID):
    """
    :param df_vasculature:
    :param OD_position:
    :param filter_type:
    :return:
    """
    angle = np.arange(0, 360, 0.01)
    aux = []
    df_vasculature['X'] = df_vasculature['X'].round(0)
    df_vasculature['Y'] = df_vasculature['Y'].round(0)
    for p in radius: 
        df_intersection = circular_df_filter(p, angle, OD_position, df_vasculature)
        df_specific_vessel_type_intersection = df_intersection[df_intersection["type"] == filter_type]
        #df_specific_vessel_type_intersection.sort_values(by=['Diameter'], ascending=False, inplace=True)
        df_specific_vessel_type_intersection = df_specific_vessel_type_intersection[df_specific_vessel_type_intersection['Diameter']>limit_diameter_main]
        data = {'N_main_vessels': df_specific_vessel_type_intersection.shape[0]}
        aux.append(data)
    return pd.DataFrame(aux)


def compute_CRE(df_vasculature, OD_position, filter_type, imageID):
    """
    :param df_vasculature:
    :param OD_position:
    :param filter_type:
    :return:
    """
    df_specific_vessel_type_intersection, CRE_eq, CRE_standard = get_intersections(df_vasculature, OD_position, filter_type, imageID)

    df_specific_vessel_type_intersection_median =df_specific_vessel_type_intersection.median()
    CRE_eq_median = CRE_eq.median()
    #print('statistics.median(df_specific_vessel_type_intersection)', statistics.median(df_specific_vessel_type_intersection))
    return df_specific_vessel_type_intersection_median[0], CRE_eq_median[0], CRE_standard


def get_intersections(df_vasculature, OD_position, filter_type):
    """
    :param df_vasculature:
    :param OD_position:
    :param filter_type:
    :return:
    """
    angle = np.arange(0, 360, 0.01)
    aux = []
    aux_eq = []

    num_segments_selected=3 ## We only select 3 because we only analyze temporal

    if filter_type==-1: 
        factor_cte=0.95 # From literature 
    else: 
        factor_cte=0.88 # From literature

    df_vasculature['X'] = df_vasculature['X'].round(0)
    df_vasculature['Y'] = df_vasculature['Y'].round(0)
    r1 = (OD_position['width'] + OD_position['height'])/4
    radius_r1 = [2*r1.iloc[0], 2.1*r1.iloc[0], 2.2*r1.iloc[0], 2.3*r1.iloc[0], 2.4*r1.iloc[0], 2.5*r1.iloc[0]]

    ## Compute the most common definition of CRAE and CRAE
    radius1 =  radius_r1[-1]
    radius0 = radius_r1[0]

    df_intersection_radius0 = circular_df_filter(radius0, angle, OD_position, df_vasculature)
    df_specific_vessel_type_intersection_radius0 = df_intersection_radius0[df_intersection_radius0["type"] == filter_type].copy()
    df_specific_vessel_type_intersection_radius0.sort_values(by=['Diameter'], ascending=False, inplace=True)

    df_specific_vessel_type_intersection_radius0 = df_specific_vessel_type_intersection_radius0.head(num_segments_selected) 
    df_vasculature_index = df_vasculature[df_vasculature['index'].isin(list(df_specific_vessel_type_intersection_radius0['index']))]
    df_vasculature_annulus = compute_vessel_radius_pixels(df_vasculature_index, radius1,  radius0, OD_position)
    #print(df_pintar_annulus)
    #plt.scatter(df_vasculature_annulus['X'], df_vasculature_annulus['Y'], color='pink', s=20)
    df_annulus_diameter = df_vasculature_annulus.groupby(['index'])['Diameter'].median()
    df_annulus_diameter.sort_values(ascending=False, inplace=True)
    equation_standard_CRE = factor_cte*((df_annulus_diameter.iloc[0])**2 + (df_annulus_diameter.iloc[num_segments_selected-1])**2)**(1/2)


    ## Compute alternative definition of CRAE and CRAE
    for p in radius_r1: 
        df_intersection = circular_df_filter(p, angle, OD_position, df_vasculature)
        df_specific_vessel_type_intersection = df_intersection[df_intersection["type"] == filter_type].copy()
        df_specific_vessel_type_intersection.sort_values(by=['Diameter'], ascending=False, inplace=True)
        # Two options: Select only those with less than a certain diameter value, or select the N first 
        num_segments_selected=3 # Option 1
        df_specific_vessel_type_intersection = df_specific_vessel_type_intersection.head(num_segments_selected) 
        D_median = df_specific_vessel_type_intersection['Diameter'].median()
        data = {'modif_CRE': D_median}
        aux.append(data)
        #limit_diameter = 2 # Option 2
        #df_specific_vessel_type_intersection = df_specific_vessel_type_intersection[df_specific_vessel_type_intersection['Diameter']>limit_diameter]
        equation_CRA = factor_cte*((df_specific_vessel_type_intersection['Diameter'].iloc[0])**2 + (df_specific_vessel_type_intersection['Diameter'].iloc[num_segments_selected-1])**2)**(1/2)
        data_eq = {'CRE': equation_CRA}
        aux_eq.append(data_eq)

        ### Save the position of the intersections per image: 
        #df_save = df_save.append(df_specific_vessel_type_intersection)
    
    ### Save the position of the intersections per image:
    #dir_CRE_position=phenotype_dir + 'CR'+X+'E_position/'
    #if not os.path.exists(dir_CRE_position):
    #    os.mkdir(dir_CRE_position)
    #df_save.to_csv(dir_CRE_position +'/' + imageID + '_CR'+X+'E_position.csv', index=False)

    return pd.DataFrame(aux), pd.DataFrame(aux_eq), equation_standard_CRE #,statistics.median(auxiliar_values_eq) #pd.DataFrame(auxiliar_values)#, statistics.median(auxiliar_values_eq)


def mask_image(img, to_gray=False, mask_radius=mask_radius):
    hh,ww = img.shape[:2]
    #print(hh//2,ww//2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    mask = cv2.circle(mask, (ww//2,hh//2), mask_radius, (255,255,255), -1)
    #mask = np.invert(mask.astype(bool))
    #result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    #result[:, :] = mask[:,:]
    #  result[:, :, 3] = mask[:,:,0]
    #plt.imshow(result)
    mask_1d = mask.astype(bool)    
    mask_3d = np.stack((mask_1d,mask_1d,mask_1d), axis=2) # axis=2 to make color channels 3rd dimension
    
    if to_gray == True:
        return np.ma.array(gray, mask=np.invert(mask_1d))
    else:
        return np.ma.array(img, mask=np.invert(mask_3d))


def ang(lineA, lineB) -> float:
    """
    :param lineA:
    :param lineB:
    :return:
    """
    vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
    vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
    # Get dot prod
    dot_prod = np.dot(vA, vB)
    # Get magnitudes
    magA = np.dot(vA, vA) ** 0.5
    magB = np.dot(vB, vB) ** 0.5
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod / magB / magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle) % 360
    return 360 - ang_deg if ang_deg >= 180 else ang_deg


def np_nonBlack(img):
    return img.any(axis=-1).sum()


def replaceRGB(img, old, new):
    """
    :param img:
    :param old:
    :param new:
    :return:
    """
    out = img.copy()
    datas = out.getdata()
    newData = []
    for item in datas:
        if item[0] == old[0] and item[1] == old[1] and item[2] == old[2]:
            newData.append((new[0], new[1], new[2]))
        else:
            newData.append(item)
    out.putdata(newData)
    return out


def bifurcation_counter(df_segments, imageID):
    """
    :param df_segments:
    :return:
    """
    X_1_aux = X_2_aux = 0.0
    bif_counter = 0
    aux = []
    df_bif_positions = pd.DataFrame([])
    number_rows = df_segments.shape[0]
    x = df_segments['X'].values
    y = df_segments['Y'].values
    dis_type = df_segments['type'].values
    index_v = df_segments['index'].values

    for s in range(number_rows):
        for j in range(number_rows - s):
            j = j + s
            # For X and Y: X[s] - neighborhood_cte <= X[j] <= X[s]
            # From different segmentes and Both arteries or both veins
            if (
                (x[j] >= x[s] - neighborhood_cte)
                and (x[j] <= x[s] + neighborhood_cte)
                and (y[j] >= y[s] - neighborhood_cte)
                and (y[j] <= y[s] + neighborhood_cte)
                and index_v[j] != index_v[s] # From different segmentes
                and (dis_type[j] == dis_type[s]) # Both arteries or both veins
                and ( # With this condition we avoid having most of the repetitions 
                    x[j] != X_1_aux
                    and x[s] != X_1_aux
                    and x[j] != X_2_aux
                    and x[s] != X_2_aux
                    )):
                X_1_aux = x[s]
                X_2_aux = x[j]
                data = {
                    'X': x[j],
                    'Y': y[j],
                    'type': dis_type[j],
                    'i': index_v[j]}
                aux.append(data)

    ## Correct for the few repetitions remained
    df_bif_positions = pd.DataFrame(aux)
    df_bif_positions = delete_points_very_close(df_bif_positions, norm_acceptance)
    bif_counter=len(df_bif_positions)

    # Save the position of the bifurcations per image
    dir_bif_position=phenotype_dir + 'bifurcations_position/'
    if not os.path.exists(dir_bif_position):
        os.mkdir(dir_bif_position)
    df_bif_positions.to_csv(dir_bif_position +'/' + imageID +'_bifurcations_position.csv', index=False)

    return bif_counter


def delete_points_very_close(df_bif_positions, norm_acceptance):
    #compute distance 
    df_bif_aux= pd.DataFrame([])
    df_bif_aux['X']=df_bif_positions['X']
    df_bif_aux['Y']=df_bif_positions['Y']

    distance=pd.DataFrame(distance_matrix(df_bif_aux.values, df_bif_aux.values), index=df_bif_aux.index, columns=df_bif_aux.index)
    distance = pd.DataFrame(np.tril(distance))
    distance = distance.replace(0, np.nan)
    auxiliar=distance[distance < norm_acceptance]
    auxiliar=auxiliar.unstack().reset_index()
    auxiliar.columns=['row','column','Norm']
    cols = auxiliar.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    auxiliar = auxiliar[cols]
    auxiliar = auxiliar.dropna()
    df_bif_positions = df_bif_positions.drop(auxiliar['row'].to_list())
    return df_bif_positions


def circular_df_filter(ratio, angle, od_position, df_vasculature):
    """
    :param ratio:
    :param angle:
    :param od_position:
    :param df_vasculature:
    :return:
    """
    df_circle = compute_circular_df(ratio, angle, od_position)
    df_merge_aux = pd.merge(df_circle, df_vasculature, how='inner', left_on=['X', 'Y'], right_on=['X', 'Y'])
    return df_merge_aux.drop_duplicates(subset=['index'], keep='last')


def compute_circular_df(ratio, angle, od_position):
    """
    :param ratio:
    :param angle:
    :param od_position:
    :return:
    """
    x = ratio * np.cos(angle) + od_position['x'].iloc[0]
    y = ratio * np.sin(angle) + od_position['y'].iloc[0]
    df_circle = pd.DataFrame([])
    df_circle['X'] = x.round(0)
    df_circle['Y'] = y.round(0)
    return df_circle


def compute_potential_vein_arteries(df_specific_vessel_type_intersection, od_position):
    """
    :param df_specific_vessel_type_intersection:
    :param od_position:
    :return:
    """
    aux = []
    veins_art_x = df_specific_vessel_type_intersection['X'].values
    veins_art_y = df_specific_vessel_type_intersection['Y'].values
    veins_art_index = df_specific_vessel_type_intersection['index'].values
    veins_art_diameter = df_specific_vessel_type_intersection['Diameter'].values
    veins_art_type = df_specific_vessel_type_intersection['type'].values
    for i, j in itertools.product(range(df_specific_vessel_type_intersection.shape[0] - 1), range(df_specific_vessel_type_intersection.shape[0] - 2)):
        lineA = ((od_position['x'].iloc[0], od_position['y'].iloc[0]), (veins_art_x[i], veins_art_y[i]))
        lineB = ((od_position['x'].iloc[0], od_position['y'].iloc[0]), (veins_art_x[j], veins_art_y[j]))
        if i == j:
            continue
        angle = ang(lineA, lineB)
        angle = round(angle, 0)
        data = {
            'X_1': veins_art_x[i],
            'Y_1': veins_art_y[i],
            'Diameter_1': veins_art_diameter[i],
            'type_1': veins_art_type[i],
            'i_1': veins_art_index[i],
            'X_2': veins_art_x[j],
            'Y_2': veins_art_y[j],
            'Diameter_2': veins_art_diameter[j],
            'type_2': veins_art_type[j],
            'i_2': veins_art_index[j],
            'angle': angle
        }
        aux.append(data)
    return pd.DataFrame(aux)


def get_main_angle_row(df_potential_points):
    """
    :param df_potential_points:
    :return:
    """
    d = {'X_1': 0, 'Y_1': 0, 'Diameter_1': 0, 'type_1': 0, 'i_1': 0, 'X_2': 0, 'Y_2': 0, 'Diameter_2': 0, 'type_2': 0,
         'i_2': 0, 'angle': 0}
    main_angle = pd.Series(data=d,
                           index=['X_1', 'Y_1', 'Diameter_1', 'type_1', 'i_1', 'X_2', 'Y_2', 'Diameter_2', 'type_2',
                                  'i_2', 'angle'])
    if not df_potential_points.empty:
        df_final_angles = df_potential_points[(df_potential_points["angle"] >= min_ta) & (df_potential_points["angle"] <= max_ta)].copy()
        df_final_angles = df_final_angles.sort_values(['Diameter_1', 'Diameter_2'], ascending=[False, False])
        if not df_final_angles.empty:
            main_angle = df_final_angles.iloc[0]
    return main_angle


def get_data_angle(df_potential_points):
    """
    :param df_potential_points:
    :return:
    """
    main_angle_row = get_main_angle_row(df_potential_points)
    return {
        'X_1': main_angle_row['X_1'],
        'Y_1': main_angle_row['Y_1'],
        'Diameter_1': main_angle_row['Diameter_1'],
        'X_2': main_angle_row['X_2'],
        'Y_2': main_angle_row['Y_2'],
        'Diameter_2': main_angle_row['Diameter_2'],
        'angle': main_angle_row['angle']
    }


def get_ratious_votes(df_vasculature, OD_position, filter_type):
    """
    :param df_vasculature:
    :param OD_position:
    :param filter_type:
    :return:
    """
    angle = np.arange(0, 360, 0.01)
    df_vasculature['X'] = df_vasculature['X'].round(0)
    df_vasculature['Y'] = df_vasculature['Y'].round(0)
    auxiliar_angle = []
    for p in radius:
        df_intersection = circular_df_filter(p, angle, OD_position, df_vasculature)
        df_specific_vessel_type_intersection = df_intersection[df_intersection["type"] == filter_type].copy()
        df_specific_vessel_type_intersection.sort_values(by=['Diameter'], ascending=False, inplace=True)
        df_potential_points = compute_potential_vein_arteries(df_specific_vessel_type_intersection, OD_position)
        auxiliar_angle.append(get_data_angle(df_potential_points))
    return pd.DataFrame(auxiliar_angle)


def get_angle_mode(df_final_vote):
    """
    :param df_final_vote:
    :return:
    """
    for i in range(len(df_final_vote) - 1):
        for j in range(len(df_final_vote)):
            if (df_final_vote['angle'].loc[i + 1] >= df_final_vote['angle'].loc[j] - lower_accept) and (
                    df_final_vote['angle'].loc[i + 1] <= df_final_vote['angle'].loc[j] + upper_accept):
                #print("get_angle_mode, if passes at i,j =", str(i), str(j))
                df_final_vote['vote_angle'].loc[i + 1] = j
                break
    return df_final_vote[df_final_vote['vote_angle'] == df_final_vote.mode()['vote_angle'][0]].copy()


def compute_mean_angle_with_mode(df_final_vote):
    """
    :param df_final_vote:
    :return:
    """
    df_final = get_angle_mode(df_final_vote)
    return (
        df_final['angle'].mean()
        if df_final.shape[0] >= 3 and df_final['angle'].mean() != 0.0
        else None
    )


def compute_mean_angle(df_vasculature, OD_position, filter_type): #filter_type=-1):
    """
    :param df_vasculature:
    :param OD_position:
    :param filter_type:
    :return:
    """
    df_final_vote = get_ratious_votes(df_vasculature, OD_position, filter_type)
    df_final_vote = df_final_vote.reset_index().rename(columns={'index': 'vote_angle'}).copy()
    return compute_mean_angle_with_mode(df_final_vote)


def compute_vessel_radius_pixels(df_vasculature, radius, radius0, od_position):
    """
    :param df_vasculature:
    :param radius:
    :param od_position:
    :return:
    """
    df_vasculature['DeltaX'] = df_vasculature['X'] - od_position['x'].iloc[0]
    df_vasculature['DeltaY'] = df_vasculature['Y'] - od_position['y'].iloc[0]
    df_vasculature['r2_value'] = df_vasculature['DeltaX'] * df_vasculature['DeltaX'] + df_vasculature['DeltaY'] * df_vasculature['DeltaY']
    df_vasculature['r_value'] = (df_vasculature['r2_value']) ** (1 / 2)
    df_inside_radius = df_vasculature[df_vasculature['r_value'] <= radius].copy()
    
    return df_inside_radius[df_inside_radius['r_value'] >= radius0]



##### Trait output data main functions:

def create_output_(out, imgfiles, function_to_execute, imgfiles_length):
    """
    :param out:
    :param imgfiles:
    :param function_to_execute:
    :param imgfiles_length:
    :return:
    """
    output_path = os.path.join(
        phenotype_dir,
        f'{datetime.now().strftime("%Y-%m-%d")}_{function_to_execute}.csv',
    )

    if function_to_execute == "aria_phenotypes":
        first_statsfile = pd.read_csv(aria_measurements_dir + imgfiles[0].split('.')[0]+"_all_segmentStats.tsv", sep='\t')
        cols = first_statsfile.columns
        cols_full = [i + "_all" for i in cols] + [i + "_artery" for i in cols] + [i + "_vein" for i in cols]\
        +  [i + "_longestFifth_all" for i in cols] + [i + "_longestFifth_artery" for i in cols] + [i + "_longestFifth_vein" for i in cols]\
        +  ["nVessels"]

        df = pd.DataFrame(out, columns=cols_full)
    else:
        print(out)
        df = pd.DataFrame(out)

    df = df.set_index(imgfiles[:imgfiles_length])
    df.to_csv(output_path)
    df.to_pickle(output_path.replace('.csv','.pkl'))

    print(len(df), "image measurements taken")
    print("NAs per phenotype")
    print(df.isna().sum())


def renaming_colum_names(function_to_execute, change_names=False):
    if change_names==True:
        if function_to_execute == 'taa': 
            df_data_taa = pd.read_csv(phenotype_dir+DATE+"_taa.csv", sep=',')
            df_data_taa.rename(columns={'mean_angle': 'mean_angle_taa'}, inplace=True)
            df_data_taa.to_csv(phenotype_dir + DATE + "_taa.csv", sep=',', index=False)

        elif function_to_execute == 'tva':
            df_data_tva = pd.read_csv(phenotype_dir+DATE+"_tva.csv", sep=',')
            df_data_tva.rename(columns={'mean_angle': 'mean_angle_tva'}, inplace=True)
            df_data_tva.to_csv(phenotype_dir + DATE + "_tva.csv", sep=',', index=False)
            
        elif function_to_execute == 'N_main_arteires':
            df_data_NA = pd.read_csv(phenotype_dir+DATE+"_N_main_arteires.csv", sep=',')
            df_data_NA.rename(columns={'N_median_main': 'N_median_main_arteries', 'N_std_main': 'N_std_main_arteries'}, inplace=True)
            df_data_NA.to_csv(phenotype_dir + DATE + "_N_main_arteires.csv", sep=',', index=False)
            
        elif function_to_execute == 'N_main_veins':
            df_data_NV = pd.read_csv(phenotype_dir+DATE+"_N_main_veins.csv", sep=',')
            df_data_NV.rename(columns={'N_median_main': 'N_median_main_veins', 'N_std_main': 'N_std_main_veins'}, inplace=True)
            df_data_NV.to_csv(phenotype_dir + DATE + "_N_main_veins.csv", sep=',', index=False)



###### '__main__'

if __name__ == '__main__':

    # all the images
    imgfiles = pd.read_csv(qcFile, header=None)
    imgfiles = imgfiles[0].values
    
    # development param
    imgfiles_length = len(imgfiles)  # len(imgfiles) is default

    for function_to_execute in traits:
        print("\nStarting function", function_to_execute, '\n')

        # computing the phenotype as a parallel process
        os.chdir(lwnet_dir)
        pool = Pool()

        if function_to_execute in {'taa', 'tva'}:
            filter_tva_taa = 1 if function_to_execute == 'taa' else (-1 if function_to_execute == 'tva' else None)
            imgages_and_filter = list(zip(imgfiles[:imgfiles_length], imgfiles_length * [filter_tva_taa]))
            out = pool.map(main_tva_or_taa, imgages_and_filter)
        elif function_to_execute in {'CRAE', 'CRVE'}:
            filter_CRAE_CRVE = 1 if function_to_execute == 'CRAE' else (-1 if function_to_execute == 'CRVE' else None)
            imgages_and_filter = list(zip(imgfiles[:imgfiles_length], imgfiles_length * [filter_CRAE_CRVE]))
            out = pool.map(main_CRAE_CRVE, imgages_and_filter)
        elif function_to_execute in {'N_main_arteires', 'N_main_veins'}:
            filter_N_main = 1 if function_to_execute == 'N_main_arteires' else (-1 if function_to_execute == 'N_main_veins' else None)
            imgages_and_filter = list(zip(imgfiles[:imgfiles_length], imgfiles_length * [filter_N_main]))
            out = pool.map(main_N_main_vessels, imgages_and_filter)
        elif function_to_execute == 'bifurcations':
            out = pool.map(main_bifurcations, imgfiles[:imgfiles_length])
        elif function_to_execute == 'diameter_variability':
            out = pool.map(main_diameter_variability, imgfiles[:imgfiles_length]) 
        elif function_to_execute == 'aria_phenotypes':
            out = pool.map(main_aria_phenotypes, imgfiles[:imgfiles_length])
        elif function_to_execute == 'fractal_dimension':
            out = pool.map(main_fractal_dimension, imgfiles[:imgfiles_length])
        elif function_to_execute == 'vascular_density':
            out = pool.map(main_vascular_density, imgfiles[:imgfiles_length])
        elif function_to_execute == 'baseline':
            out = pool.map(main_baseline_traits, imgfiles[:imgfiles_length]) 
        else:
            out = None

        pool.close()
        create_output_(out, imgfiles, function_to_execute, imgfiles_length) if out else print("WARNING Your function is ", function_to_execute, ".\nIf it is a ratio, it is al right, else, this function does not exist. Options:taa,tva,CRAE,CRVE,ratios_CRAE_CRVE,bifurcations,diameter_variability,aria_phenotypes,ratios,fractal_dimension,vascular_density,baseline,neo_vascularization,N_main_arteires,N_main_veins", sep='')
        
        if function_to_execute == 'ratios':  # For measure ratios as qqnorm(ratio)
            df_data = pd.read_csv(phenotype_dir+DATE+"_aria_phenotypes.csv", sep=',')
            df_data = df_data[['Unnamed: 0', 'medianDiameter_all', 'medianDiameter_artery', 'medianDiameter_vein', 'DF_all', 'DF_artery', 'DF_vein', 'DF_longestFifth_artery', 'DF_longestFifth_vein', 'medianDiameter_longestFifth_artery', 'medianDiameter_longestFifth_vein', 'tau2_longestFifth_artery', 'tau2_longestFifth_vein', 'tau3_longestFifth_artery', 'tau3_longestFifth_vein', 'tau4_longestFifth_artery', 'tau4_longestFifth_vein']]
            print(df_data)
            df_data['ratio_AV_medianDiameter'] = df_data['medianDiameter_artery'] / df_data['medianDiameter_vein']
            #df_data['ratio_VA_medianDiameter'] = df_data['medianDiameter_vein'] / df_data['medianDiameter_artery']
            df_data['ratio_AV_DF'] = df_data['DF_artery'] / df_data['DF_vein']
            #df_data['ratio_VA_DF'] = df_data['DF_vein'] / df_data['DF_artery']
            df_data['ratio_medianDiameter_longest'] = df_data['medianDiameter_longestFifth_artery'] / df_data['medianDiameter_longestFifth_vein']
            df_data['ratio_DF_longest'] = df_data['DF_longestFifth_artery'] / df_data['DF_longestFifth_vein']
            df_data['ratio_tau2_longest'] = df_data['tau2_longestFifth_artery'] / df_data['tau2_longestFifth_vein']
            df_data['ratio_tau3_longest'] = df_data['tau3_longestFifth_artery'] / df_data['tau3_longestFifth_vein']
            df_data['ratio_tau4_longest'] = df_data['tau4_longestFifth_artery'] / df_data['tau4_longestFifth_vein']
            # only select the ratios
            df_data.drop(['medianDiameter_all', 'medianDiameter_artery', 'medianDiameter_vein', 'DF_all', 'DF_artery', 'DF_vein', 'DF_longestFifth_artery', 
                          'DF_longestFifth_vein', 'medianDiameter_longestFifth_artery', 'medianDiameter_longestFifth_vein', 'tau2_longestFifth_artery', 'tau2_longestFifth_vein',
                          'tau3_longestFifth_artery', 'tau3_longestFifth_vein', 'tau4_longestFifth_artery', 'tau4_longestFifth_vein'], axis=1, inplace=True)
            df_data.to_csv(phenotype_dir + DATE + "_ratios_aria_phenotypes.csv", sep=',', index=False)
            
        elif function_to_execute == 'ratios_CRAE_CRVE':
            df_data_CRAE = pd.read_csv(phenotype_dir+DATE+"_CRAE.csv", sep=',')
            df_data_CRAE.rename(columns={ df_data_CRAE.columns[0]: "Unnamed: 0" }, inplace = True)
            df_data_CRAE.rename(columns={'median_CRE': 'median_CRAE', 'eq_CRE': 'eq_CRAE', 'CRE': 'CRAE'}, inplace=True)
            
            df_data_CRVE = pd.read_csv(phenotype_dir+DATE+"_CRVE.csv", sep=',')
            df_data_CRVE.rename(columns={ df_data_CRVE.columns[0]: "Unnamed: 0" }, inplace = True)
            df_data_CRVE.rename(columns={'median_CRE': 'median_CRVE', 'eq_CRE': 'eq_CRVE', 'CRE': 'CRVE'}, inplace=True)
            df_merge=df_data_CRAE.merge(df_data_CRVE, how='inner', on='Unnamed: 0')
            df_merge['ratio_median_CRAE_CRVE'] = df_merge['median_CRAE'] / df_merge['median_CRVE']
            df_merge['ratio_CRAE_CRVE'] = df_merge['eq_CRAE'] / df_merge['eq_CRVE']
            df_merge['ratio_standard_CRE'] = df_merge['CRAE'] / df_merge['CRVE']

            df_merge.to_csv(phenotype_dir + DATE + "_ratios_CRAE_CRVE.csv", sep=',', index=False)
        
        elif function_to_execute == 'ratios_VD':
            df_data_VD = pd.read_csv(phenotype_dir+DATE+"_vascular_density.csv", sep=',')
            df_data_VD.rename(columns={ df_data_VD.columns[0]: "Unnamed: 0" }, inplace = True)
            df_data_VD['ratio_VD'] = df_data_VD['VD_orig_artery'] / df_data_VD['VD_orig_vein']
            df_data_VD.to_csv(phenotype_dir + DATE + "_ratios_VD.csv", sep=',', index=False)
            
        #Renaming column names
        renaming_colum_names(function_to_execute, True)
