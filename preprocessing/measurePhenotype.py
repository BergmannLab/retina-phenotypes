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


aria_measurements_dir = sys.argv[3] #'/Users/sortinve/PycharmProjects/pythonProject/sofia_dev/data/ARIA_MEASUREMENTS_DIR' 
qcFile = sys.argv[1] # '/Users/sortinve/PycharmProjects/pythonProject/sofia_dev/data/noQC.txt'  # qcFile used is noQCi, as we measure for all images
phenotype_dir = sys.argv[2]
lwnet_dir = sys.argv[4] # '/Users/sortinve/PycharmProjects/pythonProject/sofia_dev/data/LWNET_DIR'  
OD_output_dir = sys.argv[5]
df_OD = pd.read_csv(OD_output_dir+"/od_all.csv", sep=',')

mask_radius=660 # works for UKBB, may be adapted in other datasets, though only used for PBV (percent annotated as blood vessels) phenotype

def main_bifurcations(imgname: str) -> dict:
    """
    :param imgname:
    :return:
    """
    try:
        imageID = imgname.split(".")[0]
        df_pintar = read_data(imageID)
        aux = df_pintar.groupby('index')
        df_results = pd.concat([aux.head(1), aux.tail(1)]).drop_duplicates().sort_values('index').reset_index(drop=True)
        df_results['type'] = np.sign(df_results['type'])
        df_results.sort_values(by=['X'], inplace=True, ascending=False)
        return {'bifurcations': float(bifurcation_counter(df_results, imageID))}

    except Exception as e:
        print(e)
        return {'bifurcations': np.nan}


def main_tva_or_taa(imgname_and_filter: str and int) -> dict:
    """
    :param imgname_and_filter:
    :return:
    """
    try:
        imgname = imgname_and_filter[0]
        filter_type = imgname_and_filter[1]
        imageID = imgname.split(".")[0]
        df_pintar = read_data(imageID, diameter=True)
        df_pintar['type'] = np.sign(df_pintar['type'])
        OD_position = df_OD[df_OD['image'] == imgname]
        OD_position.dropna(subset=['center_x_y'], inplace=True)
        return {
            'mean_angle': None
            if OD_position.empty
            else compute_mean_angle(df_pintar, OD_position, filter_type)
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
        df_pintar = read_data(imageID, diameter=True)
        df_pintar['type'] = np.sign(df_pintar['type'])
        OD_position = df_OD[df_OD['image'] == imgname]
        OD_position.dropna(subset=['center_x_y'], inplace=True)
        if OD_position.empty:
            return {
            'median_CRE': None,
            'eq_CRE': None
            }

        else:
            #median_CRE = compute_CRE(df_pintar, OD_position, filter_type)
            median_CRE, eq_CRE = compute_CRE(df_pintar, OD_position, filter_type, imageID)
            return {
                'median_CRE': median_CRE,
                'eq_CRE': eq_CRE }

    except Exception as e:
        print(e)
        return {
                'median_CRE': np.nan,
                'eq_CRE': np.nan
                } 

def compute_CRE(df_pintar, OD_position, filter_type, imageID): #filter_type=-1):
    """
    :param df_pintar:
    :param OD_position:
    :param filter_type:
    :return:
    """
    df_veins_arter, CRE_eq = get_intersections(df_pintar, OD_position, filter_type, imageID)

    df_veins_arter_median =df_veins_arter.median()
    CRE_eq_median =CRE_eq.median()
    #print('statistics.median(df_veins_arter)', statistics.median(df_veins_arter))
    return df_veins_arter_median[0], CRE_eq_median[0]

def get_intersections(df_pintar, OD_position, filter_type, imageID):
    """
    :param df_pintar:
    :param OD_position:
    :param filter_type:
    :return:
    """
    angle = np.arange(0, 360, 0.01)
    aux = []
    aux_eq = []
    df_pintar['X'] = df_pintar['X'].round(0)
    df_pintar['Y'] = df_pintar['Y'].round(0)
    r1 = (OD_position['width'] + OD_position['height'])/4
    radius = [2*r1.iloc[0], 2.5*r1.iloc[0]]
    for p in radius: 
        new_df_2 = circular_df_filter(p, angle, OD_position, df_pintar)
        df_veins_arter = new_df_2[new_df_2["type"] == filter_type]
        df_veins_arter.sort_values(by=['Diameter'], ascending=False, inplace=True)
        # Two options: Select only those with less than a certain diameter value, or select the N first 
        num_segments_selected=3
        df_veins_arter = df_veins_arter.head(num_segments_selected) 
        D_median = df_veins_arter['Diameter'].median()
        data = {'modif_CRE': D_median}
        aux.append(data)
        #limit_diameter = 2 # Select it better 
        #df_veins_arter = df_veins_arter[df_veins_arter['Diameter']>limit_diameter]
        if filter_type==-1: 
            cte=0.95
            X ='A'
        else: 
            cte=0.88
            X ='V'
        equation_CRA = cte*((df_veins_arter['Diameter'].iloc[0])**2 + (df_veins_arter['Diameter'].iloc[num_segments_selected-1])**2)**(1/2)
        data_eq = {'CRE': equation_CRA}
        aux_eq.append(data_eq)

        # Save the position of the intersections per image: TO MODIFY
        #df_save = df_save.append(df_veins_arter)
    
    # Save the position of the intersections per image: TO MODIFY
    #dir_CRE_position=phenotype_dir + 'CR'+X+'E_position/'
    #if not os.path.exists(dir_CRE_position):
    #    os.mkdir(dir_CRE_position)
    #df_save.to_csv(dir_CRE_position +'/' + imageID + '_CR'+X+'E_position.csv', index=False)

    return pd.DataFrame(aux), pd.DataFrame(aux_eq) #,statistics.median(auxiliar_values_eq) #pd.DataFrame(auxiliar_values)#, statistics.median(auxiliar_values_eq)



def diameter_variability(imgname: str) -> dict:
    """
    :param imgname_and_filter:
    :return:
    """
    try:
        imageID = imgname.split(".")[0]
        print('imageID', imageID)
        df_pintar = read_data(imageID, diameter=True)
        df_pintar['type'] = np.sign(df_pintar['type'])
        std_values=df_pintar.groupby(['index'])['Diameter'].std()
        print('std_values.median()', std_values.median())
        median_values_vessels = df_pintar.groupby(['type'])['Diameter'].median()
        print('median_values_vessels', median_values_vessels)
        mean_values_vessels = df_pintar.groupby(['type'])['Diameter'].mean()
        print('mean_values_vessels', mean_values_vessels)
        std_values_vessels = df_pintar.groupby(['type'])['Diameter'].std()
        print('std_values_vessels', std_values_vessels)

        return {'D_median_std': std_values.median(), 'D_mean_std': std_values.mean(), 'D_std_std': std_values.std(),
        'D_A_median_std': median_values_vessels[1], 'D_A_mean_std': mean_values_vessels[1], 'D_A_std_std': std_values_vessels[1],
        'D_V_median_std': median_values_vessels[-1], 'D_V_mean_std': mean_values_vessels[-1], 'D_V_std_std': std_values_vessels[-1]}
        #'D_G_median_std': median_values_vessels[0], 'D_G_mean_std': mean_values_vessels[0], 'D_G_std_std': std_values_vessels[0]} ## only for green segments

    except Exception as e:
        print(e)
        return {'D_median_std': np.nan, 'D_mean_std': np.nan, 'D_std_std': np.nan, 
        'D_A_median_std': np.nan, 'D_A_mean_std': np.nan, 'D_A_std_std': np.nan,
        'D_V_median_std': np.nan, 'D_V_mean_std': np.nan, 'D_V_std_std': np.nan}
        #'D_G_median_std': np.nan, 'D_G_mean_std': np.nan, 'D_G_std_std': np.nan} ## only for green segments


def baseline_traits(imgname: str) -> dict:
    """
    :param imgname_and_filter:
    :return:
    """
    try:
        imageID = imgname.split(".")[0]
        # Next step: Include only the pixels inside the mask? Save the masks from LWnet?
        img = io.imread(imageID + '.png')
        # print(img.shape)
        return {'std_intensity': np.std(img), 'mean_intensity': np.mean(img), 'median_intensity': np.median(img)}

    except Exception as e:
        print(e)
        return {'std_intensity': np.nan, 'mean_intensity': np.nan, 'median_intensity': np.nan}


def main_neo_vascularization_od(imgname: str) -> dict:
    """
    :param imgname:
    :return:
    """
    try:
        imageID = imgname.split(".")[0]
        df_pintar = read_data(imageID)
        df_pintar['type'] = np.sign(df_pintar['type'])
        OD_position = df_OD[df_OD['image'] == imgname]
        return (
            None
            if OD_position.empty
            else compute_neo_vascularization_od(df_pintar, OD_position)
        )

    except Exception as e:
        print(e)
        return {'pixels_fraction': np.nan, 'od_green_pixel_fraction': np.nan}


def main_num_green_segment_and_pixels(imgname: str) -> dict:
    """_summary_ this only will work if file are sorted correctly.

    Args:
        imgname (str): _description_

    Returns:
        float: _description_
    """
    try:
        imageID = imgname.split(".")[0]
        df_pintar = read_data(imageID)
        df_type_0 = df_pintar[df_pintar["type"] == 0]
        num_green_pixels = df_type_0.shape[0]
        num_green_segments = df_type_0['index'].nunique()
        return {
            'N_green_segments': float(num_green_segments),
            'N_green_pixels': float(num_green_pixels)
        }

    except Exception as e:
        print(e)
        return {'N_green_segments': np.nan, 'N_green_pixels': np.nan}


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
            print(e)
            print("longest 5th failed")
            quintStats_all = [np.nan for _ in range(14)]
            quintStats_artery = quintStats_all
            quintStats_vein = quintStats_all
        df_im = pd.read_csv(aria_measurements_dir + imageID + "_all_imageStats.tsv", delimiter='\t')

        return np.concatenate((all_medians, artery_medians, vein_medians, quintStats_all, \
                               quintStats_artery, quintStats_vein, df_im['nVessels'].values), axis=None).tolist()
    except Exception as e:
        print(e)
        print("ARIA didn't have stats for img", imageID)
        return [np.nan for _ in range(84)]

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

def main_vascular_density(imgname: str) -> dict:
    """

    :param imgname:
    :return:
    """

    scale_factor = 100/660 # fraction smaller compared to original

    imageID = imgname.split(".")[0]
    #print(imageID)
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

        return { 'VD_orig_all': vd_orig_all, 'VD_orig_artery': vd_orig_artery, 'VD_orig_vein': vd_orig_vein,
                 'VD_small_all': vd_small_all, 'VD_small_artery': vd_small_artery, 'VD_small_vein': vd_small_vein }

    except Exception as e:
        print(e)
        return { 'VD_orig_all': np.nan, 'VD_orig_artery': np.nan, 'VD_orig_vein': np.nan,
                 'VD_small_all': np.nan, 'VD_small_artery': np.nan, 'VD_small_vein': np.nan }


def main_fractal_dimension(imgname: str) -> dict:
    """
    :param imgname:
    :return:
    """
    imageID = imgname.split(".")[0]
    #print(imageID)
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
            'slope': float(slope),
            'slope_artery': float(slope_artery),
            'slope_vein': float(slope_vein)
        }

    except Exception as e:
        print(e)
        return {'slope': np.nan, 'slope_artery': np.nan, 'slope_vein': np.nan }



def get_data_unpivot(path):
    """
    :param path:
    :return:
    """
    # Use .read_fwf since *.tsv have diferent fixed-width formatted lines
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


def ang(lineA, lineB):
    """
    :param lineA:
    :param lineB:
    :return:
    """
    # Get nicer vector formå
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


def bifurcation_counter(df_results, imageID):
    """
    :param df_results:
    :return:
    """
    norm_acceptance=7.5
    X_1_aux = X_2_aux = 0.0
    bif_counter = 0
    cte = 3.5
    aux = []
    df_bif_positions = pd.DataFrame([])
    number_rows = df_results.shape[0]
    x = df_results['X'].values
    y = df_results['Y'].values
    dis_type = df_results['type'].values
    index_v = df_results['index'].values

    for s in range(number_rows):
        for j in range(number_rows - s):
            j = j + s
            # For X and Y: X[s] - cte <= X[j] <= X[s]
            # Both arteries or both veins and != type 0
            if (
                (x[j] >= x[s] - cte)
                and (x[j] <= x[s] + cte)
                and (y[j] >= y[s] - cte)
                and (y[j] <= y[s] + cte)
                and index_v[j] != index_v[s]
                and (dis_type[j] == dis_type[s])
                and (dis_type[j] != 0 and dis_type[s] != 0)
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

def circular_df_filter(radio, angle, od_position, df_pintar):
    """
    :param radio:
    :param angle:
    :param od_position:
    :param df_pintar:
    :return:
    """
    df_circle = compute_circular_df(radio, angle, od_position)
    new_df = pd.merge(df_circle, df_pintar, how='inner', left_on=['X', 'Y'], right_on=['X', 'Y'])
    return new_df.drop_duplicates(subset=['index'], keep='last')


def compute_circular_df(radio, angle, od_position):
    """
    :param radio:
    :param angle:
    :param od_position:
    :return:
    """
    x = radio * np.cos(angle) + od_position['x'].iloc[0]
    y = radio * np.sin(angle) + od_position['y'].iloc[0]
    df_circle = pd.DataFrame([])
    df_circle['X'] = x.round(0)
    df_circle['Y'] = y.round(0)
    return df_circle


def compute_potential_vein_arteries(df_veins_arter, od_position):
    """
    :param df_veins_arter:
    :param od_position:
    :return:
    """
    aux = []
    veins_art_x = df_veins_arter['X'].values
    veins_art_y = df_veins_arter['Y'].values
    veins_art_index = df_veins_arter['index'].values
    veins_art_diameter = df_veins_arter['Diameter'].values
    veins_art_type = df_veins_arter['type'].values
    for i, j in itertools.product(range(df_veins_arter.shape[0] - 1), range(df_veins_arter.shape[0] - 2)):
        lineA = ((od_position['x'].iloc[0], od_position['y'].iloc[0]), (veins_art_x[i], veins_art_y[i]))
        lineB = ((od_position['x'].iloc[0], od_position['y'].iloc[0]), (veins_art_x[j], veins_art_y[j]))
        if i == j:
            continue
        angulo = ang(lineA, lineB)
        angulo = round(angulo, 0)
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
            'angle': angulo
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
        df_angles_1 = df_potential_points[(df_potential_points["angle"] >= 90) & (df_potential_points["angle"] <= 230)]
        df_angles_1 = df_angles_1.sort_values(['Diameter_1', 'Diameter_2'], ascending=[False, False])
        if not df_angles_1.empty:
            main_angle = df_angles_1.iloc[0]
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


def get_radious_votes(df_pintar, OD_position, filter_type):
    """
    :param df_pintar:
    :param OD_position:
    :param filter_type:
    :return:
    """
    angle = np.arange(0, 360, 0.01)
    df_pintar['X'] = df_pintar['X'].round(0)
    df_pintar['Y'] = df_pintar['Y'].round(0)
    auxiliar_angle = []
    radius = [240, 250, 260, 270, 280, 290]
    for p in radius:
        new_df_2 = circular_df_filter(p, angle, OD_position, df_pintar)
        df_veins_arter = new_df_2[new_df_2["type"] == filter_type]
        df_veins_arter.sort_values(by=['Diameter'], ascending=False, inplace=True)
        df_potential_points = compute_potential_vein_arteries(df_veins_arter, OD_position)
        auxiliar_angle.append(get_data_angle(df_potential_points))
    return pd.DataFrame(auxiliar_angle)


def get_angle_mode(df_final_vote):
    """
    :param df_final_vote:
    :return:
    """
    for i in range(len(df_final_vote) - 1):
        for j in range(len(df_final_vote)):
            if (df_final_vote['angle'].loc[i + 1] >= df_final_vote['angle'].loc[j] - 15) and (
                    df_final_vote['angle'].loc[i + 1] <= df_final_vote['angle'].loc[j] + 2):
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


def compute_mean_angle(df_pintar, OD_position, filter_type): #filter_type=-1):
    """
    :param df_pintar:
    :param OD_position:
    :param filter_type:
    :return:
    """
    df_final_vote = get_radious_votes(df_pintar, OD_position, filter_type)
    df_final_vote = df_final_vote.reset_index().rename(columns={'index': 'vote_angle'}).copy()
    return compute_mean_angle_with_mode(df_final_vote)


def compute_vessel_radius_pixels(df_pintar, radius, od_position):
    """
    :param df_pintar:
    :param radius:
    :param od_position:
    :return:
    """
    df_pintar['DeltaX'] = df_pintar['X'] - od_position['x'].iloc[0]
    df_pintar['DeltaY'] = df_pintar['Y'] - od_position['y'].iloc[0]
    df_pintar['r2_value'] = df_pintar['DeltaX'] * df_pintar['DeltaX'] + df_pintar['DeltaY'] * df_pintar['DeltaY']
    df_pintar['r_value'] = (df_pintar['r2_value']) ** (1 / 2)

    return df_pintar[df_pintar['r_value'] <= radius].copy()


def compute_od_green_pixels_fraction(df_vessel_pixels_OD, n_rows):
    """
    :param df_vessel_pixels_OD:
    :param n_rows:
    :return:
    """
    n_rows_pixels_fraction = df_vessel_pixels_OD.shape[0]
    pixels_fraction = n_rows_pixels_fraction / n_rows
    green_pixels_OD = df_vessel_pixels_OD[df_vessel_pixels_OD['type'] == 0].shape[0]
    return {
        'pixels_fraction': float(pixels_fraction),
        'od_green_pixel_fraction': float(green_pixels_OD / n_rows_pixels_fraction)
    }


def compute_neo_vascularization_od(df_pintar, OD_position):
    """
    :param df_pintar:
    :param OD_position:
    :return:
    """
    radius = 280
    n_rows = df_pintar.shape[0]
    df_vessel_pixels_OD = compute_vessel_radius_pixels(df_pintar, radius, OD_position)
    return compute_od_green_pixels_fraction(df_vessel_pixels_OD, n_rows)

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
        first_statsfile = pd.read_csv(aria_measurements_dir + "1027180_21015_0_0_all_segmentStats.tsv", sep='\t')
        cols = first_statsfile.columns
        cols_full = [i + "_all" for i in cols] + [i + "_artery" for i in cols] + [i + "_vein" for i in cols]\
        +  [i + "_longestFifth_all" for i in cols] + [i + "_longestFifth_artery" for i in cols] + [i + "_longestFifth_vein" for i in cols]\
        +  ["nVessels"]

        df = pd.DataFrame(out, columns=cols_full)
    else:
        df = pd.DataFrame(out)

    df = df.set_index(imgfiles[:imgfiles_length])
    df.to_csv(output_path)
    df.to_pickle(output_path.replace('.csv','.pkl'))

    print(len(df), "image measurements taken")
    print("NAs per phenotype")
    print(df.isna().sum())

if __name__ == '__main__':
    # command line arguments
    qcFile = sys.argv[1] # '/Users/sortinve/PycharmProjects/pythonProject/sofia_dev/data/noQC.txt'  # qcFile used is noQCi, as we measure for all images
    phenotype_dir = sys.argv[2] # '/Users/sortinve/PycharmProjects/pythonProject/sofia_dev/data/OUTPUT/' 
    lwnet_dir = sys.argv[4] # '/Users/sortinve/PycharmProjects/pythonProject/sofia_dev/data/LWNET_DIR' 
    # function_to_execute posibilities: 'tva', 'taa', 'bifurcations', 'green_segments', 'neo_vascularization', 'aria_phenotypes', 'fractal_dimension', 'ratios'
    traits = sys.argv[6].split(',')
    print(traits)
    filter_tva_taa = 1 if function_to_execute == 'taa' else (-1 if function_to_execute == 'tva' else None)
    filter_CRAE_CRVE = 1 if function_to_execute == 'CRAE' else (-1 if function_to_execute == 'CRVE' else None)
    # all the images
    imgfiles = pd.read_csv(qcFile, header=None)
    imgfiles = imgfiles[0].values
    DATE = datetime.now().strftime("%Y-%m-%d")
    # development param
    imgfiles_length = len(imgfiles)  # len(imgfiles) is default

    # computing the phenotype as a parallel process
    os.chdir(lwnet_dir)
    
    for function_to_execute in traits:
        
        pool=Pool()

        if function_to_execute in {'taa', 'tva'}:
            imgages_and_filter = list(zip(imgfiles[:imgfiles_length], imgfiles_length * [filter_tva_taa]))
            out = pool.map(main_tva_or_taa, imgages_and_filter)
            # create_output_(out, imgfiles, function_to_execute, imgfiles_length) if out else print( "you didn't chosee any function")
        elif function_to_execute in {'CRAE', 'CRVE'}:
            imgages_and_filter = list(zip(imgfiles[:imgfiles_length], imgfiles_length * [filter_CRAE_CRVE]))
            out = pool.map(main_CRAE_CRVE, imgages_and_filter)
        elif function_to_execute == 'bifurcations':
            out = pool.map(main_bifurcations, imgfiles[:imgfiles_length])
        elif function_to_execute == 'diameter_variability':
            out = pool.map(diameter_variability, imgfiles[:imgfiles_length]) 
        elif function_to_execute == 'aria_phenotypes':
            out = pool.map(main_aria_phenotypes, imgfiles[:imgfiles_length])
        elif function_to_execute == 'fractal_dimension':
            out = pool.map(main_fractal_dimension, imgfiles[:imgfiles_length])
        elif function_to_execute == 'vascular_density':
            out = pool.map(main_vascular_density, imgfiles[:imgfiles_length])
        elif function_to_execute == 'baseline':
            out = pool.map(baseline_traits, imgfiles[:imgfiles_length])
        elif function_to_execute == 'ratios':  # For measure ratios as qqnorm(ratio)
            df_data = pd.read_csv(phenotype_dir+DATE+"_aria_phenotypes.csv", sep=',')
            df_data = df_data[['Unnamed: 0', 'medianDiameter_all', 'medianDiameter_artery', 'medianDiameter_vein', 'DF_all', 'DF_artery', 'DF_vein']]
            df_data['ratio_AV_medianDiameter'] = df_data['medianDiameter_artery'] / df_data['medianDiameter_vein']
            df_data['ratio_VA_medianDiameter'] = df_data['medianDiameter_vein'] / df_data['medianDiameter_artery']
            df_data['ratio_AV_DF'] = df_data['DF_artery'] / df_data['DF_vein']
            df_data['ratio_VA_DF'] = df_data['DF_vein'] / df_data['DF_artery']
            df_data.to_csv(phenotype_dir + DATE + "_ratios_aria_phenotypes.csv", sep=',', index=False)
        elif function_to_execute == 'ratios_CRAE_CRVE':
            df_data_CRAE = pd.read_csv(phenotype_dir+DATE+"_CRAE.csv", sep=',')
            df_data_CRVE = pd.read_csv(phenotype_dir+DATE+"_CRVE.csv", sep=',')

            df_data_CRAE.rename(columns={ df_data_CRAE.columns[0]: "image" }, inplace = True)
            df_data_CRAE.rename(columns={'median_CRE': 'median_CRAE', 'eq_CRE': 'eq_CRAE'}, inplace=True)

            df_data_CRVE.rename(columns={ df_data_CRVE.columns[0]: "image" }, inplace = True)
            df_data_CRVE.rename(columns={'median_CRE': 'median_CRVE', 'eq_CRE': 'eq_CRVE'}, inplace=True)

            df_merge=df_data_CRAE.merge(df_data_CRVE, how='inner', on='image')

            df_merge['ratio_median_CRAE_CRVE'] = df_merge['median_CRAE'] / df_merge['median_CRVE']
            df_merge['ratio_CRAE_CRVE'] = df_merge['eq_CRAE'] / df_merge['eq_CRVE']
            df_merge.to_csv(phenotype_dir + DATE + "_ratios_CRAE_CRVE.csv", sep=',', index=False)

        #elif function_to_execute == 'green_segments': #NOT ANYMORE SINCE WE USE LWNET
        #    out = pool.map(main_num_green_segment_and_pixels, imgfiles[:imgfiles_length])
        #elif function_to_execute == 'neo_vascularization': #NOT ANYMORE SINCE WE USE LWNET
        #    out = pool.map(main_neo_vascularization_od, imgfiles[:imgfiles_length])

        else:
            out = None

        pool.close()
        create_output_(out, imgfiles, function_to_execute, imgfiles_length) if out else print(
            "You didn't chose any possible function. Options: tva, taa, bifurcations, green_segments,"
            " neo_vascularization, diameter_variability, aria_phenotypes, fractal_dimension, ratios, or baseline.")
