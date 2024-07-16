#!/usr/bin/env python
# coding: utf-8

# ### Training of modified U-Net for Optic Disc on DRIONS-DB database, 256 px images (cross-validation fold #0).
# 
# You can either train your model or upload a pre-trained one from:
# *../models_weights/05.03,02:40,U-Net light, on DRIONS-DB 256 px fold 0, SGD, high augm, CLAHE, log_dice loss/last_checkpoint.hdf5*.

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:

import sys
indir = sys.argv[1]
outdir = indir
no_batches = int(sys.argv[2])
dataset = sys.argv[3]

# In[4]:

import os
import glob
from datetime import datetime
#import warnings
#warnings.simplefilter('ignore')
import scipy as sp
import scipy.ndimage
import numpy as np
import pandas as pd
import tensorflow as tf
import skimage
import skimage.exposure
import mahotas as mh
from sklearn.model_selection import KFold
from PIL import Image
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import h5py
from tqdm import tqdm_notebook
from IPython.display import display
from dual_IDG import DualImageDataGenerator


# In[5]:


import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,     Conv2D, MaxPooling2D, ZeroPadding2D, Input, Embedding,     Lambda, UpSampling2D, Cropping2D, Concatenate
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD, Adam
# from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


# In[6]:


# # running on CPU instead of GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[7]:


print('Keras version:', keras.__version__)
print('TensorFlow version:', tf.__version__)


# In[8]:


K.set_image_data_format('channels_first')


# In[9]:


def mean_IOU_gpu(X, Y):
    """Computes mean Intersection-over-Union (IOU) for two arrays of binary images.
    Assuming X and Y are of shape (n_images, w, h)."""
    
    #X_fl = K.clip(K.batch_flatten(X), K.epsilon(), 1.)
    #Y_fl = K.clip(K.batch_flatten(Y), K.epsilon(), 1.)
    X_fl = K.clip(K.batch_flatten(X), 0., 1.)
    Y_fl = K.clip(K.batch_flatten(Y), 0., 1.)
    X_fl = K.cast(K.greater(X_fl, 0.5), 'float32')
    Y_fl = K.cast(K.greater(Y_fl, 0.5), 'float32')

    intersection = K.sum(X_fl * Y_fl, axis=1)
    union = K.sum(K.maximum(X_fl, Y_fl), axis=1)
    # if union == 0, it follows that intersection == 0 => score should be 0.
    union = K.switch(K.equal(union, 0), K.ones_like(union), union)
    return K.mean(intersection / K.cast(union, 'float32'))


def mean_IOU_gpu_loss(X, Y):
    return -mean_IOU_gpu(X, Y)


# In[10]:


def dice(y_true, y_pred):
    # Workaround for shape bug. For some reason y_true shape was not being set correctly
    #y_true.set_shape(y_pred.get_shape())

    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    #y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    #y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)
    y_true_f = K.clip(K.batch_flatten(y_true), 0., 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), 0., 1.)
    #y_pred_f = K.greater(y_pred_f, 0.5)

    intersection = 2 * K.sum(y_true_f * y_pred_f, axis=1)
    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    return K.mean(intersection / union)


def dice_loss(y_true, y_pred):
    return -dice(y_true, y_pred)


def log_dice_loss(y_true, y_pred):
    return -K.log(dice(y_true, y_pred))


def dice_metric(y_true, y_pred):
    """An exact Dice score for binary tensors."""
    y_true_f = K.cast(K.greater(y_true, 0.5), 'float32')
    y_pred_f = K.cast(K.greater(y_pred, 0.5), 'float32')
    return dice(y_true_f, y_pred_f)


# In[11]:


def tf_to_th_encoding(X):
    return np.rollaxis(X, 3, 1)


def th_to_tf_encoding(X):
    return np.rollaxis(X, 1, 4)


# ### U-Net architecture
# 
# <img src="../pics/u_net_arch.png" width=40%>

# In[12]:


def get_unet_light(img_rows=256, img_cols=256):
    
    inputs = Input((3, img_rows, img_cols))
        
    conv1 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv5)

    up6 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(up6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv6)
    
    up7 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(up7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(up8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(up9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, kernel_size=3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')(conv9)
    #conv10 = Flatten()(conv10)
    
    model = Model(inputs, conv10)
    
    return model


# In[13]:



model = get_unet_light(img_rows=256, img_cols=256)
model.compile(optimizer=SGD(learning_rate=3e-4, momentum=0.95),
              loss=log_dice_loss,
              metrics=[mean_IOU_gpu, dice_metric])

model.summary()


# # Predicting UKBB in batches

# In[14]:


def loadModel():
    load_model = True   # lock
    if not load_model:
        print('load_model == False')
    else:
        # specify file:
        #model_path = '../models_weights/01.11,22:38,U-Net on DRIONS-DB 256 px, Adam, augm, log_dice loss/' \
        #    'weights.ep-20-val_mean_IOU-0.81_val_loss_0.08.hdf5'

        # or get the most recently modified file in a folder:
        model_folder = os.path.join(os.path.dirname(os.getcwd()), 'models_weights', '04.02.18_unet_on_ukbiobank_256px')

        model_path = max(glob.glob(os.path.join(model_folder, '*.hdf5')), key=os.path.getctime)
        if load_model and not os.path.exists(model_path):
            raise Exception('`model_path` does not exist')
        print('Loading weights from', model_path)

        if load_model:
            #with open(model_path + ' arch.json') as arch_file:
            #    json_string = arch_file.read()
            #new_model = model_from_json(json_string)
            model.load_weights(model_path)

        # Reading log statistics
        import pandas as pd

        log_path = os.path.join(model_folder, 'training_log.csv')
        if os.path.exists(log_path):
            log = pd.read_csv(log_path)
            if log['epoch'].dtype != 'int64':
                log = log.loc[log.epoch != 'epoch']
            print('\nmax val mean IOU: {}, at row:'.format(log['val_mean_IOU_gpu'].max()))
            print(log.loc[log['val_mean_IOU_gpu'].idxmax()])
            if 'val_dice_metric' in log.columns:
                print('\n' + 'max val dice_metric: {}, at row:'.format(log['val_dice_metric'].max()))
                print(log.loc[log['val_dice_metric'].idxmax()])
            if 'val_dice' in log.columns:
                print('\n' + 'max val dice: {}, at row:'.format(log['val_dice'].max()))
                print(log.loc[log['val_dice'].idxmax()])


# In[15]:

os.chdir("optic-nerve-cnn/scripts/")
loadModel()


# In[16]:


# fct for finding potential OD candidates using CV2

import matplotlib.patches as patches
import cv2

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    return labeled_img
    # plt.imshow(labeled_img)


# In[17]:


def batch_od_detection(batchno):
    
#     print('2')
#     # we need to build the model for every batch
    
#     K.set_session(tf.compat.v1.Session())
    
#     # as we only predict, I use CPU not GPU:
        
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     print('2.2')
#     model = get_unet_light(img_rows=256, img_cols=256)
#     print('2.3')
#     model.compile(optimizer=SGD(learning_rate=3e-4, momentum=0.95),
#                   loss=log_dice_loss,
#                   metrics=[mean_IOU_gpu, dice_metric])
#     print('2.4')
#     model.summary()
#     loadModel()
#     print('2.5')
    
    h5f_ukbb = h5py.File(indir+"resized_256px_batch"+str(batchno)+".hdf5")
    files=[os.path.split(i.decode("utf-8"))[1] for i in h5f_ukbb['filenames']]

    od_df = pd.DataFrame(index=files, columns=['width', 'height', 'area', 'center_x_y'])
        
    process = h5f_ukbb['raw_256px']
    idx = range(0,len(process))
    batch_ukb = [process[i] for i in idx]
    batch_ukb = np.array(batch_ukb).copy()
    batch_ukb = tf_to_th_encoding(batch_ukb)
    batch_ukb = th_to_tf_encoding(batch_ukb)
    batch_ukb = batch_ukb / 255.0
    batch_ukb = [skimage.exposure.equalize_adapthist(batch_ukb[i]) for i in range(len(batch_ukb))]
    batch_ukb = np.array(batch_ukb)
    batch_ukb = tf_to_th_encoding(batch_ukb)
        
    # optic disc center must be in 1/3 left 2/3 top third
    # asterisk indicates allowed boxes
    #  - - -
    # | | | |
    #  - - -
    # |*| |*|
    #  - - -
    # | | | |
    #  - - -
    
    
    top_limit = 256/3
    bottom_limit = 256/3*2
    
    # account for datasets other than uk_biobank
    # where left-right distinction is not in filename
    # now we only check that box is in top and bottom 2nd third
    # should work for many datasets as I have never seen ODs located on top or bottom
    if dataset.lower() == "uk_biobank": 
        left_limit = 256/3
        right_limit = 256/3*2
    else:
        left_limit = 999999
        right_limit = 0

    #print(top_limit, bottom_limit, left_limit, right_limit)

    for batch_i in range(0,len(files)):

        file = files[batch_i]
        if '21015' in file:
            side = 'left'
        else:
            side = 'right'

        out = (model.predict(batch_ukb[batch_i:batch_i+1])[0, 0] > 0.5).astype(np.float64)
        out_cv2 = (out * 255).astype(np.uint8)
        
        # save prediction mask
        #cv2.imwrite("/tmp/"+file,out_cv2)
        
        # f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4)
        # ax1.imshow(process[batch_i])
        # ax1.set_title('original')
        # ax2.imshow(np.rollaxis(batch_ukb[batch_i],0,3))
        # ax2.set_title('augmented')
        # ax3.imshow(out)
        # ax3.set_title('raw output')
        # ax4.set_title('OD prediction')

        # computations
        num_labels, labels_im, stats, d = cv2.connectedComponentsWithStats(out_cv2)
        #print(num_labels)
        if len(labels_im) > 0:
            labeled_img = imshow_components(labels_im)
            # ax4.imshow(labeled_img)

        else:
            tmp=1
            # ax4.imshow()


        candidate_df = pd.DataFrame(columns=['width','height','area','center_x_y','within_box'])
        for idx,i in enumerate(stats[1:]):
            width = i[2]
            height = i[3]
            area = i[4]
            center_x = i[0] + width/2
            center_y = i[1] + height/2
            # rect = patches.Rectangle((i[0], i[1]), i[2], i[3], linewidth=1, edgecolor='r', facecolor='none')
            # ax4.add_patch(rect)

            # limits
            if side == 'left':
                if (center_x < left_limit) & (center_y < bottom_limit) & (center_y > top_limit):
                    within_box = True
                else:
                    within_box = False
            else:
                if (center_x > right_limit) & (center_y < bottom_limit) & (center_y > top_limit):
                    within_box = True
                else:
                    within_box = False

            candidate_df.loc[len(candidate_df)] = [width,height,area,(center_x,center_y),within_box]


        # keeping only boxes that are within plausibility area
        # defining largest candidate as the correct one
        try:
            candidate_df = candidate_df[candidate_df['within_box']==True].sort_values('area',ascending=False)
            # ax4.scatter(candidate_df['center_x_y'].iloc[0][0], candidate_df['center_x_y'].iloc[0][1], marker='*', color='white')
            od_df.loc[file] = candidate_df.iloc[0][0:4]
        except Exception as e:
            print(e)
            od_df.loc[file] = [np.nan for i in range(0,4)]       


        # plt.tight_layout()
        # plt.show()    
    
    od_df.to_pickle(outdir + "/OD_batch" + str(batchno) + ".pkl")


# In[ ]:


# from multiprocessing import Pool
# # import threading
# can't do multiprocessing as cannot laod multiple CNN models simultaneously in python notebook

# pool = Pool(10)

batches = [i for i in range(1,no_batches+1)]

for i in batches:
    batch_od_detection(i)

# pool.map(batch_od_detection, batches[0:20])

# thread=threading.Thread(target=batch_od_detection, args=batches[0:20])
# thread.start()


# # Correcting scaling effects

# In[19]:


import pickle
with open(indir + '/resize_params.pkl', 'rb') as f:
    resize_dict = pickle.load(f)
    
scaling_param = resize_dict['imsq_orig']/resize_dict['imsq_resized']

# In[29]:


def center_adjust_position(center_pos, resize_dict, scaling_param):    
    try:
#         my_str = my_str.replace('(','')
#         my_str = my_str.replace(')','')
        
#         my_tuple = tuple(map(int,map(float, my_str.split(', '))))
        
        return (center_pos[0]*scaling_param+resize_dict['rm_left'], center_pos[1]*scaling_param+resize_dict['rm_top'])
    
    # some rows are nan
    except Exception as e:
        print(e)
        return np.nan

for i in range(1,no_batches+1):
    # print(i)
    df = pd.read_pickle(outdir + "/OD_batch"+str(i)+".pkl")
    df.index.name='image'

    df["width"] = df["width"] * scaling_param
    df['height'] = df['height'] * scaling_param

    df['area'] = df['area'] * scaling_param**2
    df['center_x_y'] = [center_adjust_position(i, resize_dict, scaling_param) for i in df['center_x_y']]
    
    center_x = []
    center_y = []
    for j in df['center_x_y']:
        if type(j) == tuple:
            center_x.append(j[0])
            center_y.append(j[1])
        else:
            center_x.append(np.nan)
            center_y.append(np.nan)

    df['x'] = center_x
    df['y'] = center_y

# print(df.shape)
    
    if i==1:
        df_out = df
        #print(df_out.shape)
    else:
        df_out = pd.concat([df_out,df])
        #print(df_out.shape)


# In[17]:


df_out.to_pickle(outdir + '/od_all.pkl')
df_out.to_csv(outdir + '/od_all.csv')

