#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
#import warnings
#warnings.simplefilter('ignore')
from operator import attrgetter
from platform import python_version_tuple

if python_version_tuple()[0] == 3:
    xrange = range

import scipy as sp
import scipy.ndimage
import numpy as np
import pandas as pd
import skimage
import skimage.measure
from PIL import Image
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import h5py
from tqdm import tqdm_notebook
from IPython.display import display
from extract_data import *


# In[3]:


# storing test set

import sys
import glob
import h5py
import cv2

datapath = sys.argv[2] # where to store output. h5f requires absolute path
rawdir = sys.argv[1]
nfiles = len(glob.glob(rawdir+'*.png'))
print(f'count of image files nfiles={nfiles}')


# In[4]:


# batch prep

n_cpus = int(sys.argv[3])
batch_len = int(np.ceil(nfiles/n_cpus))

batches = [(i+1,0+i*batch_len,min((i+1)*batch_len,nfiles-1)) for i in range(0,n_cpus)]


# In[5]:


# generate UKBB full dataset


# resize all images and load into a single dataset

import time

def generate_batch(variables):
    
    start = time.time()
    
    batch_no,start_b,stop = variables
    
    h5file = datapath + '/resized_256px_batch'+str(batch_no)+'.hdf5'

    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    with h5py.File(h5file,'w') as  h5f:
        
        batch_files = sorted(glob.iglob(rawdir + "*.png"))[start_b:stop]
        h5f.create_dataset("filenames", data=np.array(batch_files, dtype='S'))
        img_ds = h5f.create_dataset('raw_256px',shape=(stop-start_b, 256, 256,3), dtype=int)
        for cnt, ifile in enumerate(batch_files) :
            # if cnt % 1000 == 0:
            #     print(cnt)

            img = cv2.imread(ifile, cv2.IMREAD_COLOR)
#             print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # cv2 uses BGR, we would like RGB, I technically used the inverse function, but I believe it ends up the same as BGR and RGB are point symmetrical

            hh,ww=img.shape[0:2]
            diff = ww-hh
            if diff > 0:
                rm_left = np.floor(diff/2).astype(int)
                rm_right = np.ceil(diff/2).astype(int)
                rm_top = 0
                rm_bottom = 0
                
                im_sq=img[:,rm_left:-rm_right]
            else:
                rm_top = np.floor(np.abs(diff)/2).astype(int)
                rm_bottom = np.ceil(np.abs(diff)/2).astype(int)
                rm_left = 0
                rm_right = 0
                im_sq=img[rm_top:-rm_bottom]

            
            # making square
#             print(im_sq.shape)

            # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
            img_resize = cv2.resize( im_sq, (IMG_WIDTH, IMG_HEIGHT) )
                        
            if cnt==0:
                import pickle
                with open(datapath + 'resize_params.pkl', 'wb') as f:
                    pickle.dump({'rm_left':rm_left, 'rm_right':rm_right, 'rm_top':rm_top, 'rm_bottom':rm_bottom, 'imsq_orig':im_sq.shape[0], 'imsq_resized':img_resize.shape[0]}, f)
            
            img_ds[cnt:cnt+1:,:,:] = img_resize
            
#             plt.figure()
#             plt.imshow(img)
#             plt.figure()
#             plt.imshow(im_sq)
            
    return (time.time() - start) / 60.0


# In[6]:


# ukb generation in parallel

from multiprocessing import Pool
if __name__ == "__main__":   

    pool = Pool(n_cpus)

    #print(batches)

    times = pool.map(generate_batch, batches)
    pool.close()
