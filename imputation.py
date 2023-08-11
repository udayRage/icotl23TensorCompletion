import os
import cv2
import copy
import scipy.misc
from time import time
from geotiff import GeoTiff as gt
import tensorly as tl
import numpy as np
import pandas as pd
from numba import jit
from tensorly import random

from matplotlib import pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# Importing our image completion algorithms and helper functions
from algos import *
from helpers import *
from UIhandler import *


class Imputation:
    def __init__(self, dataframe = None, tensor_rank_estimate = 12):
        '''
            def: Imputes missing data in a given dataframe.
            :param dataframe: The dataframe/np array/filepath with missing pixels
            :param  speed: "slow"/"fast"(default), speed of the algorithm
            :param  visualize: bool (default False) to determine whether to
                             visualize the output.
            :param  trank: Estimate for tensor rank.
        '''

        if type(dataframe) == np.ndarray:
            self.img = dataframe
            self.pts_arr = None
        elif type(dataframe) == pd.DataFrame:
            self.dataframe = dataframe
            self.img, self.pts_arr = df2np(dataframe)
        elif type(dataframe) == str:
            self.dataframe = pd.read_csv(dataframe)
            self.img, self.pts_arr = df2np(self.dataframe)
        else:
            raise Exception("Invalid dataframe type")
        
        self.imSize = self.img.shape
        self.tensor_rank_estimate = tensor_rank_estimate
        self.auxilary_arr_for_cmtf = None
        self.hil_mask = None # Human-in-Loop mask
        self.current_mask = None # m1
        self._corrupt_img = None # must be private

        self.is_imputation_done = False
        self.brightness_param = 1.0
        self.fixed_img_np = None # fixed image after imputation
        self.fixed_img_df = None # fixed image df after imputation
        self.hal = None
        self.sil = None
        self.cmtf = None
        self.cpals = None
        self.cmsi = None
        
        self.algo_runtimes = [] # list to keep track of runtime for each algo
        self.visualization_arr = [] # list to keep track of visualization for each algo


    def get_corrupt_img(self):

        """
        gets the corrupt image if it exists otherwise returns None
        :return: corrupt image
        """
        if self._corrupt_img is not None:
            return copy.deepcopy(self._corrupt_img)
        else:
            return None

    def set_corrupt_img(self, value):
        self._corrupt_img = copy.deepcopy(value)

    def set_brightness(self, value):
        self.brightness_param = value

    def draw_mask(self):
        '''
            Generates a rectangular/hand drawn mask for the image
            using manual user input.
        '''
        if self.get_corrupt_img() is None:
            self.set_corrupt_img(self.img)
        
        # consider only the first 3 layers,
        # otherwise OpenCV throws errors

        # preprocessed_img = self.get_corrupt_img().astype(np.float32)
        # print("preprocessed_img.shape",preprocessed_img.shape)
        # preprocess_img = self.get_corrupt_img()
        # preprocess_img = preprocess_img[:,:,0:3]

        # preprocessed_img /= np.max(preprocessed_img[:,:,0:3])
        # # preprocessed_img *= 255

        imgdm = copy.deepcopy(self.get_corrupt_img())
        imgdm = np.array(imgdm[:,:,:3]*self.brightness_param)

        _, inv_mask = drawMask(imgdm)

        inv_mask = np.array(inv_mask)
        ivx = (inv_mask == 255)
        ivx = np.array(ivx,dtype=bool)
        ivx=~ivx

        self.current_mask = ivx
        self.set_corrupt_img(mask_on_img(copy.deepcopy(self.img),self.current_mask))

        return self.current_mask


    def generate_mask(self, approximate_mask = False, threshold_value = 0.1):
        '''
            def: Generates a mask for the image.
            :param  threshold_value: The threshold value for the mask,
            depends completely on the dataframe
            We detect a mask by identifying outlier pixels and dead pixels
            The exact_masker works only where the missing pixels are all 0s.
        '''
        if self.get_corrupt_img() is None:
            self.set_corrupt_img(self.img)
        if approximate_mask:
            # approximate, neighbour-based masker
            self.current_mask = masker(self.get_corrupt_img(), threshold_value)
        else:
            # masks only if values in all bands are 0.
            self.current_mask = exact_masker(self.get_corrupt_img())

        return self.current_mask

    def synthetic_mask(self, fraction_to_keep = 0.9):
        '''
            def: Generates a synthetic mask for the image.
        '''
        corrupt_img_output, perfect_mask = randomDelete(self.img, fraction_to_keep)
        self.set_corrupt_img(corrupt_img_output)
        self.current_mask = perfect_mask
        print("fraction to keep", fraction_to_keep)
        return self.current_mask, corrupt_img_output

    def impute(self, speed = "fast", hil = 0, skip_vis=0, hil_param = 25):
        '''
            def: Imputes missing data in a given dataframe.
            :param  speed: "slow"/"fast"(default), speed of the algorithm
            :param  hil: 0-> No CMTF4SI, 1-> Include CMTF4SI, 2-> Only CMTF4SI
            
        '''
    
        # Defining constants for (Ha/Si)LRTC algos
        a = abs(np.random.rand(self.imSize[-1], 1))
        a = a / np.sum(a)
        b = abs(np.random.rand(self.imSize[-1], 1))/200

        self.current_mask = np.array(self.current_mask, dtype=bool)

        self.auxilary_arr_for_cmtf = self.get_corrupt_img()[:,:,0]

        if hil != 0:
            ### HiL MASK MAKER CODE --------------------

            imgdummy = copy.deepcopy(self.get_corrupt_img())
            imgdummy = np.array(imgdummy[:,:,:3]*self.brightness_param)
            # Draw a HiL mask. Use 'o'/'p' to zoom in/out, LMB to draw.
            Zt,Z = drawMask(imgdummy)
            Z = np.array(Z, dtype=float)
            temp_mask = (Z == 255) # [FIX] this value is hardcoded in UIhandler.py
            imputation_val = np.average(self.get_corrupt_img()).astype('float')
            print("Imputation val = ",imputation_val)
            Z[temp_mask] = imputation_val

            self.hil_mask = self.get_corrupt_img()[:,:,0] * self.current_mask + Z * ~self.current_mask

            ### end HiL MASK MAKER CODE ----------------


        itx = { "hal": 10, "sil": 0, "cmtf": 200, "cp": 200, "si": 1}

        # typically needed only when the percentage of missing pixels is
        # more than 40-50%
        if speed == "slow":
            itx = { "hal": 30, "sil": 30, "cmtf": 200, "cp": 200, "si": 1}
        
        if hil:
            if hil == 2:
                itx = { "hal": 0, "sil": 0, "cmtf": 1, "cp": 1, "si": 1}
            itx["si"] = 400 if speed == "slow" else 200
        else:
            self.hil_mask = self.auxilary_arr_for_cmtf

        self.algo_runtimes = [] # list to keep track of runtime
        img_orig = copy.deepcopy(self.get_corrupt_img())

        # try:
        self.algo_runtimes.append(time())
        self.hal = haLRTC(self.get_corrupt_img(), self.current_mask, a, b, itx["hal"])
        print('')

        self.algo_runtimes.append(time())
        self.sil = siLRTC(self.get_corrupt_img(), self.current_mask, a, b, itx["sil"])
        print('')

        self.algo_runtimes.append(time())
        _, self.cmtf = cmtf(self.get_corrupt_img(), [self.auxilary_arr_for_cmtf,self.auxilary_arr_for_cmtf], [0,1], self.tensor_rank_estimate, self.current_mask, tol=1e-4, maxiter=itx["cmtf"])
        print('')

        self.algo_runtimes.append(time())
        _,self.cmsi ,_ = cmtf4si(self.get_corrupt_img(), [self.hil_mask,self.hil_mask], [0,1], self.tensor_rank_estimate, self.current_mask, alpha = hil_param, tol=1e-4, maxiter=itx["si"])
        print('')

        self.algo_runtimes.append(time())
        _, self.cpals = cp_als(self.get_corrupt_img(), self.tensor_rank_estimate, self.current_mask, tol=1e-4, maxiter=itx["cp"], original_img = img_orig)
        print('')

        self.algo_runtimes.append(time())
        self.is_imputation_done = True
        
        # except Exception as e:
        #     print("Error in imputation", e)


        print('Base RSE:', RSE(self.img, self.get_corrupt_img()))

        self.hal[self.current_mask] = self.img[self.current_mask]
        self.sil[self.current_mask] = self.img[self.current_mask]
        self.cmtf[self.current_mask] = self.img[self.current_mask]
        self.cmsi[self.current_mask] = self.img[self.current_mask]
        self.cpals[self.current_mask] = self.img[self.current_mask]

        self.hal = normalize(self.hal, self.current_mask)
        self.sil = normalize(self.sil, self.current_mask)
        self.cmtf = normalize(self.cmtf, self.current_mask)
        self.cmsi = normalize(self.cmsi, self.current_mask)
        self.cpals = normalize(self.cpals, self.current_mask)

        print('HaLRTC RSE:', RSE(self.img, self.hal))
        print('SiLRTC RSE:', RSE(self.img, self.sil))
        print('CMTF RSE:', RSE(self.img, self.cmtf))
        print('CMSI RSE:', RSE(self.img, self.cmsi))
        print('CP-ALS RSE:', RSE(self.img, self.cpals))

        print('HaLRTC runtime:', self.algo_runtimes[1]-self.algo_runtimes[0])
        print('SiLRTC runtime:', self.algo_runtimes[2]-self.algo_runtimes[1])
        print('CMTF runtime:', self.algo_runtimes[3]-self.algo_runtimes[2])
        print('CMTF4SI runtime:', self.algo_runtimes[4]-self.algo_runtimes[3])
        print('CP-ALS runtime:', self.algo_runtimes[5]-self.algo_runtimes[4])

        print("Imputation completed successfully !")

        self.imputedArrays = [self.sil, self.hal, self.cmtf, self.cmsi, self.cpals]

        if skip_vis == 0:
            self.visualization_arr = [
                    [self.img[:,:,3],'Original', RSE(self.img, self.img)],
                    [self.get_corrupt_img()[:,:,3],'Corrupt', RSE(self.img, self.get_corrupt_img())],
                    [self.current_mask,'Mask', 100.0],
                    [self.sil[:,:,3],'SiLRTC', RSE(self.img, self.sil)],
                    [self.hal[:,:,3],'HaLRTC', RSE(self.img, self.hal)],
                    [self.cpals[:,:,3],'CP-ALS', RSE(self.img, self.cpals)],
                    [self.cmsi[:,:,3],'CMTF4SI', RSE(self.img, self.cmsi)],
                    [self.cmtf[:,:,3],'CMTF-Base', RSE(self.img, self.cmtf)]]

            candidates = [self.sil, self.hal, self.cpals, self.cmtf, self.cmsi]

            # save image with lowest RSE
            best_idx = np.argmin([RSE(self.img,candidates[i]) for i in range(1,len(candidates))])
            best = np.array(candidates[best_idx], dtype = object)
            self.best_img_np = best

            best_to_df = best.reshape(best.shape[0]*best.shape[1],best.shape[2])
            if (self.pts_arr is not None):
                best_to_df = np.insert(best_to_df, 0, np.array(self.pts_arr), axis=1)
                best_to_df = pd.DataFrame(best_to_df)
                self.best_img_df = best_to_df

        return self.imputedArrays, self.algo_runtimes
    
    def visualize(self):
        '''
            def: Visualizes the results
        '''
        if not self.is_imputation_done:
            raise Exception("Imputation not done yet!")
        
        rows = (len(self.visualization_arr)+1) // 2
        cols = 2
        f, axarr = plt.subplots(rows,cols,figsize=(15,15))
        # for i in range(rows):
        #     for j in range(cols):
        #         axarr[i,j].imshow(self.visualization_arr[i*cols+j][0])
        #         axarr[i,j].set_title(self.visualization_arr[i*cols+j][1] 
        #         + ', RSE:'+ str(self.visualization_arr[i*cols+j][2]))
        #         axarr[i,j].axis('off')

        for i in range(len(self.visualization_arr)):
            axarr[i//cols,i%cols].imshow(self.visualization_arr[i][0])
            axarr[i//cols,i%cols].set_title(self.visualization_arr[i][1] 
            + ', RSE:'+ str(self.visualization_arr[i][2]))
            axarr[i//cols,i%cols].axis('off')

        plt.show()

        return
    
    def get_best_np(self):
        '''
            def: Returns the best image as a numpy array
        '''
        if not self.is_imputation_done:
            raise Exception("Imputation not done yet!")
        return self.best_img_np
    
    def get_best_df(self):
        '''
            def: Returns the best image as a pandas dataframe
        '''
        if not self.is_imputation_done:
            raise Exception("Imputation not done yet!")
        return self.best_img_df
    
    def get_mask(self):
        '''
            def: Returns the mask
        '''
        return self.current_mask
    
    def show_img(self, img):
        '''
            def: Shows the image
        '''
        plt.imshow(img * self.brightness_param)
        plt.show()
        return