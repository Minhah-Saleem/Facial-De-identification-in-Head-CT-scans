# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 03:48:16 2021

@author: Rydstorm
"""

import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_file
import cv2
from os import makedirs,listdir
from os.path import join, exists, split
from skimage import morphology
import numpy as np
from tqdm import tqdm
import glob as glob
import SimpleITK as sitk
import sys


def new_blurred(image, prev = 0,s=0):
    gray = np.zeros((image.shape[0], image.shape[1]) , dtype=int)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if image[i,j] > 100 and image[i,j] < 500:
                gray[i,j] = 1
    #plt.imshow(gray , cmap = 'gray')
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1)) 
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 
    gray = cv2.morphologyEx(np.float32(gray), cv2.MORPH_CLOSE, ker1)
    gray = cv2.morphologyEx(np.float32(gray), cv2.MORPH_CLOSE, ker2)
    #plt.imshow(gray , cmap = 'gray')
    gray = gray>0
    gray = morphology.remove_small_objects(gray, min_size=200)
    #plt.imshow(gray , cmap = 'gray')
    gray = gray.astype(np.uint8)
    kernel = np.ones((30, 30), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = gray.astype(np.uint8)
    #plt.imshow(gray , cmap = 'gray')
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gray = cv2.drawContours(gray, contours,-1, 255, 10)
    #plt.imshow(gray , cmap = 'gray')
    gray[300:,:] = 0
    #plt.imshow(diff, cmap ='gray')
    seg = gray
    seg = np.where((seg==255),seg,0)
    z=0
    if not s==0:
        diff = np.subtract(seg,prev)
        diff = diff>1
        xx = int((diff.shape[1])/2)
        diff[:,0:(xx-100)] =0
        diff[:,(xx+100):] = 0
        diff = morphology.remove_small_objects(diff, min_size=500)
        if np.any(diff):
            for l in range(diff.shape[1]):
                for j in range(diff.shape[0]):
                    if diff[j,l]==True:
                        if np.any(prev[(j-10):(j+10),l]):
                            diff[j,l] = False
            seg = np.where((diff==True),0,seg)
            z=1
    blurred_img = cv2.GaussianBlur(image, (101, 101), 400)
    out = np.where((seg==255), blurred_img, image)
    out = out.astype(np.int16)
    if z==0:
        return out , seg
    else:
        return out, prev





def blurred(image):
    gray = np.zeros((image.shape[0], image.shape[1]) , dtype=int)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if image[i,j] > 100 and image[i,j] < 500:
                gray[i,j] = 1
    #plt.imshow(gray , cmap = 'gray')
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1)) 
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) 
    gray = cv2.morphologyEx(np.float32(gray), cv2.MORPH_CLOSE, ker1)
    gray = cv2.morphologyEx(np.float32(gray), cv2.MORPH_CLOSE, ker2)
    #plt.imshow(gray , cmap = 'gray')
    gray = gray>0
    gray = morphology.remove_small_objects(gray, min_size=200)
    #plt.imshow(gray , cmap = 'gray')
    gray = gray.astype(np.uint8)
    kernel = np.ones((30, 30), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = gray.astype(np.uint8)
    #plt.imshow(gray , cmap = 'gray')
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gray = cv2.drawContours(gray, contours,-1, 255, 3)
    #plt.imshow(gray , cmap = 'gray')
    gray[300:,:] = 0
    #plt.imshow(diff, cmap ='gray')
    seg = gray
    seg = np.where((seg==255),seg,0)
    blurred_img = cv2.GaussianBlur(image, (201, 201), 1000)
    out = np.where((seg==255), blurred_img, image)
    out = out.astype(np.int16)
    
    return out


i_folder = r'C:\Rabeea\Work\Sine wave\face\MRI\9264'
o_folder = r'C:\Rabeea\Work\Sine wave\face\MRI\9264_C'
i_folder=i_folder + "/*.dcm"
data = np.empty((0,1))
axial = np.array([1., 0., 0., 0., 1., 0.])
files_list = sorted(glob.glob(i_folder))
for filename in tqdm(files_list):
    dcmData = pydicom.dcmread(filename)
    a = (np.around(dcmData.ImageOrientationPatient,0))
    a = np.multiply(a,a)
    if (a==axial).all():
        des = join(o_folder,dcmData.SeriesDescription)
        if not exists(des):
            makedirs(des)
        h,name = split(filename)
        des = join(des,name)
        dcmData.save_as(des)
##i=0
#for series in tqdm(listdir(o_folder)):
#    series_path = join(o_folder,series) + "/*.dcm"
#    for filename in tqdm(sorted(glob.glob(series_path))):
#        dcmData = pydicom.dcmread(filename)
#        img = dcmData.pixel_array
#        #plt.imshow(img, cmap = 'gray')
#        out = blurred(img)
#        #plt.imshow(out,cmap = 'gray')
#        #if i == 0:
#        #    out, prev_img =new_blurred(img)
#        #else:
#        #    out, prev_img =new_blurred(img, prev_img,s=1)
#        
#        out = out.astype(np.int16)
#        dcmData.PixelData = out.tobytes()
#        #head , tail = split(filename)
#        #des_path = join(output,tail)
#        dcmData.save_as(filename)
#    
#    
    
#filename = r'C:\Rabeea\Work\Sine wave\face\MRI\9264_A\AXL FSE T2/5d6f53dee4b0aecb29a30942.dcm'
#dcmData = pydicom.dcmread(filename)
#img = dcmData.pixel_array
#plt.imshow(img, cmap = 'gray')
#out = blurred(img)
#plt.imshow(out , cmap = 'gray')
    
    