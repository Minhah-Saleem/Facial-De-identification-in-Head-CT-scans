#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tempfile
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_file
import cv2
import os
from skimage import morphology
import math
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import glob as glob
import SimpleITK as sitk
import sys
from pydicom.encaps import encapsulate
from pydicom.uid import JPEG2000
from pydicom.pixel_data_handlers.util import apply_modality_lut
from imagecodecs import jpeg_encode
import imagecodecs
import random
from pydicom.uid import ExplicitVRLittleEndian
from imagecodecs import jpeg_encode
import imagecodecs
import pylibjpeg


# # CT

# In[ ]:



def sorted_files(folder):
    ins=[]
    ser=[]
    f= os.listdir(folder)
    for name in tqdm( f):
        itkimage = sitk.ReadImage(os.path.join(folder,name))
        temp= float(itkimage.GetMetaData('0020|0013')) # instance number 
        temp1= itkimage.GetMetaData('0020|000e') #series instance UID 
        ins.append(int(temp))
        ser.append(temp1)
    series=np.unique(ser)
    files=[[x for sr,_,x in sorted(zip(ser,ins,f)) if sr==s] for s in series]
    return files

def scan(path):
    f= sorted_files(path)
    series=[]
    files=[]
    for i in tqdm (range (len(f))):
        img=[]
        file=[]
        for filename in f[i]:
            itkimage = sitk.ReadImage(os.path.join(path,filename))
            numpyImage = sitk.GetArrayFromImage(itkimage)
            img.append(numpyImage[0,:,:])
            file.append(filename)
        series.append(img)
        files.append(file)
    return series,files


def CT_blurred_sag(image,rows,columns):
    imagee = image>-50
    skull = image>500
    kernel = np.ones((30, 30), np.uint8)
    imagee = cv2.morphologyEx(imagee.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    imagee = ndimage.morphology.binary_fill_holes(imagee)
    imagee = imagee.astype(np.uint8)
    contours, hierarchy = cv2.findContours(imagee, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gray = np.zeros_like(imagee)
    gray = cv2.drawContours(gray, contours,-1, 255,-1 )
    temp=gray.copy()
    temp=np.where((temp==255),1,0)
    gray = np.where((gray==255), 0, 1)
    gray[:,200:] = 0
    r=[[random.randint(-250,500) for i in range(rows)] for j in range(columns)]
    r=np.asarray(r)
    blurred_img= np.multiply(gray,r)
    out = np.where((gray==1), blurred_img, image)
    out = np.where(skull>0,image,out)
    for i in range(np.shape(out)[0]):
        if not np.any(temp[i]):
            out[i]=np.multiply(out[i],0)
    out = out.astype(np.int16)
    return out



def CT_blurred_axial(image,rows,columns):
    imagee = image>-50
    skull = image>500
    kernel = np.ones((30, 30), np.uint8)
    imagee = cv2.morphologyEx(imagee.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    imagee = ndimage.morphology.binary_fill_holes(imagee)
    imagee = imagee.astype(np.uint8)
    contours, hierarchy = cv2.findContours(imagee, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gray = np.zeros_like(imagee)
    gray = cv2.drawContours(gray, contours,-1, 255,-1 )
    gray = np.where((gray==255), 0, 255)
    gray[300:,:] = 0
    gray1 = np.zeros_like(imagee)
    gray1 = cv2.drawContours(gray1, contours,-1, 255,4 )
    gray1[300:,:] = 0
    gray=np.where((gray1==255), gray1, gray)
    gray=np.where((gray==255),1,0)
    r=[[random.randint(-250,500) for i in range(rows)] for j in range(columns)]
    r=np.asarray(r)
    blurred_img= np.multiply(gray,r)
    out = np.where((gray==1), blurred_img, image)
    out = np.where(skull>0,image,out)
    out = out.astype(np.int16)
    return out


# In[ ]:


def predictions_ct(path, output):
    if not os.path.exists(output):
        os.makedirs(output)
    output_list= []
    classUID = []
    scan_series,filenames=scan(path)
    for j in tqdm(range(len(scan_series))):
        scan, names = scan_series[j],filenames[j]
        scan=np.asarray(scan)
        dcmData = pydicom.dcmread(os.path.join(path,names[0]))
        axial  =  np.array([1., 0., 0., 0., 1., 0.])
        sagittal= np.array([-1., 0., 0., 0., 0., -1.])
        if (0x0020,0x0037) in dcmData:
            a = (np.around(dcmData.ImageOrientationPatient, 0))
            if (a == axial).all():
                for i in range(scan.shape[0]):
                    name = names[i]
                    img = scan[i, :, :]
                    dcmData = pydicom.dcmread(os.path.join(path,name)) ##read file to change it's pixel data to anonymized
                    r= dcmData.Rows
                    c=dcmData.Columns
                    out= CT_blurred_axial(img,r,c)
                    out = out.astype(np.int16)
                    dcmData.file_meta.TransferSyntaxUID= '1.2.840.10008.1.2.1'
                    dcmData.PixelData = out.tobytes()
                    des_path = os.path.join(output, name)
                    dcmData.save_as(des_path) ##save file in anonym folder
                    classUID.append(str(dcmData.SOPClassUID))
                    output_list.append(des_path)
            elif (a==sagittal).all():
                for i in range(scan.shape[0]):
                    name = names[i]
                    img = scan[i, :, :]
                    dcmData = pydicom.dcmread(os.path.join(path,name)) ##read file to change it's pixel data to anonymized
                    r= dcmData.Rows
                    c=dcmData.Columns
                    if i == 0:
                        out = img
                    else:
                        out = CT_blurred_sag(img,r,c)
                    out = out.astype(np.int16)
                    dcmData.file_meta.TransferSyntaxUID= '1.2.840.10008.1.2.1'
                    dcmData.PixelData = out.tobytes()
                    des_path = os.path.join(output, name)
                    dcmData.save_as(des_path) ##save file in anonym folder
                    classUID.append(str(dcmData.SOPClassUID))
                    output_list.append(des_path)
            else:
                for i in range(scan.shape[0]):
                    name = names[i]
                    img = scan[i, :, :]
                    dcmData = pydicom.dcmread(os.path.join(path,name))
                    des_path = os.path.join(output, name)
                    dcmData.save_as(des_path) ##save file in anonym folder
                    classUID.append(str(dcmData.SOPClassUID))
                    output_list.append(des_path)
        else:
            for i in range(scan.shape[0]):
                name = names[i]
                img = scan[i, :, :]
                dcmData = pydicom.dcmread(os.path.join(path,name)) 
                des_path = os.path.join(output, name)
                dcmData.save_as(des_path) 
                classUID.append(str(dcmData.SOPClassUID))
                output_list.append(des_path)
    mimeType = "application/dicom"
    recommendation_string = {"finding": "finding","conclusion":"conclusion","recommendation":"recommendation"} 
    return output_list, classUID, mimeType, recommendation_string


# # CBCT 

# In[ ]:


def get_full_scan(folder_path):
    files_List = glob.glob(folder_path + '/**/*.dcm', recursive=True)
    itkimage = sitk.ReadImage(files_List[0])
    rows = int(itkimage.GetMetaData('0028|0010'))
    cols = int(itkimage.GetMetaData('0028|0011'))
    mn = 1000
    mx = 0
    for file in tqdm(files_List):
        itkimage = sitk.ReadImage(file)
        mn = np.min([mn, int(itkimage.GetMetaData('0020|0013'))])
        mx = np.max([mx, int(itkimage.GetMetaData('0020|0013'))])
    full_scan = np.ndarray(shape=(mx - mn + 1, rows, cols), dtype=float, order='F')
    new_list = np.ndarray(shape=(mx - mn + 1), dtype=object)
    for file in tqdm(files_List):
        img, n = dcm_image(file)
        n = int(n)
        full_scan[n - mn, :, :] = img[0, :, :]
        new_list[n-mn] = file
    return full_scan,new_list

def dcm_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    ins = float(itkimage.GetMetaData('0020|0013'))
    return numpyImage, ins
    

def CBCT_blurred(image,ex, prev = 0,s=0):
    gray = np.zeros((image.shape[0], image.shape[1]) , dtype=int)
    skull = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if image[i,j] > -400:# and image[i,j] < 500:
                gray[i,j] = 1
                if image[i,j]>500:
                    skull[i,j] = 1

    gray = gray>0
    gray = morphology.remove_small_objects(gray, min_size=200)
    gray = gray.astype(np.uint8)
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    gray = cv2.morphologyEx(np.float32(gray), cv2.MORPH_CLOSE, ker1)
    gray = cv2.morphologyEx(np.float32(gray), cv2.MORPH_CLOSE, ker2)
    kernel = np.ones((30, 30), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    skull = cv2.morphologyEx(skull.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    gray = gray.astype(np.uint8)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area = 0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a>area:
            contour = cnt
            area = a
    gray = np.zeros_like(gray)
    gray = cv2.drawContours(gray, contour,-1, 255, 10)
    points = np.argwhere(gray>0)
    mini = np.amin(points[:,0])
    maxi = np.amax(points[:,0])
    mini = mini + int((maxi-mini)/2)
    gray[mini:,:] = 0
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
    blurred_img = cv2.GaussianBlur(ex, (101, 101), 400)
    out = np.where((seg==255), blurred_img, ex)
    out = np.where(skull>0,ex,out)
    if z==0:
        return out , seg
    else:
        return out, prev



# In[ ]:


def predictions_cbct(input_folder, output_folder):
    if not os.path.exists(output_folder):
        print('making')
        os.makedirs(output_folder)
    files_list = sorted(glob.glob((input_folder+ "/**/*.dcm"),recursive = True))
    output_list= []
    classUID = []
    initialdata = pydicom.dcmread(files_list[0])
    mode = str(initialdata.Modality)
    tmode = 'a'
    if not ((str(initialdata.file_meta.TransferSyntaxUID) == '1.2.840.10008.1.2.1') and (str(initialdata.file_meta.TransferSyntaxUID) == '1.2.840.10008.1.2') ):
        tmode = 'jpeg'

    i = 0
    scan, names = get_full_scan(input_folder)
    for i in tqdm(range(len(names))):
        name = names[i]
        dcmData = pydicom.dcmread(name)
        ex = dcmData.pixel_array
        img = scan[i, :, :]
        if i == 0:
            out, prev_img = CBCT_blurred(img,ex)
        else:
            out, prev_img = CBCT_blurred(img,ex, prev_img, s=1)
        out = out.astype(ex.dtype)
        dcmData1 = dcmData.copy()

        if not tmode=='a':
            dcmData1.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'

        dcmData1.PixelData = out.tobytes()
        _, tail = os.path.split(name)
        des_path = os.path.join(output_folder, tail)
        classUID.append(str(dcmData1.SOPClassUID))
        output_list.append(des_path)
        dcmData1.save_as(des_path)
        i = i + 1
    mimeType = "application/dicom"
    recommendation_string = {"finding": "finding","conclusion":"conclusion","recommendation":"recommendation"} 
    return output_list, classUID, mimeType, recommendation_string


# # Main

# In[ ]:


def predictions(input_folder, output_folder):
    files_list = sorted(glob.glob((input_folder+ "/**/*.dcm"),recursive = True))
    initialdata = pydicom.dcmread(files_list[0])
    if 'ct' or 'CT' in str(initialdata.Modality):
        if (0x0008,0x1030) in initialdata:
            if 'cbct' or 'CBCT' in str(ds.StudyDescription):
                output_list, classUID, mimeType, recommendation_string = predictions_cbct(input_folder, output_folder)
            else:
                flag = cbct
                ###insert check for number of series from metadata
                for file in files_list:
                    d = pydicom.dcmread(file)
                    if str(initialdata.SeriesNumber) != d.SeriesNumber:
                        flag = ct
                        break
                if flag == ct:
                    output_list, classUID, mimeType, recommendation_string = predictions_ct(input_folder, output_folder)
                else:
                    output_list, classUID, mimeType, recommendation_string = predictions_cbct(input_folder, output_folder)
    else:
        output_list = classUID = mimeType = recommendation_string = []
    return output_list, classUID, mimeType, recommendation_string

