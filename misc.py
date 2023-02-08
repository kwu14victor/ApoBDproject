import os, sys, random, argparse, shutil, math, cv2
import numpy as np
from scipy import ndimage, misc
import pandas as pd
from skimage import io, exposure, transform, filters, morphology, measure, color, segmentation, feature
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy import ndimage as ndi
from math import pi as PI
from datetime import datetime
#from nanowell import *

from itertools import groupby, permutations
import copy
from scipy.signal import convolve2d
sys.path.append('/project/varadarajan/kwu14/proj/getvideo/CNN/')
from capsulelayers import CapsuleLayer, PrimaryCap, Mask, Length
from capsnet_KLW import margin_loss
from tensorflow.keras import models
import statistics
######################################################################
import tensorflow as tf
from numpy import stack, zeros, array, bool
from numpy import logical_or
from skimage.io import imsave
from skimage.exposure import rescale_intensity
import sys
import os


def MRCNN_ContactMap(model, images):
    outlist, contact_map, cont_index, maskseq, cellnums = [],[],[],[],[]
    output = np.zeros((len(images), 281,281,3)).astype(np.uint8)
    for index, image in enumerate(images):
        if len(image.shape)==2:
            image = np.stack([image,image,image],axis=-1)
        image_RS = exposure.rescale_intensity(image, out_range=(0,255)) 
        r2 = model.detect([image_RS], verbose=0)
        outmask, cont_mask = np.zeros((281,281)).astype(np.bool), np.zeros((281,281)).astype(np.uint8)
        output[index,:,:,:], cellnum, frame_seq, bbox_ind, maskpool, contours = image, 0, [], 0, [], np.zeros((281,281))#.astype(np.bool)#, coors = image, 0, [], 0, [], []
        
        for k in range(len(r2[0]['class_ids'])):
            if r2[0]['class_ids'][k]!=3 and np.sum((r2[0]['masks'][:,:,k]))>00 and np.std(image[r2[0]['masks'][:,:,k]])>10:
                this_coor = getseq(r2[0]['masks'][:,:,k], bbox_ind)[1:5]
                #if this_coor not in coors:
                if len(maskpool)==0 or (len(maskpool)>0 and max([getIOU(m, r2[0]['masks'][:,:,k]) for m in maskpool])<0.4):
                    outmask = np.logical_or(outmask, (r2[0]['masks'][:,:,k]))
                    output[index,:,:,:] =plot_contour(output[index,:,:,:], (r2[0]['masks'][:,:,k]), ch=0)
                    frame_seq.append(getseq(r2[0]['masks'][:,:,k], bbox_ind))
                    bbox_ind, cellnum, cont_mask = bbox_ind+1 , cellnum+1 , cont_mask+(r2[0]['masks'][:,:,k]).astype(np.uint8)
                    maskpool.append(r2[0]['masks'][:,:,k])
                    contours = get_contour_bi(contours, r2[0]['masks'][:,:,k], bbox_ind)
                    
        contours = contours.astype(np.uint8)        
        if np.amax(cont_mask)>1 or np.amax(measure.label(outmask))<cellnum:
            cont_index.append('Y')
        else:
            cont_index.append('N')
        cont_mask= np.where(cont_mask>1, True, False)
        contact_map.append(cont_mask)
        output[index,:,:,:] =plot_contour(output[index,:,:,:], cont_mask, ch=1)
        maskseq.append(frame_seq)
        cellnums.append(cellnum)
    return(np.array(output), np.array(contact_map), cont_index, maskseq, cellnums)

   
def seq2mask(seq, size=281):
    mask = np.zeros((size,size))
    mask[seq[1]:seq[1]+seq[3], seq[2]:seq[2]+seq[4]] = 255
    return(mask.astype(np.bool))
    
def getseq(mask, index):
    NU_seq = []
    single_region = measure.regionprops(measure.label(mask))[0]
    (min_row, min_col, max_row, max_col) = single_region.bbox
    NU_seq.append(index)
    NU_seq.append(min_row)
    NU_seq.append(min_col)
    NU_seq.append(max_row-min_row)
    NU_seq.append(max_col-min_col)
    NU_seq.append(index)
    return(NU_seq)

def get_contour_bi(image, mask, val):
    image_out = image
    m_label = mask==True
    if np.sum(m_label)>0:
        contours = measure.find_contours(m_label.astype(np.uint8)*255, 200)
        for cont in contours:
            for contour in cont:
                image_out[int(contour[0]), int(contour[1])]= val
    return(image_out)

def plot_contour(image, mask, ch=1):
    image_out = image
    if len(image_out.shape)==2:
        image_out = color.gray2rgb(image_out)
    m_label = mask==True
    if np.sum(m_label)>0:
        contours = measure.find_contours(m_label.astype(np.uint8)*255, 200)
        for cont in contours:
            for contour in cont:
                image_out[int(contour[0]), int(contour[1]), ch]=255
    return(image_out)

def getIOU(m1,m2):
    return(np.sum(np.logical_and(m1,m2))/np.sum(np.logical_or(m1,m2)))    

def pPCC(img1, img2):
    one = img1-np.mean(img1)
    two = img2-np.mean(img2)
    numerator = np.sum(one*two)
    denominator = (np.sum((one**2))*np.sum((two**2)))**0.5
    return(numerator/denominator)

def event_start_4(seq, condition, howlong, before):
    time = []
    labels, length = [k for k,g in groupby(seq)], [len(list(g)) for k,g in groupby(seq)]
    for ind, cla in enumerate(labels):
        if cla==condition:
            if length[ind]>=howlong and sum(length[:ind])<before:
                time.append(sum(length[:ind]))
    if len(time)==0:
        time.append(math.inf)
    return(time)
    
def ANV_time_blob(stack_raw, seq, cellmask=None, mode='FF'):
    if len(stack_raw.shape)==4:
        stack_raw = stack_raw[:,:,:,0]
    size, ANVnum, blobmasks = 281, [], []
    if size!=281:
        smoothANV = transform.resize(stack_raw, (73,size,size))
    else:
        smoothANV = stack_raw
    ind = np.argmax([np.std(smoothANV[k,:,:]) for k in range(73)])
    if mode=='FF' or mode=='FF_otsu':
        limit = filters.threshold_otsu(smoothANV)
        for k in range(len(stack_raw)):
            this_mask = np.zeros((size,size)).astype(np.bool)
            ANVs = feature.blob_log(smoothANV[k,:,:], min_sigma=1, max_sigma=35, threshold=0.1)
            ANVs = [BB for BB in ANVs if max([getIOU(blob2mask(BB, scale = 281/size), np.logical_and(seq2mask(ss),blob2mask(BB, scale = 281/size))) for ss in seq[k]])>0.2]
            ANVn = len(ANVs)
            for ANV in ANVs:
                if mode=='original':
                    this_mask = np.logical_or(this_mask, blob2mask(ANV, scale = 281/size))
                #0629 edited
                elif mode=='FF':
                    this_mask = np.logical_or(this_mask, blob2mask_FF(ANV, smoothANV[k,:,:], scale = 281/size))
            if mode=='FF_otsu':
                signal = smoothANV[k,:,:] 
                signal_mask = signal>max(filters.threshold_otsu(signal), limit)
                signal_mask = morphology.remove_small_objects(signal_mask, min_size=5)
                if np.sum(signal_mask)>np.sum(cellmask[k,:,:])*0.9:
                    signal_mask = np.logical_and(signal_mask, cellmask[k,:,:])
                    signal_mask = signal>filters.threshold_multiotsu(signal[cellmask[k,:,:]],3)[1]
                    signal_mask = morphology.remove_small_objects(signal_mask, min_size=5)
                    if np.sum(signal_mask)>np.sum(cellmask[k,:,:])*0.9:
                        signal_mask = np.zeros((281,281)).astype(np.bool)
                    signal_mask = morphology.binary_closing(signal_mask, morphology.disk(1))
                    signal_mask = morphology.remove_small_objects(signal_mask, min_size=5)
                signal_mask = np.logical_and(signal_mask, cellmask[k,:,:])
                signal_mask = morphology.remove_small_objects(signal_mask, min_size=5)
                this_mask = signal_mask
            blobmasks.append(this_mask)
            if ANVn<5 and ANVn>0:
                ANVa = (np.sum([(f[-1])**2 for f in ANVs]))
                if mode=='original':
                    ANVnum.append(ANVa)
                elif mode=='FF':
                    ANVnum.append(np.sum(this_mask))
                elif mode=='FF_otsu' and np.sum(this_mask)<np.sum(cellmask[k,:,:]):
                    ANVnum.append(np.sum(this_mask))
            else:
                ANVnum.append(0)
        try:
            ANVnum = ANVnum>np.array(min(filters.threshold_otsu(np.array(ANVnum)), 20))
        except:
            ANVnum = [False for f in ANVnum]
        ANVtime = event_start_4(ANVnum, True, 3, 73)
    
    elif mode=='otsu':
        limit = filters.threshold_otsu(smoothANV)
        for k in range(len(stack_raw)):
            this_mask = np.zeros((size,size)).astype(np.bool)
            signal = filters.median(smoothANV[k,:,:], morphology.disk(3))
            signal_mask = signal>max(filters.threshold_otsu(signal), limit)
            this_mask = np.logical_and(signal_mask, cellmask[k,:,:])
            blobmasks.append(this_mask)
            ANVnum.append(np.sum(this_mask))
        ANVnum = ANVnum>np.array(min(filters.threshold_otsu(np.array(ANVnum)), 20))
        ANVtime = event_start_4(ANVnum, True, 3, 73)
            
    
    if math.inf not in ANVtime:
        start = ANVtime[0]
        MIOUS = [getIOU(blobmasks[start], seq2mask(ss)) for ss in seq[start]]
        return(ANVtime, blobmasks, MIOUS)
    else:
        return(ANVtime, np.zeros((size,size)).astype(np.bool), 0)  
        
def blob2mask(blob, size=281, scale = 1):
    mask = np.zeros((size,size))
    cv2.circle(mask, (int(blob[1]*scale), int(blob[0]*scale)), int(blob[-1]*scale), 255, 3)
    out = segmentation.flood_fill(mask, (int(blob[0]*scale), int(blob[1]*scale)), 255)
    return(out.astype(np.bool))
    
def celltype_check_5(region, markerT, markerE):
    #markerT, markerE = (markerT/256).astype(np.uint8), (markerE/256).astype(np.uint8)
    try:
        TH_T = filters.threshold_otsu(markerT)
    except:
        TH_T = 0
    try:
        TH_E = filters.threshold_otsu(markerE)
    except:
        TH_E = 0
        
    maskT, maskE, regionmask = markerT>max(TH_T, 10), markerE>max(TH_E, 10), seq2mask(region)
    IOU_T, IOU_E = getIOU(regionmask, np.logical_and(regionmask, maskT)), getIOU(regionmask, np.logical_and(regionmask, maskE))
    
    if max(IOU_T, IOU_E)>0.1:
        return(np.argmax([IOU_T, IOU_E]))
    else:
        return(2) 
def get_dist(P1,P2):
    r1,c1 = P1
    r2,c2 = P2
    return(((r1-r2)**2+(c1-c2)**2)**0.5)
            
def get_sum_dist(region, ApoBDlabel):
    sum_dist = 0
    for a in range(np.amax(ApoBDlabel)):
        this_mask = ApoBDlabel==a+1
        R, C = np.mean(np.where(this_mask==True)[0]), np.mean(np.where(this_mask==True)[1])
        sum_dist+=(get_dist([R,C],[region[1]+0.5*region[3], region[2]+0.5*region[4]]))
    sum_dist = sum_dist/np.amax(ApoBDlabel)
    return(sum_dist)

def getpatch(img, region, out_size=51):
    size = max(region[3], region[4])
    patch = img[region[1]:region[1]+size,region[2]:region[2]+size]
    patch = transform.resize(patch, (out_size,out_size))
    return(patch)

def getcellnum_blob(marker, seq, CT='T'):
    nums = []
    for k in list(range(5)):
        frame = filters.median(exposure.rescale_intensity(marker[k,:,:], out_range = np.uint8), morphology.disk(3))
        frame_seq = seq[k]
        if len(frame_seq)>0:
            maxsize = int(max([max(f[3], f[4]) for f in frame_seq]))
            #maxsize=int(min([min(f[3], f[4]) for f in frame_seq])*0.5)
            minsize = 10#min([min(f[3], f[4]) for f in frame_seq])
        else:
            maxsize = 50
            minsize = 10
        if CT=='E' and len(frame_seq)>0:
            maxsize=int(min([min(f[3], f[4]) for f in frame_seq])*0.5)
        blobs = feature.blob_log(frame, min_sigma=minsize, max_sigma=maxsize, threshold=0.1)
        #blobs_new = blobs
        
        blobs_new = [BB for BB in blobs if max([getIOU(blob2mask(BB), np.logical_and(seq2mask(s), blob2mask(BB))) for s in frame_seq])>0.3]

        nums.append(len(blobs_new))
    try:
        #ans = statistics.mode(nums)
        ans = max(nums)
    except:
        ans = int(np.mean(nums))
    return(ans)
        
class Basic_Tracker:
    def __init__(self, OUTPUT_PATH, DATASET, BLOCK, NANOWELL, FRAMES, DETECTOR_TYPE, CELL_TYPE, CELL_COUNT, sequence):
        self.OUTPUT_PATH, self.DATASET, self.BLOCK, self.NANOWELL, self.FRAMES, self.CELL_TYPE, self.CELL_COUNT, self.DETECTOR_TYPE  = OUTPUT_PATH, DATASET, BLOCK, NANOWELL, FRAMES, CELL_TYPE, CELL_COUNT, DETECTOR_TYPE

        self.TRACKER_TYPE, self.label_sequence, self.output_label_sequence = 'EZ', copy.deepcopy(sequence), copy.deepcopy(sequence)
        #copy.deepcopy(sequence), 

    def Run_Tracking(self):
        ### Step 1: Run LAP for each time step
        for t in range(self.FRAMES-1):
            self.LAP(t)
    ### Step 2: Write all the tracks to the file 
        return(self.output_label_sequence)
        #self.write_tracks()
        
    def get_detected_cell_current(self, t, N):
        x0, y0, w, h = self.output_label_sequence[t][N][1], self.output_label_sequence[t][N][2], self.output_label_sequence[t][N][3], self.output_label_sequence[t][N][4]
        
        xc, yc = x0 + w/2.0, y0 + h/2.0
        if w<4 or h<4:
            zc = 0
        else:
            zc = 1
        
        return [xc, yc, zc]
    
    def get_detected_cell_next(self, t, N):
        x0, y0, w, h = self.output_label_sequence[t+1][N][1], self.output_label_sequence[t+1][N][2], self.output_label_sequence[t+1][N][3], self.output_label_sequence[t+1][N][4]
        
        xc, yc = x0 + w/2.0, y0 + h/2.0
        if w<4 or h<4:
            zc = 0
        else:
            zc = 1
        
        return [xc, yc, zc]
        
    def get_current_state(self, t):
        state_0 = []
        for N in range(self.CELL_COUNT):
            temp = self.get_detected_cell_current(t, N)
            state_0.append(temp)
        
        return state_0
    
    def get_current_speed(self, t):
        speed_0 = []
        if t == 0:
            for N in range(self.CELL_COUNT):
                temp = [0,0]
                speed_0.append(temp)
            
        if t > 0:   ##### This could result problems
            for N in range(self.CELL_COUNT):
                temp1 = self.get_detected_cell_current(t,N)
                temp0 = self.get_detected_cell_current(t-1,N)
                if temp1[2] > 0 and temp0[2] > 0:
                    vx = temp1[0] - temp0[0]
                    vy = temp1[1] - temp0[1]
                else:
                    vx = 0
                    vy = 0
                speed_0.append([vx, vy])
            
        return speed_0       
                        
        
    def get_next_state(self, t):
        state_1 = []
        for N in range(self.CELL_COUNT):
            temp = self.get_detected_cell_next(t, N)
            state_1.append(temp)
        
        return state_1
    
        
    def predict_next_state(self, t):
        '''
        the simplest prediction of next state is to add the position with speed*decay(0.5)
        '''
        state_0 = self.get_current_state(t)
        speed_0 = self.get_current_speed(t)
        
        state_1_predict = np.array(state_0)
        
        for N in range(self.CELL_COUNT):
            state_1_predict[N][0] += speed_0[N][0]*0.5
            state_1_predict[N][1] += speed_0[N][1]*0.5
            
        return state_1_predict
            
             
    def LAP(self, t):
        '''
        STEP 1: Calculate the LINK COST MATRIX
        STEP 2: PARSE the MATRIX to get the MAPPING Relation
        STEP 3: Update output_label_sequence
        '''
        ### STEP 1: GET PAA
        PAA = np.zeros((self.CELL_COUNT, self.CELL_COUNT))
        
        state_1_predict = self.predict_next_state(t)
        state_1 = self.get_next_state(t)
        
        
        ### trace back and get effective record for each cell
        missing_cells = [1 for i in range(self.CELL_COUNT)]
        t0 = t
        
        for i in range(self.CELL_COUNT):
            if state_1_predict[i][2] > 0:
                missing_cells[i] = 0
        
        while sum(missing_cells) > 0 and t0 > 0:
            t0 = t0 - 1
            state_0_predict = self.predict_next_state(t0)
            
            for i in range(self.CELL_COUNT):
                if state_1_predict[i][2] == 0:
                    if state_0_predict[i][2] > 0:
                        missing_cells[i] =0
                        state_1_predict[i] = state_0_predict[i]
            
        
        ### calculate the cost MATRIX
        for i in range(self.CELL_COUNT):
            for j in range(self.CELL_COUNT):
                if state_1_predict[i][2]>0 and state_1[j][2]>0:
                    dx = state_1_predict[i][0] - state_1[j][0]
                    dy = state_1_predict[i][1] - state_1[j][1]
                    PAA[i][j] = -(dx*dx + dy*dy)
                else:
                    PAA[i][j] = -160000
        
        
        ### STEP 2: PARSE PAA to get ASSO
        ASSO = self.PAS(PAA)
        #print(ASSO, PAA)
        ### STEP 3: UPDATE output_label_sequence
        for i in range(self.CELL_COUNT):
            self.output_label_sequence[t+1][i] = self.label_sequence[t+1][ASSO[i]]
            self.output_label_sequence[t+1][i][0] = i#i+1
        


    def PAS(self, PAA):
        '''
        Generate the track mapping results based on Patch Association Array (PAA)
        Input:  PAA A1  A2  A3
                C1  p11 p12 p13
                C2  p21 p22 p23
                C3  p31 p32 p33
        Output:
            ASSO: [0,2,1]' which means 0-->0, 1-->2, 2-->1
        '''
        n = PAA.shape[0]
        temp = range(n)

        perms = list(permutations(temp))
        scores = []
        for perm in perms:
            score = 0
            for i in range(n):
                score += PAA[i, perm[i]]
            scores.append(score)

        index = np.argmax(scores)

        return np.array(perms[index]) 
  