import os, sys, random, argparse, shutil, math, cv2
sys.path.append('/project/varadarajan/kwu14/proj/gen_mask/')
sys.path.append('/project/varadarajan/kwu14/proj/getvideo/CNN/')
sys.path.append('/project/varadarajan/kwu14/proj/getvideo/')
import numpy as np
from scipy import ndimage, misc
import pandas as pd
from skimage import io, exposure, transform, filters, morphology, measure, color
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy import ndimage as ndi
from math import pi as PI
from datetime import datetime
from nanowell import *
from FromTiming import *
from itertools import groupby, permutations
import copy
from scipy.signal import convolve2d
from capsulelayers import CapsuleLayer, PrimaryCap, Mask, Length
from capsnet_KLW import margin_loss
from tensorflow.keras import models
import statistics

    
    
    
def pipeline(args):
    pfiletype, dshome, frames = args.dtype, args.dhome, args.frames   
    config = CellsInferenceConfig(args.dhome)
    config.DETECTION_MIN_CONFIDENCE, config.DETECTION_NMS_THRESHOLD = 0.5, 0.7
    model = modellib.MaskRCNN(mode="inference",config=config,model_dir=mrcnnhome)
    model.load_weights(args.weight_name, by_name=True)
    ApoBD_detect(args, model)


def MRCNN_ApoBDs(model, images):
    outmasks, areas, nums, cellnums, rimsize, rs2, rimmask, rmask2  = [], [], [], [], 65, 40, np.zeros((281,281)).astype(np.bool), np.zeros((281,281)).astype(np.bool)
    rimmask[rimsize:-rimsize, rimsize:-rimsize]=True
    rmask2[rs2:-(rs2), rs2:-(rs2)]=True
    
    for image in images:
        if len(image.shape)==2:
            image = np.stack([image,image,image],axis=-1)
        image_RS = exposure.rescale_intensity(image, out_range=(0,255)) 
        r2 = model.detect([image_RS], verbose=0)
        outmask = np.zeros((281,281)).astype(np.bool)
        area, num, cellnum = 0, 0, 0
        
        for k in range(len(r2[0]['class_ids'])):
            if r2[0]['class_ids'][k]==3 and np.amax(np.logical_and(np.logical_not(rimmask), r2[0]['masks'][:,:,k]))==False and len(np.where(r2[0]['masks'][:,:,k]==True)[0])>10 and np.std(image[r2[0]['masks'][:,:,k]])>5.5:
                outmask = np.logical_or(outmask, (r2[0]['masks'][:,:,k]))
                area+= len(np.where((r2[0]['masks'][:,:,k])==True)[0])
                num+=1
            elif r2[0]['class_ids'][k]!=3 and np.amax(np.logical_and(np.logical_not(rmask2), r2[0]['masks'][:,:,k]))==False and len(np.where(r2[0]['masks'][:,:,k]==True)[0])>1000:
                cellnum+=1
        areas.append(area)
        nums.append(num)        
        outmasks.append(outmask)
        cellnums.append(cellnum)
    return(areas, nums, cellnums, np.array(outmasks))


def ApoBD_detect(args, model):            
    pfiletype, dshome, frames = args.dtype, args.dhome, args.frames
    for dsname in args.ds:
        now = (datetime.now()).strftime('%m')+(datetime.now()).strftime('%d')+(datetime.now()).strftime('%H')+(datetime.now()).strftime('%M')
        moviehome = os.path.join(os.getcwd(), args.expname)
        dead_dir = os.path.join(moviehome, 'dead_raw')
        os.mkdir(moviehome)
        os.mkdir(dead_dir)
        dh, NWS = dshome + dsname +'/', []
        print([b for b in os.listdir(dh) if 'B' in b])
        for b_no in [b for b in os.listdir(dh) if 'B' in b][:args.endpoint]:
            print(b_no)
            t_dir, i_dir = os.path.join(dh, b_no, 'labels/TRACK/EZ/FRCNN-Fast/'), os.path.join(dh, b_no, 'images/crops_8bit_s/')
            if os.path.exists(t_dir):
                for i_no in os.listdir(t_dir):
                    NW = nanowell(b_no, dh, i_no)
                    if NW.Tnum>=1 and NW.Enum>=0:
                        NWS.append(NW)
        
        for well in NWS:
            img= (well.get_stack('phase')/256).astype(np.uint8)
            areas, nums, cell_nums, mask = MRCNN_ApoBDs(model, list(img))
            
            if np.amin(cell_nums)>=1:
                
                generate = False

                if np.amax(nums)>=3:
                    generate = True

                if generate:
                
                    ABs = np.array(nums)
                    ABs = ABs>filters.threshold_otsu(np.array(nums))
                    DorA, noF, dead, D_start = [k for k,g in groupby(ABs)], [len(list(g)) for k,g in groupby(ABs)], False, args.frames
                    for ind, cla in enumerate(DorA):
                        if cla==True:
                            if noF[ind]>=3: #and ind>0 and DorA[-1]==1:
                                D_start = sum(noF[:ind])
                                try:
                                    D_end = sum(noF[:ind+1])
                                except:
                                    D_end = sum(noF[:-1])
                                dead = True
                                break
                    if dead:
                        well.generate_tif(os.path.join(dead_dir, dsname+ well.bnum + well.inum + '_die_at_frame' + str(D_start) + '_.tif'))
                        img = color.gray2rgb(img)
                        img[:,:,:,0] = np.where(mask==True, 255, img[:,:,:,0])
                        img[:,:,:,2] = np.where(mask==True, 255, img[:,:,:,2])
                        io.imsave(os.path.join(dead_dir, dsname+ well.bnum + well.inum + '_die_at_frame' + str(D_start) + '_masked_.tif'), img)

 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="annotating TIMING processed data")
    parser.add_argument('--dtype', default='.tif', type=str)
    parser.add_argument('--dhome', default='/project/varadarajan/kwu14/DT-HPC/', type=str)
    parser.add_argument('--frames', default=73, type=int)
    parser.add_argument('--weight_name', default='/project/varadarajan/kwu14/Mask_RCNN/logs/ApoBD_all/0419_ApoBD_iter1_0077.h5', type=str)
    parser.add_argument('--expname', default='debug_1', type=str)
    #for AnnV measurement
    parser.add_argument('--ds', '--names-list', nargs='+', default=[])
    parser.add_argument('--endpoint', default=-1, type=int)
    args = parser.parse_args()
    
    
    #if args.mode == 'ANXV_filter':
    pipeline(args)    
    