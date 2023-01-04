import os, sys, random, argparse, shutil, math, cv2
sys.path.append('/project/varadarajan/kwu14/proj/gen_mask/')
sys.path.append('/project/varadarajan/kwu14/proj/getvideo/CNN/')
sys.path.append('/project/varadarajan/kwu14/proj/getvideo/')

import numpy as np
from scipy import ndimage, misc
import pandas as pd
from skimage import io, exposure, transform, filters, morphology, measure, color, feature
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
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
from contact_and_ApoBD_pos import *

def pipeline(args):
    ####call DL models as instances####
    config, config2 = CellsInferenceConfig(os.getcwd()), CellsInferenceConfig(os.getcwd())
    config.DETECTION_MIN_CONFIDENCE, config.DETECTION_NMS_THRESHOLD, config2.DETECTION_MIN_CONFIDENCE, config.DETECTION_NMS_THRESHOLD = 0.6,0.3,0.5,0.7#0.7, 0.5, 0.5, 0.7
    model, model2 = modellib.MaskRCNN(mode="inference",config=config,model_dir=mrcnnhome), modellib.MaskRCNN(mode="inference",config=config2,model_dir=mrcnnhome)
    model.load_weights(args.MRCNNweights1, by_name=True)
    model2.load_weights(args.MRCNNweights2, by_name=True)
    model3 = models.load_model('/project/varadarajan/kwu14/proj/getvideo/CNN/0201Caps_RWM_L2.h5', custom_objects={"CapsuleLayer": CapsuleLayer, "Mask": Mask, "Length": Length, "margin_loss": margin_loss})
    I, O = model3.layers[0].input, model3.layers[-2].output
    detector = models.Model(I,O)
    
    now = (datetime.now()).strftime('%m')+(datetime.now()).strftime('%d')+(datetime.now()).strftime('%H')+(datetime.now()).strftime('%M')
    ####setup directories####
    if args.vidinput=='None':
        GCmap_dir, names = os.path.join(os.getcwd(), args.expname, 'focus_map'), [ f for f in os.listdir(os.path.join(os.getcwd(), args.expname, 'dead')) if '.tif' in f]
        
    elif args.vidinput!='None':
        GCmap_dir, names = os.path.join(os.getcwd(), args.vidinput, 'focus_map'), [ f for f in os.listdir(os.path.join(os.getcwd(), args.vidinput, 'dead')) if '.tif' in f ]  
    #names = [f for f in names if 'B015imgNo17' in f]
    names.sort()
    ####output home directories####
    savedir, outputdir = os.path.join(os.getcwd(), args.expname), os.path.join(os.getcwd(), args.expname, now)
    ####event type directories####
    non_dir, kill_dir, kill_dir_r, death_dir, S_dir, C_dir = os.path.join(outputdir, 'non'), os.path.join(outputdir, 'kill'), os.path.join(outputdir, 'kill_raw'), os.path.join(outputdir, 'death'), os.path.join(outputdir, 'synpase'), os.path.join(outputdir, 'cell')
    ####event type directories####
    AnnV_first_dir, ApoBD_first_dir, E_death_dir, T_death_dir = os.path.join(outputdir, 'AnnV_first'), os.path.join(outputdir, 'ApoBD_first'), os.path.join(outputdir, 'E_dead'), os.path.join(outputdir, 'T_dead')
    ####meta source directories####
    labeled_dir, cont_map_dir, cont_index_dir, mask_seq_dir, cell_num_dir, marker_dir = os.path.join(savedir, 'labeled'), os.path.join(savedir, 'contact_map'), os.path.join(savedir, 'contact_index'), os.path.join(savedir, 'mask_seq'), os.path.join(savedir, 'cell_num'), os.path.join(savedir, 'marker_mask')
    ####ApoBD meta directories####
    ApoBD_num_dir, ApoBD_mask_dir, ANNV_time_dir = os.path.join(savedir, 'ApoBD_num'), os.path.join(savedir, 'ApoBD_mask'), os.path.join(savedir, 'ANNV_time')
    ####create directories####
    if args.vidinput!='None':
        os.mkdir(savedir)
    if args.generate_all:
        for dr in [outputdir, non_dir, kill_dir, kill_dir_r, death_dir, S_dir, C_dir, AnnV_first_dir, ApoBD_first_dir, labeled_dir, cont_map_dir, cont_index_dir, mask_seq_dir, cell_num_dir, marker_dir, ApoBD_num_dir, ApoBD_mask_dir, ANNV_time_dir, E_death_dir, T_death_dir]:
            os.mkdir(dr)
    if args.just_get_table:
    #needs to be changed here!!!
        for dr in [outputdir, non_dir, kill_dir, kill_dir_r, death_dir, S_dir, C_dir, AnnV_first_dir, ApoBD_first_dir, E_death_dir, T_death_dir]:
            os.mkdir(dr)
            
    row1,row2,row3,row4,row5, row6, row7, row8, row9, row10, row11, row12, row13, row14, row15, row16, row17,row18,row19, row20, row21,row22,row23,row24,row25,row26,row27 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[], [], [],[],[],[],[],[],[]
    if args.endpoint!=99999:
        endpoint = args.endpoint
    else:
        endpoint = len(names)
    OTSUdiff=10
    video_len=98
    for n in names[:endpoint]:
        try:
    ##STEP1: get all files (images, masks,...)
            print(n)
            ####get images####
            if args.vidinput=='None':
                testinput = io.imread(os.path.join(os.getcwd(), args.expname, 'dead',n))
            else:
                testinput = io.imread(os.path.join(os.getcwd(), args.vidinput, 'dead',n))
            NW = nanowell(n.split('img')[0][-4:], os.path.join(args.DT_home, n.split('img')[0][:-4]), 'img'+n.split('img')[1].split('_die')[0])
            markerT, markerE, markerANV = NW.get_stack('T'), NW.get_stack('E'), (NW.get_stack('ANXV')/256).astype(np.uint8)
            markerANV_out,BmaskT, BmaskE, synapse_mask, allcell_mask = color.gray2rgb((NW.get_stack('ANXV')/256).astype(np.uint8)),np.zeros((video_len,281,281)).astype(np.bool),np.zeros((video_len,281,281)).astype(np.bool), np.zeros((video_len,281,281)).astype(np.bool), np.zeros((video_len,281,281)).astype(np.bool)
                
    
    ####generate or load inference result####
            
            if args.generate_all:
                A, B, C, maskseq, CN = MRCNN_ContactMap(model, testinput)
                _,ApoBDnums,_,D1 = MRCNN_ApoBDs(model2, testinput)
                
                io.imsave(os.path.join(labeled_dir, n.replace('.tif','_cell_contour.tif')), A)
                np.save(os.path.join(cont_map_dir, n.replace('.tif','_contact_map.npy')), B)
                np.save(os.path.join(cont_index_dir, n.replace('.tif','_contact_label.npy')), np.array(C))
                np.save(os.path.join(mask_seq_dir, n.replace('.tif','_mask_seq.npy')), np.array(maskseq))
                np.save(os.path.join(cell_num_dir, n.replace('.tif','_cell_num.npy')), np.array(CN))
                np.save(os.path.join(ApoBD_num_dir, n.replace('.tif','_ApoBD_num.npy')), np.array(ApoBDnums))
                np.save(os.path.join(ApoBD_mask_dir, n.replace('.tif','_ApoBD_mask.npy')), D1)
    
            
            else:
                if args.just_get_table:
                    A = io.imread(os.path.join(labeled_dir, n.replace('.tif','_cell_contour.tif')))
                    #B = np.load(os.path.join(cont_map_dir, n.replace('.tif','_contact_map.npy')))
                    B = synapse_mask
                    #C = list(np.load(os.path.join(cont_index_dir, n.replace('.tif','_contact_label.npy'))))
                    
                    maskseq = list(np.load(os.path.join(mask_seq_dir, n.replace('.tif','_mask_seq.npy')), allow_pickle=True))
                    CN = list(np.load(os.path.join(cell_num_dir, n.replace('.tif','_cell_num.npy'))))
                    ApoBDnums = list(np.load(os.path.join(ApoBD_num_dir, n.replace('.tif','_ApoBD_num.npy'))))
                    D1 = np.load(os.path.join(ApoBD_mask_dir, n.replace('.tif','_ApoBD_mask.npy')))
                    
            for k in range(len(markerT)):
                EE = filters.median(markerE[k,:,:], morphology.disk(3))
                TT = filters.median(markerT[k,:,:], morphology.disk(3))
                BmaskE[k,:,:] = (EE>filters.threshold_otsu(EE))
                BmaskT[k,:,:] = (TT>filters.threshold_otsu(TT))
                synapse_mask[k,:,:] = np.logical_and(BmaskE[k,:,:], BmaskT[k,:,:])
                for seq in maskseq[k]:
                    allcell_mask[k,:,:] = np.logical_or(allcell_mask[k,:,:], seq2mask(seq))
            C = []
            for ff in range(len(B)):
                if np.amax(B[ff,:,:])==True:
                    C.append('Y')
                else:
                    C.append('N')
                    
    ####get ApoBD time####        
            if args.ApoBD_Time=='CNN':
                TApoBD = (int(n.split('says_')[1].split('.tif')[0]))
            elif args.ApoBD_Time=='MRCNN':
                TApoBD = (int(n.split('die_at_frame')[1].split('__')[0]))
            elif args.ApoBD_Time=='dual':
                TApoBD = min((int(n.split('die_at_frame')[1].split('__')[0])), (int(n.split('says_')[1].split('.tif')[0])))
            
            if ApoBDnums[TApoBD]<=1:
                counter=0
                while ApoBDnums[TApoBD]<=1 and TApoBD<video_len and TApoBD>=0:
                    TApoBD=TApoBD+((-1)**counter)*(counter+1)
                    counter+=1
    ####get contact time####
            Tconts = event_start_4(C, 'Y', 3, TApoBD)
            for k in range(len(D1)):
                A[k,:,:,:] = plot_contour(A[k,:,:,:], D1[k,:,:], ch=2)
            print('Tcontact below')
            print(Tconts)
            found = False
    ####get AnnV time####        
            ANVtime, ANVmask, MIOUS = ANV_time_blob(markerANV, maskseq, allcell_mask, mode='FF_otsu')
    ####split maskseq####
            T_seq, E_seq, both_seq = [],[],[]
            for frame, seqs in enumerate(maskseq):
                #print('Frame no. '+str(frame))
                frame_T, frame_E, frame_both, T_index, E_index, B_index = [],[],[],0,0,0
                for seq in seqs:
    
                    if celltype_check_5(seq, markerT[frame,:,:], markerE[frame,:,:])==0:
                        frame_T.append([T_index, seq[1], seq[2], seq[3], seq[4], T_index])
                        frame_both.append([B_index, seq[1], seq[2], seq[3], seq[4], B_index])
                        T_index+=1
                        B_index+=1
                    
                    elif celltype_check_5(seq, markerT[frame,:,:], markerE[frame,:,:])==1:
                        frame_E.append([E_index, seq[1], seq[2], seq[3], seq[4], E_index])
                        frame_both.append([B_index, seq[1], seq[2], seq[3], seq[4], B_index])
                        E_index+=1
                        B_index+=1
                T_seq.append(frame_T)
                E_seq.append(frame_E)
                both_seq.append(frame_both)
                #both_seq.append(frame_T+frame_E)
                            
    ####generate AnnV visualization and mark AnnV cell in sequence####                
            print(ANVtime)
            if math.inf not in ANVtime:
                start = ANVtime[0]-2
                AnnvCell = maskseq[ANVtime[0]][np.argmax(MIOUS)]
                dead_class_AnnV = ['T','E','not_sure'][celltype_check_5(AnnvCell, markerT[ANVtime[0],:,:], markerE[ANVtime[0],:,:])]
                
                print('AnnV is found in '+ dead_class_AnnV+' cell')
                if start>0 and start+4<video_len:
                    visual, visual2  = [markerANV[k,:,:] for k in range(start,start+5)], [testinput[k,:,:] for k in range(start,start+5)]
                    visual, visual2 = np.concatenate([a for a in visual], axis=-1), np.concatenate([a for a in visual2], axis=-1)
                    
                    ANVR, ANVC = int(AnnvCell[1]+0.5*AnnvCell[3]) , int(AnnvCell[2]+0.5*AnnvCell[4])
                    A[ANVtime[0], ANVR-5:ANVR+5, ANVC-5:ANVC+5]=[255,125,0]
                    
                    MIOUS.sort()
                    MIOUS = MIOUS[::-1]
                    
                    if len(MIOUS)==1 or MIOUS[0]-MIOUS[1]>0.1:
                        vis_dir = C_dir
                    else:
                        vis_dir = S_dir 
       
    ##STEP2: decide which cell is generating ApoBDs
            ApoBDcells, ApoBDindexes, candidate_seq, candidates, start = [],[],[],[],-2
            time = TApoBD+start
            while len(ApoBDindexes)<3:
                try:
                    ApoBDseq, ApoBDmask = [sq for sq in both_seq[time]], D1[time,:,:]
                    if np.amax(ApoBDmask)==True:
                        ApoBDindexes.append(time)
                        ApoBDlabel = measure.label(ApoBDmask)
                        mindist = math.inf    
                        dists = [get_sum_dist(region, ApoBDlabel) for region in ApoBDseq]
                        ApoBDcell = ApoBDseq[np.argmin(dists)]
                        visualize = np.stack([testinput[time],testinput[time],testinput[time]], axis=-1)
                        R, C = int(ApoBDcell[1]+0.5*ApoBDcell[3]), int(ApoBDcell[2]+0.5*ApoBDcell[4])
                        visualize[R-5:R+5,C-5:C+5]=[255,255,0]
                        ApoBDcells.append(ApoBDcell)
                        candidate_seq.append(both_seq[time])
                        #io.imsave(n+str(time)+'visualize.png',visualize)
                except:
                    print('missing frame no. '+str(time))
                time+=1
            
            #print('ApoBD candidates are: ', ApoBDcells)
            tracker_new = Basic_Tracker(os.getcwd(), '', '', '', len(ApoBDindexes), '', 'dummy', np.amin([len(s) for s in candidate_seq]), candidate_seq)
            tracked_new = tracker_new.Run_Tracking()
            
            for k in range(len(ApoBDindexes)):
                candidates.append([seq[0] for seq in tracked_new[k] if seq[1:-1]==ApoBDcells[k][1:-1]][0])
            print(candidates)
            try:
                ApoBDcell_id = statistics.mode(candidates)
                ApoBD_frame = ApoBDindexes.index(TApoBD) 
                visualize = np.stack([testinput[TApoBD],testinput[TApoBD],testinput[TApoBD]], axis=-1)
                ApoBDcell = tracked_new[ApoBD_frame][ApoBDcell_id]
                R, C = int(ApoBDcell[1]+0.5*ApoBDcell[3]), int(ApoBDcell[2]+0.5*ApoBDcell[4])
                visualize[R-5:R+5,C-5:C+5]=[255,255,0]
                #io.imsave(n+'findingApoBDcell_voted_'+'visualize.png',visualize)
            except:
                ApoBD_frame = ApoBDindexes.index(TApoBD)
                ApoBDcell_id = candidates[ApoBD_frame]
                visualize = np.stack([testinput[TApoBD],testinput[TApoBD],testinput[TApoBD]], axis=-1)
                ApoBDcell = tracked_new[ApoBD_frame][ApoBDcell_id]
                R, C = int(ApoBDcell[1]+0.5*ApoBDcell[3]), int(ApoBDcell[2]+0.5*ApoBDcell[4])
                visualize[R-5:R+5,C-5:C+5]=[255,255,0]
                #io.imsave(n+'findingApoBDcell_not_voted_'+'visualize.png',visualize)
            
            ApoBDcell_pureMRCNN = ApoBDcell
            
            
            ApoBDseq = [sq for sq in maskseq[TApoBD] if celltype_check_5(sq, markerT[TApoBD,:,:], markerE[TApoBD,:,:])!=2]
            #print(ApoBDseq)
            ApoBDmask = D1[TApoBD,:,:]
            ApoBDlabel = measure.label(ApoBDmask)
            mindist, ApoBDcell = math.inf, 0
            dists = [get_sum_dist(region, ApoBDlabel) for region in ApoBDseq]#maskseq[TApoBD]]
            ApoBDcell = ApoBDseq[np.argmin(dists)]#maskseq[TApoBD][np.argmin(dists)]
            
    ####Find the cell that got the most attention from CNN####
            GCmask = markerANV[TApoBD,:,:] 
            #io.imread(os.path.join(GCmap_dir, n.replace('.tif','_GradCam.tif')))[TApoBD,:,:]
            GCval = [np.mean(GCmask[region[1]:region[1]+region[3],region[2]:region[2]+region[4]]) for region in ApoBDseq]#maskseq[TApoBD]]
            GCcell = ApoBDseq[(np.argmax(GCval))] #maskseq[TApoBD][(np.argmax(GCval))]
            ####Find the cell got the highest apoptosis score from CapsNet####
            Caps_in = []
            for region in ApoBDseq:#maskseq[TApoBD]:
                Caps_in.append(getpatch(testinput[TApoBD,:,:], region))
            while len(Caps_in)<16:
                Caps_in.append(np.zeros((51,51)))
            Caps_in = np.array(Caps_in).astype(np.float32)
            Caps_result = detector.predict(Caps_in)[:len(ApoBDseq),0]#len(maskseq[TApoBD]),0]
            Capscell = ApoBDseq[np.argmax(Caps_result)] #maskseq[TApoBD][np.argmax(Caps_result)]
            
            #cont_index = Tcont
            APOR, APOR1, APOR2 = int(ApoBDcell[1]+0.5*ApoBDcell[3]), int(GCcell[1]+0.5*GCcell[3]), int(Capscell[1]+0.5*Capscell[3])
            APOC, APOC1 ,APOC2 = int(ApoBDcell[2]+0.5*ApoBDcell[4]), int(GCcell[2]+0.5*GCcell[4]), int(Capscell[2]+0.5*Capscell[4])
            
            A[TApoBD, APOR-5:APOR+5, APOC-5:APOC+5, 2]=255
            A[TApoBD, APOR1-5:APOR1+5, APOC1-5:APOC1+5, 0]=255
            A[TApoBD, APOR2-5:APOR2+5, APOC2-5:APOC2+5, 1]=255
            
            candidate = [np.argmax(Caps_result), np.argmax(GCval), np.argmin(dists)]
            try:
                ApoBDcell_ori = ApoBDseq[statistics.mode(candidate)] #maskseq[TApoBD][statistics.mode(candidate)]
            except:
                ApoBDcell_ori = ApoBDcell
            
            if args.ApoBDmode== 'pure_MRCNN':
            #if try_pure_MRCNN:
                ApoBDcell_ori = ApoBDcell_pureMRCNN
            
            APOR, APOC = int(ApoBDcell_ori[1]+0.5*ApoBDcell_ori[3]) , int(ApoBDcell_ori[2]+0.5*ApoBDcell_ori[4])
            A[TApoBD, APOR-15:APOR-5, APOC-15:APOC-5, :]=255
            ACindex = ApoBDcell[0]
            dead_class = ['T','E'][celltype_check_5(ApoBDcell_ori, markerT[TApoBD,:,:], markerE[TApoBD,:,:])]
            Tnum2,Enum2 = getcellnum_blob((markerT/256).astype(np.uint8), maskseq), getcellnum_blob((markerE/256).astype(np.uint8), maskseq, 'E')
            
            compare_ApoBD = np.concatenate([visualize, A[TApoBD]], axis=1)
            #io.imsave(n+'_ApoBDcompare.png', compare_ApoBD)
    
    ##STEP3: to see if AnnV lights up in this nanowell
            RR13 = 'NAN'
            RR15, RR19, RR20, RR21,RR22,RR23,RR24,RR25 = 0,0,-99,0,-99,0,-99,0
            AnVpostApoBD=0
            AnV_IOU_rest = 0
            mask_stack_s, mask_stack_c = np.zeros(B.shape).astype(np.bool), np.zeros(B.shape).astype(np.bool)
            if math.inf not in ANVtime:
                secondD = TApoBD#min(TApoBD, ANVtime[0]), max(TApoBD, ANVtime[0])
                firstD = 0
                while firstD<TApoBD and RR13=='NAN':
                    if dead_class=='T':
                        trackerD = Basic_Tracker(os.getcwd(), '', '', '', secondD-firstD+1, '', 'dummy', np.amin([len(s) for s in T_seq[firstD:secondD+1]]), T_seq[firstD:secondD+1])
                    elif dead_class=='E':
                        trackerD = Basic_Tracker(os.getcwd(), '', '', '', secondD-firstD+1, '', 'dummy', np.amin([len(s) for s in E_seq[firstD:secondD+1]]), E_seq[firstD:secondD+1])
                    trackedD = trackerD.Run_Tracking()
                    frameApoBD = trackedD[-1]
                    ApoBD_cell_index = [f[0] for f in frameApoBD if f[1:-1]==ApoBDcell_ori[1:-1]]
                    assert len(ApoBD_cell_index)==1
                    ApoBD_cell_index, start  = ApoBD_cell_index[0], 0 
                    while start<len(trackedD):
                        #print('checking frame no. '+str(start))
                        try:
                            ANV_f = ANVmask[start]
                            cell_f = trackedD[start][ApoBD_cell_index]
                            #mask_stack_c[start,:,:] = seq2mask(cell_f)
                            #mask_stack_s[start,:,:] = np.logical_and(seq2mask(cell_f), B[start,:,:])
                            #mask_stack_s[start,:,:] = screen_region(B[start,:,:], seq2mask(cell_f))
                            #print('found before TApoBD')
                            IOU_f = getIOU(seq2mask(cell_f), np.logical_and(seq2mask(cell_f), ANV_f))
                            Th1 = filters.threshold_otsu(markerANV[start,:,:])
                            Th2 = filters.threshold_otsu(markerANV[start,:,:][seq2mask(cell_f)])
                            #if Th2-Th1>=OTSUdiff:
                            if IOU_f>0.1 and start>=min(ANVtime) and RR13=='NAN' and pPCC(seq2mask(cell_f).astype(np.uint8), markerANV[start])>0.1:
                                print('ApoBD cell already dead!!!!!!')
                                RR13=start
                                break
                            
                            
                            #contact_f = np.logical_and(B[start,:,:], seq2mask(cell_f))
                            
                            #test_ANV = markerANV[start,:,:]
                            
                            #cell_rest = [cell for cell in trackedD[start] if cell!=cell_f]+[cell for cell in rest_seq[start+TApoBD]]
                            
                            #for cr in cell_rest:
                            #    test_ANV = np.where(seq2mask(cr)==True, np.amin(test_ANV), test_ANV)
                            
                            #RR15 = max(RR15, IOU_f)
                            #IOU_c = getIOU(contact_f, np.logical_and(contact_f, ANV_f))
                            #PCC_cont = pPCC(contact_f.astype(np.uint8), test_ANV)#ANV_f.astype(np.uint8)))
                            #PCC_cell = pPCC(seq2mask(cell_f).astype(np.uint8), test_ANV) 
                            #print('pcc check', PCC_cont, PCC_cell)       
                            #if np.amax(contact_f)==True:
                            #    RR20 = [start,RR20][np.argmax([IOU_c, RR19])]
                            #    RR19 = max(IOU_c, RR19)
                            #    RR22 = [start, RR22][np.argmax([PCC_cont-PCC_cell, RR21])]
                            #    RR21 = max(PCC_cont-PCC_cell,RR21)
                            #    RR24 = [start, RR24][np.argmax([PCC_cont, RR23])]
                            #    RR25 = [PCC_cell, RR25][np.argmax([PCC_cont, RR23])]
                            #    RR23 = max(PCC_cont, RR23)
                                
        
        
        
                            
                                        
                                
                        except:
                            print('not tracked')
                        start+=1
                    firstD+=1
            
            if math.inf not in ANVtime and RR13=='NAN':# and TApoBD < ANVtime[0]:
                firstD = TApoBD#min(TApoBD, ANVtime[0]), max(TApoBD, ANVtime[0])
                secondD = firstD+1
                while secondD<video_len and RR13=='NAN':
                    if dead_class=='T':
                        trackerD = Basic_Tracker(os.getcwd(), '', '', '', secondD-firstD+1, '', 'dummy', np.amin([len(s) for s in T_seq[firstD:secondD+1]]), T_seq[firstD:secondD+1])
                    elif dead_class=='E':
                        trackerD = Basic_Tracker(os.getcwd(), '', '', '', secondD-firstD+1, '', 'dummy', np.amin([len(s) for s in E_seq[firstD:secondD+1]]), E_seq[firstD:secondD+1])
                    rest_seq = T_seq if dead_class=='E' else E_seq
                    
                    trackedD = trackerD.Run_Tracking()
                    frameApoBD = trackedD[0] 
                    ApoBD_cell_index = [f[0] for f in frameApoBD if f[1:-1]==ApoBDcell_ori[1:-1]]
                    assert len(ApoBD_cell_index)==1
                    ApoBD_cell_index, start  = ApoBD_cell_index[0], 0#ANVtime[0]-TApoBD 
                    
                    while start<len(trackedD):
                        #print('checking frame no. '+str(start+TApoBD))
                        try:
                            ANV_f = ANVmask[start+TApoBD]#markerANV[start+TApoBD,:,:]>filters.threshold_otsu(markerANV[start+TApoBD,:,:])
                            cell_f = trackedD[start][ApoBD_cell_index]
                            IOU_f = getIOU(seq2mask(cell_f), np.logical_and(seq2mask(cell_f), ANV_f))
                            Th1 = filters.threshold_otsu(markerANV[start+TApoBD,:,:])
                            Th2 = filters.threshold_otsu(markerANV[start+TApoBD,:,:][seq2mask(cell_f)])
                            #if Th2-Th1>=OTSUdiff:
                            if IOU_f>0.1 and (start+TApoBD)>=min(ANVtime) and RR13=='NAN' and pPCC(seq2mask(cell_f).astype(np.uint8), markerANV[start+TApoBD])>0.1:
                                print('ApoBD cell also dead!!!!!!')
                                RR13=start+TApoBD
                                break
                        except:
                            print('not tracked')
                        start+=1    
                    secondD+=1
            
            #measure PCC for classifying AnnV distribution mode
            cellPCC, synapsePCC, cellIOU, synapseIOU = 0,0,0,0
            if RR13!='NAN':
                T1,T2 = min(TApoBD, RR13)-2, max(TApoBD, RR13)+2
                print(T1,T2)
                if dead_class=='T':
                    trackerD = Basic_Tracker(os.getcwd(), '', '', '', T2-T1+1, '', 'dummy', np.amin([len(s) for s in T_seq[T1:T2+1]]), T_seq[T1:T2+1])
                    otherseq = E_seq
                elif dead_class=='E':
                    trackerD = Basic_Tracker(os.getcwd(), '', '', '', T2-T1+1, '', 'dummy', np.amin([len(s) for s in E_seq[T1:T2+1]]), E_seq[T1:T2+1])
                    otherseq = T_seq
                trackedD = trackerD.Run_Tracking()
                start, step = 0,1
                #frameApoBD = trackedD[TApoBD-T1] 
                #if RR13>=TApoBD:
                #    frameApoBD = trackedD[0]
                #    ApoBD_cell_index = [f[0] for f in frameApoBD if f[1:-1]==ApoBDcell_ori[1:-1]]
                #    start=0
                #    step=1
    
                #else:
                #    frameApoBD = trackedD[-1]
                #    ApoBD_cell_index = [f[0] for f in frameApoBD if f[1:-1]==ApoBDcell_ori[1:-1]]
                #    start=0
                #    step=1
                #print('index found: ',ApoBD_cell_index)
                
                
                for ind in range(len(trackedD)):
    #                    #print(ind+T1)
                    try:
                        mask_f = maskseq[T1+int(ind*(step))]
                        #cell_f = trackedD[start+int(ind*(step))][ApoBD_cell_index[0]]
                        ANV_f =  markerANV[T1+int(ind*(step)),:,:]
                        pccs = [pPCC(seq2mask(cell).astype(np.uint8), ANV_f) for cell in mask_f]
                        cell_f = mask_f[np.argmax(pccs)]
                        
                        ANV_mask = ANVmask[T1+int(ind*(step))]
                        #other = [rg for rg in otherseq[RR13+int(ind*(step))]]
                        Cmask = synapse_mask[T1+int(ind*(step)),:,:]#np.zeros(ANV_f.shape).astype(np.bool)
                        #for region in other:
                        #    Cmask = np.logical_or(Cmask, np.logical_and(seq2mask(cell_f), seq2mask(region)))
                        #if np.sum(Cmask)>0 and np.sum(B[:RR13+int(ind*(step)),:,:])>0:#pPCC(Cmask.astype(np.uint8), ANV_f)!=math.nan:
                        if np.sum(ANV_mask)>0:
                        #if np.sum(Cmask)>0 and np.sum(synapse_mask[:T1+int(ind*(step)),:,:])>0 and np.sum(ANV_f)>0:#pPCC(Cmask.astype(np.uint8), ANV_f)!=math.nan:
                            #img = np.concatenate([testinput[ind+T1,:,:], ANV_f.astype(np.uint8)*255, seq2mask(cell_f).astype(np.uint8)*255, Cmask.astype(np.uint8)*255], axis=1)
                            img = np.concatenate([testinput[ind+T1,:,:], ANV_f, seq2mask(cell_f).astype(np.uint8)*255, Cmask.astype(np.uint8)*255, ANVmask[T1+int(ind*(step))].astype(np.uint8)*255, allcell_mask[T1+int(ind*(step)),:,:].astype(np.uint8)*255], axis=1)
                            
                            #img = np.stack([testinput[ind+T1,:,:,0], seq2mask(cell_f).astype(np.uint8)*255, ANVmask[T1+int(ind*(step))].astype(np.uint8)*255])

                            #print(pPCC(seq2mask(cell_f).astype(np.uint8), ANV_f), pPCC(Cmask.astype(np.uint8), ANV_f), )
                            synapsePCC = max(synapsePCC, pPCC(Cmask.astype(np.uint8), ANV_f))
                            cellPCC = max(cellPCC, pPCC(seq2mask(cell_f).astype(np.uint8), ANV_f))
                            cellIOU = max(cellIOU, getIOU(ANV_mask, seq2mask(cell_f)))
                            synapseIOU = max(synapseIOU, getIOU(ANV_mask, Cmask))
                            #io.imsave(n+'_frame'+str(T1+ind)+'_'+str(round(pPCC(seq2mask(cell_f).astype(np.uint8), ANV_f),2))+'_'+str(round(pPCC(Cmask.astype(np.uint8), ANV_f),2))+'_'+'.png', img)
                            #print(pPCC(seq2mask(cell_f).astype(np.uint8), ANV_f), pPCC(Cmask.astype(np.uint8), ANV_f))
    #                    #mask_stack_s[start,:,:] = screen_region(B[start,:,:], seq2mask(cell_f))
                    except:
                        print('something missing in index:', str(T1+int(ind*(step))))
                
                           
            
                
                            #contact_f = np.logical_and(B[start+TApoBD,:,:], seq2mask(cell_f))
                            #cell_rest = [cell for cell in trackedD[start] if cell!=cell_f]+[cell for cell in rest_seq[start+TApoBD]]
                            #test_ANV = markerANV[start+TApoBD,:,:]
                            #for cr in cell_rest:
                            #    test_ANV = np.where(seq2mask(cr)==True, np.amin(test_ANV), test_ANV)
                            
                            #IOU_c = getIOU(contact_f, np.logical_and(contact_f, ANV_f))
                            #PCC_cont = pPCC(contact_f.astype(np.uint8), test_ANV)#ANV_f.astype(np.uint8)))
                            #PCC_cell = pPCC(seq2mask(cell_f).astype(np.uint8), test_ANV)
                            #RR15 = max(RR15, IOU_f)
                            #AnVpostApoBD = max(AnVpostApoBD, np.sum(np.logical_and(ANV_f, np.logical_not(seq2mask(cell_f)))))
                            #AnV_IOU_rest = max(AnV_IOU_rest, max([getIOU(seq2mask(cell), np.logical_and(seq2mask(cell), ANV_f)) for cell in cell_rest]))
                            #print('pcc check', PCC_cont, PCC_cell)
                            
                            #if np.amax(contact_f)==True:
                            #    RR20 = [start+TApoBD, RR20][np.argmax([IOU_c, RR19])]
                            #    RR19 = max(IOU_c, RR19)
                            #    RR22 = [start+TApoBD, RR22][np.argmax([PCC_cont-PCC_cell, RR21])]
                            #    RR21 = max(PCC_cont-PCC_cell,RR21)
                            #    RR24 = [start, RR24][np.argmax([PCC_cont, RR23])]
                            #    RR25 = [PCC_cell, RR25][np.argmax([PCC_cont, RR23])]
                            #    RR23 = max(PCC_cont, RR23)
                                
                          
                            #0722 use PCC to find AnnV time
                            #Pccvalue = pPCC(seq2mask(cell_f).astype(np.uint8), test_ANV)#ANV_f.astype(np.uint8)))
                            #print(pPCC(seq2mask(cell_f).astype(np.uint8), ANV_f.astype(np.uint8)))
                            #if Pccvalue>0.5 and (start+TApoBD)>=min(ANVtime):
                            #print(cell_f)
                        
                    #AnVpostApoBD = np.sum(np.logical_and(ANV_f, np.logical_not)
                    
    ##STEP4: tracking for contact and event type
         
            print('ApoBD is found in '+ dead_class+' cell')
            screen_frames=False
            
            Tcombine = min(TApoBD, RR13) if RR13!='NAN' else TApoBD 
            for T_ind, Tcont in enumerate(Tconts):
                if Tcont<Tcombine and found==False:#TApoBD and found==False:
                    
                    #tracker = Basic_Tracker(os.getcwd(), '', '', '', TApoBD-Tcont+1, '', 'dummy', np.amin(CN[Tcont:TApoBD+1]), maskseq[Tcont:TApoBD+1])
                    if dead_class=='T':
                        if screen_frames==True:
                            this_Tseq = [seq for seq in T_seq[Tcont:TApoBD+1] if len(seq)==len(T_seq[Tcont:TApoBD+1][-1])]
                            this_index = [k for k in range(len(T_seq[Tcont:TApoBD+1])) if len(T_seq[Tcont:TApoBD+1][k])==len(T_seq[Tcont:TApoBD+1][-1])]
                            tracker = Basic_Tracker(os.getcwd(), '', '', '', len(this_Tseq), '', 'dummy', np.amin([len(s) for s in this_Tseq]), this_Tseq[::-1])
                        else:
                            this_index = [k for k in range(len(T_seq[Tcont:TApoBD+1]))]
                            tracker = Basic_Tracker(os.getcwd(), '', '', '', TApoBD-Tcont+1, '', 'dummy', np.amin([len(s) for s in T_seq[Tcont:TApoBD+1]]), T_seq[Tcont:TApoBD+1][::-1])
                        seq_no2 = E_seq
                    elif dead_class=='E':
                        if screen_frames==True:
                            this_Eseq = [seq for seq in E_seq[Tcont:TApoBD+1] if len(seq)==len(E_seq[Tcont:TApoBD+1][-1])]
                            this_index = [k for k in range(len(E_seq[Tcont:TApoBD+1])) if len(E_seq[Tcont:TApoBD+1][k])==len(E_seq[Tcont:TApoBD+1][-1])]
                            tracker = Basic_Tracker(os.getcwd(), '', '', '', len(this_Eseq), '', 'dummy', np.amin([len(s) for s in this_Eseq]), this_Eseq[::-1])
                        else:
                            this_index = [k for k in range(len(E_seq[Tcont:TApoBD+1]))]
                            tracker = Basic_Tracker(os.getcwd(), '', '', '', TApoBD-Tcont+1, '', 'dummy', np.amin([len(s) for s in E_seq[Tcont:TApoBD+1]]), E_seq[Tcont:TApoBD+1][::-1])
                        seq_no2 = T_seq
                    tracked = tracker.Run_Tracking()
                    tracked = tracked[::-1]
                    checkindex = Tcont
        
                    ApoBDcell = [f for f in tracked[-1] if f[1:-1]==ApoBDcell_ori[1:-1]]
                    assert len(ApoBDcell)==1
                    ApoBDcell = ApoBDcell[0]
                    while checkindex<Tcombine and found==False:
                    #checkindex>=Tcont:#np.amax(B[cont_index,:,:])!=True:
                        print(checkindex)
                        if np.amax(B[checkindex,:,:])==True and checkindex-Tcont in this_index:
                            #region in this frame
                            contregion = [f for f in tracked[checkindex-Tcont] if f[0] == ApoBDcell[0]]
                            #check if they have contact with the mask
                            check = [contact_check(region, B[checkindex]) for region in contregion]
                            #print(check)
                            #other regions that is not the ApoBD cell
                            other_region = [f for f in tracked[checkindex-Tcont] if f[0] != ApoBDcell[0]]
                            
                            try:
                                other_region = [f for f in other_region if contact_check2(f, contregion[0])==True]
                            except:
                                other_region = []
                            allregions = contregion+other_region
                            
                            #condition: there is contact, and there is the other type of cell having contact
                            #0802edit to screen out dead effectors
                            Th1 = filters.threshold_otsu(markerANV[checkindex,:,:])
                            opposite_seq = seq_no2[checkindex]
                            opposite_seq = [r for r in opposite_seq if getIOU(ANVmask[checkindex], seq2mask(r))<0.5]
                            #print(opposite_seq)
                            #print(contregion[0])
                            #B_label, B_mask = measure.label(B[checkindex,:,:]), np.zeros(B[checkindex,:,:].shape).astype(np.bool)
                            #for k in list(np.unique(B_label)):
                            #    if k!=0 and np.amax(np.logical_and(B_label==k, seq2mask(contregion[0])))==True:
                            #        B_mask = np.logical_or(B_mask, B_label==k)


                            if True in check and True in [np.amax(np.logical_and(seq2mask(contregion[0]), seq2mask(RR))) for RR in opposite_seq]:#seq_no2[checkindex]]:
                            #if True in check and True in [np.amax(np.logical_and(B_mask, seq2mask(RR))) for RR in opposite_seq]:
                            # and list(set([celltype_check_2(r, markerT, markerE, checkindex) for r in allregions]))==[0,1]:
                                
                                #print(contregion, checkindex)
                                io.imsave(os.path.join(kill_dir, n),A)
                                io.imsave(os.path.join(kill_dir, n.replace('.tif','_raw.tif')), testinput)
                                io.imsave(os.path.join(kill_dir, n.replace('.tif','_Tmarker.tif')), (markerT/256).astype(np.uint8))
                                io.imsave(os.path.join(kill_dir, n.replace('.tif','_Emarker.tif')), (markerE/256).astype(np.uint8))
                                
                                #io.imsave(os.path.join(kill_dir_r, n.replace('.tif','_Dmarker.tif')), (markerANV/256).astype(np.uint8))
                                io.imsave(os.path.join(kill_dir, n.replace('.tif','_Dmarker.tif')), markerANV_out)
                                
                                row1.append(n)
                                row2.append(checkindex)
                                row3.append(TApoBD)
    
                                row6.append(ANVtime[0])
                                row12.append('kill')
                                found = True
                                #print('this is killing')
                                
                            #else:
                                #io.imsave(os.path.join(death_dir, n),A)
                            #if checkindex==Tcont and True not in check and T_ind+1==len(Tconts):
                            if checkindex==TApoBD and True not in check and T_ind+1==len(Tconts): 
                                io.imsave(os.path.join(death_dir, n),A)
                                io.imsave(os.path.join(death_dir, n.replace('.tif','_raw.tif')), testinput)
                                io.imsave(os.path.join(death_dir, n.replace('.tif','_Tmarker.tif')), (markerT/256).astype(np.uint8))
                                io.imsave(os.path.join(death_dir, n.replace('.tif','_Emarker.tif')), (markerE/256).astype(np.uint8))
                                io.imsave(os.path.join(death_dir, n.replace('.tif','_Dmarker.tif')), markerANV_out)
                                
                                row12.append('death')
                                row2.append(video_len)
                                found = True
                                
                                
                                
                        else:
                            #if checkindex==Tcont and T_ind+1==len(Tconts):
                            if checkindex==TApoBD and T_ind+1==len(Tconts):
                                io.imsave(os.path.join(death_dir, n),A)
                                io.imsave(os.path.join(death_dir, n.replace('.tif','_raw.tif')), testinput)
                                io.imsave(os.path.join(death_dir, n.replace('.tif','_Tmarker.tif')), (markerT/256).astype(np.uint8))
                                io.imsave(os.path.join(death_dir, n.replace('.tif','_Emarker.tif')), (markerE/256).astype(np.uint8))
                                io.imsave(os.path.join(death_dir, n.replace('.tif','_Dmarker.tif')), markerANV_out)
                                row12.append('death')
                                row2.append(video_len)
                                found = True
                        checkindex+=1
    
            if found==False:
                io.imsave(os.path.join(non_dir, n),A)
                io.imsave(os.path.join(non_dir, n.replace('.tif','_raw.tif')), testinput)
                io.imsave(os.path.join(non_dir, n.replace('.tif','_Tmarker.tif')), (markerT/256).astype(np.uint8))
                io.imsave(os.path.join(non_dir, n.replace('.tif','_Emarker.tif')), (markerE/256).astype(np.uint8))
                io.imsave(os.path.join(non_dir, n.replace('.tif','_Dmarker.tif')), markerANV_out)
                row12.append('non')
                row2.append(video_len)
            #if abs(TApoBD-ANVtime[0])>0 and abs(TApoBD-ANVtime[0])<math.inf and dead_class==dead_class_AnnV:#and Enum2==0 and Tnum2==1:
            #    combined = np.concatenate([testinput, markerANV],axis=-1)
            #    if TApoBD<ANVtime[0]:
            #        io.imsave(os.path.join(ApoBD_first_dir, n.replace('.tif','_P+D.tif')), combined)
            #    else:
            #        io.imsave(os.path.join(AnnV_first_dir, n.replace('.tif','_P+D.tif')), combined)    
            
            
            
            
            row4.append(Tnum2)
            row5.append(Enum2)
            row7.append(n)
            row8.append(TApoBD)
            if math.inf not in ANVtime:
                row10.append(dead_class_AnnV)
                row9.append(ANVtime[0])
            else:
                row9.append(math.inf)
                row10.append('NAN')
            #if math.inf not in ANVtime and TApoBD > ANVtime[0]:
            ApoBD_NUM = [np.amax(measure.label(D1[k,:,:])) for k in range(len(D1))]
            row11.append(dead_class)
            row13.append(RR13)
            row14.append(max(ApoBD_NUM))
            row18.append(synapsePCC)
            row17.append(cellPCC)
            row26.append(synapseIOU)
            row27.append(cellIOU)
        #row15.append(RR15)
        #APOBDCELLarea = (ApoBDcell_ori[3]-ApoBDcell_ori[1])*(ApoBDcell_ori[4]-ApoBDcell_ori[2])
        #row16.append(APOBDCELLarea)
        #row17.append(AnVpostApoBD)
        #row18.append(AnV_IOU_rest)
        #row19.append(RR19)
        #row20.append(RR20)
        #row21.append(RR21)
        #row22.append(RR22)
        #row23.append(RR23)
        #row24.append(RR24)
        #row25.append(RR25)
                
        except:
            print(n+' has error')
        ####use tracking to decide order of ApoBD and AnnV death####  
    #(pd.DataFrame([row1,row4,row5,row2,row3,row6])).T.to_excel(os.path.join(kill_dir, 'table.xlsx'))
    (pd.DataFrame([row7,row8,row11, row9, row10,row12,row13,row14,row15,row16,row17,row18,row4,row5,row2,row26,row27])).T.to_excel(os.path.join(outputdir, 'table.xlsx'))
    #,row19,row20,row21,row22,row23,row24,row25
'''
        if abs(TApoBD-ANVtime[0])>0 and abs(TApoBD-ANVtime[0])<math.inf and found==False:#and Enum2==0 and Tnum2==1:
            combined = np.concatenate([testinput, markerANV],axis=-1)
            if TApoBD<ANVtime[0]:
                io.imsave(os.path.join(ApoBD_first_dir, n.replace('.tif','_P+D.tif')), combined)
            else:
                io.imsave(os.path.join(AnnV_first_dir, n.replace('.tif','_P+D.tif')), combined)
        if celltype_check_3(ApoBDcell, markerT, markerE, TApoBD)==0:
            io.imsave(os.path.join(T_death_dir, n),A)
        else:
            io.imsave(os.path.join(E_death_dir, n),A)
        
        
            
            
            
            firstD, secondD = min(TApoBD, ANVtime[0]), max(TApoBD, ANVtime[0])
            print('Now track between T= ', firstD, ' and T= ', secondD)
            trackerD = Basic_Tracker(os.getcwd(), '', '', '', secondD-firstD+1, '', 'dummy', np.amin(CN[firstD:secondD+1]), maskseq[firstD:secondD+1])
            trackedD = trackerD.Run_Tracking()
            frameApoBD, frameAnnV = trackedD[0 if TApoBD==firstD else -1], trackedD[-1 if TApoBD==firstD else 0]
            print('-----')
            print(ApoBDcell)
            print(frameApoBD)
            print(AnnvCell)
            print(frameAnnV)
            print('-----')
            
            tracked_ApoBD, tracked_AnnV = [f for f in frameApoBD if f[1:]==ApoBDcell[1:]][0], [f for f in frameAnnV if f[1:]==AnnvCell[1:]][0]
            
            

            print('AnnV check result is: ', frameApoBD[0]==frameAnnV[0])
            
            #print([f[1:]==ApoBDcell[1:] for f in frameApoBD], [f[1:]==AnnvCell[1:] for f in frameAnnV])
            
            
            #start1 = ANVtime[0]-1
            #visua1_AnV1, visual_AnV2  = [markerANV[k,:,:] for k in range(start1,start1+5)], [testinput[k,:,:] for k in range(start1,start1+5)]
            #visual_AnV1, visual_AnV2 = np.concatenate([a for a in visual_AnV1], axis=-1), np.concatenate([a for a in visual_AnV2], axis=-1)
            #start2 = TApoBD-1
            #visua1_ApoBD1, visual_ApoBD2  = [markerANV[k,:,:] for k in range(start2,start2+5)], [testinput[k,:,:] for k in range(start2,start2+5)]
            #visual_ApoBD1, visual_ApoBD2 = np.concatenate([a for a in visual_ApoBD1], axis=-1), np.concatenate([a for a in visual_ApoBD2], axis=-1)
            
''' 
#    def profile(args):
#        print('still working on it')



if __name__ == "__main__":

    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description="run MaskRCNN inference for profiling")
    parser.add_argument('--home_dir', default='/project/varadarajan/kwu14/proj/getvideo/', type=str)
    parser.add_argument('--DT_home', default='/project/varadarajan/kwu14/DT-HPC/', type=str)
    parser.add_argument('--dataset', default='05130055NR_HBSSv2_CNNclassified_0610', type=str)
    parser.add_argument('--MRCNNweights1', default='/project/varadarajan/kwu14/repo/VisTR-master/MRCNN_weights/0419_compare_VisTR.h5', type=str)
    parser.add_argument('--MRCNNweights2', default='/project/varadarajan/kwu14/Mask_RCNN/logs/ApoBD_all/0419_ApoBD_iter1_0077.h5', type=str)
    parser.add_argument('--endpoint', default=99999, type=int)
    parser.add_argument('--expname', default='debug', type=str)
    parser.add_argument('--vidinput', default='None', type=str)
    parser.add_argument('--generate_all', default=True, action='store_false')
    parser.add_argument('--just_get_table', default=False, action='store_true')
    parser.add_argument('--ApoBDmode', default='vote', type=str)
    parser.add_argument('--ApoBD_Time', default='CNN', type=str)
    
    args = parser.parse_args()
    print('ApoBD mode is:', args.ApoBDmode)
    print('ApoBD time is from:', args.ApoBD_Time)
    pipeline(args)
    #if args.generate_all:
    #    pipeline(args)
    #else:
    #    profiling(args)