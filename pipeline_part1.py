import os, sys, random, argparse, shutil, math, cv2
import numpy as np
from scipy import ndimage, misc
import pandas as pd
from skimage import io, exposure, transform, filters, morphology, measure, color
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from math import pi as PI
from datetime import datetime
from itertools import groupby
from PIL import Image
from torchvision import transforms

sys.path.append('/project/varadarajan/kwu14/proj/getvideo/CNN/')
sys.path.append('/project/varadarajan/kwu14/proj/getvideo/')
sys.path.append('/project/varadarajan/kwu14/repo/practice/')
sys.path.append('/project/varadarajan/kwu14/repo/PyTorch-CNN-Visualizations-Saliency-Maps-master/src/')
from vis_grad import *
from gradcam import GradCam
from mymodels import *
from nanowell import *
from pipeline_ops_torch import *

def getmodel(args):
    if args.model_name == 'ResNet50':
        model = MyResnet50(args)
        model.load_state_dict(torch.load(os.path.join(args.weights_root, args.weights_name)))
        model.eval()
        return(model)
def event_start(seq, condition, howlong):
    out, time = False, math.inf
    labels, length = [k for k,g in groupby(seq)], [len(list(g)) for k,g in groupby(seq)]
    for ind, cla in enumerate(labels):
        if cla==condition and ind!=0:
            if length[ind]>=howlong :
                out, time = True, sum(length[:ind])
                break
    return(out, time)


def generate_GCmask(args, imgs, model, classno=0):
    if args.model_name=='alex':
        cam_index = 10
    elif args.model_name=='resnet':
        cam_index = 46
    elif args.model_name=='ResNet50':
        cam_index = 120
    output = []
    for img in imgs:
        A, _, _ = vis_gradcam(model, classno, cam_index, img, size = [args.size, args.size])
        output.append(A)
    return(np.array(output).astype(np.uint8))

def pipeline(args):
    model = getmodel(args)
    model.cuda()
    TFM = transforms.Compose([transforms.Resize((args.size,args.size)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    #now = (datetime.now()).strftime('%m')+(datetime.now()).strftime('%d')+(datetime.now()).strftime('%H')+(datetime.now()).strftime('%M')
    img_root, outdir = os.path.join(args.home_dir, args.img_folder), os.path.join(os.getcwd(), args.expname)
    videos = [f for f in os.listdir(img_root) if 'masked_.tif' not in f] 
    rawimg_dir, focusmask_dir, dead_dir = os.path.join(outdir, 'rawimg'), os.path.join(outdir, 'focus_map'), os.path.join(outdir, 'dead')
    for dr in [outdir, rawimg_dir, focusmask_dir, dead_dir]:
        try:
            os.mkdir(dr)
        except:
            print('directory created')
    
    for vid in videos[:args.endpoint]:
        img = io.imread(os.path.join(img_root, vid))
        ABs, dead_label, dead_time, imgdirs = [], [], [], []
        #for ind, model in enumerate(models):
        for k in range(len(img)):
            rs_img = exposure.rescale_intensity(img[k,:,:], out_range=np.uint8)
            rawdir = os.path.join(rawimg_dir, vid.replace('.tif','_frame'+str(k+1)+'.tif'))
            io.imsave(rawdir, rs_img)
            frame = (Image.fromarray(exposure.rescale_intensity(rs_img, out_range=np.uint8))).convert('RGB')
            #frame = frame.convert('RGB')
            ABs.append(np.argmax(model((TFM(frame).unsqueeze_(0).cuda()).float()).detach().cpu().numpy()))
            imgdirs.append(rawdir)
        focus = generate_GCmask(args, imgdirs, model)
        dead, TApoBD = event_start(ABs, 0, 3)
        print(dead, TApoBD)
        
        
        if dead:
            io.imsave(os.path.join(focusmask_dir, vid.replace('.tif','_CNNsays_'+str(TApoBD)+'_GradCam.tif')), np.array(focus).astype(np.uint8))
            shutil.copyfile(os.path.join(img_root, vid), os.path.join(dead_dir, vid.replace('.tif', '_CNNsays_'+str(TApoBD)+'.tif')))                 
        
    
if __name__ == "__main__":

    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description="get ApoBD time and GradCamMap")
    parser.add_argument('--size', default=281, type=int)
    parser.add_argument('--mode', type=str, default = 'ApoBD_detect')
    
    parser.add_argument('--home_dir', default='/project/varadarajan/kwu14/proj/getvideo/', type=str)
    parser.add_argument('--img_folder', default='05130055NR_HBSSv2/dead/', type=str)
    parser.add_argument('--model_name', default='ResNet50', type=str)
    parser.add_argument('--weights_root', default='/project/varadarajan/kwu14/repo/practice/result/', type=str)
    parser.add_argument('--weights_name', default='0610_Log/0610_ResNet50ApoBDepo=5.pth', type=str)
    parser.add_argument('--frames', default=73, type=int)
    parser.add_argument('--endpoint', default=-1, type=int)
    parser.add_argument('--expname', default='debug', type=str)
    
    args = parser.parse_args()
    pipeline(args)