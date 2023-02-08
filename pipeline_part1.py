import os, sys, argparse, shutil, math
import numpy as np
from skimage import io, exposure
from datetime import datetime
from itertools import groupby
from PIL import Image
from torchvision import transforms
sys.path.append('gradcam/')
from vis_grad import *
from gradcam import GradCam
from PTmodels import *

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
    img_root, outdir = os.path.join(args.home_dir, args.img_folder), os.path.join(os.getcwd(), args.expname)
    
    videos = [f for f in os.listdir(img_root) if 'masked_.tif' not in f and '_.tif' in f] 
    rawimg_dir, focusmask_dir, dead_dir = os.path.join(outdir, 'rawimg'), os.path.join(outdir, 'focus_map'), os.path.join(outdir, 'dead')
    for dr in [outdir, rawimg_dir, focusmask_dir, dead_dir]:
        try:
            os.mkdir(dr)
        except:
            print('directory created')
    print(videos)
    
    for vid in videos:
        
        img = io.imread(os.path.join(img_root, vid))
        ABs, dead_label, dead_time, imgdirs = [], [], [], []
        for k in range(len(img)):
            rs_img = exposure.rescale_intensity(img[k,:,:], out_range=np.uint8)
            rawdir = os.path.join(rawimg_dir, vid.replace('.tif','_frame'+str(k+1)+'.tif'))
            io.imsave(rawdir, rs_img)
            frame = (Image.fromarray(exposure.rescale_intensity(rs_img, out_range=np.uint8))).convert('RGB')
            ABs.append(np.argmax(model((TFM(frame).unsqueeze_(0).cuda()).float()).detach().cpu().numpy()))
            imgdirs.append(rawdir)
        focus = generate_GCmask(args, imgdirs, model)
        dead, TApoBD = event_start(ABs, 0, 3)
        
        if dead:
            vid2 = vid.replace('.tif','')
            print(vid2)
            io.imsave(os.path.join(focusmask_dir, vid2+'_CNNsays_'+str(TApoBD)+'_GradCam.tif'), np.array(focus).astype(np.uint8))
            shutil.copyfile(os.path.join(img_root, vid), os.path.join(dead_dir, vid2+'_CNNsays_'+str(TApoBD)+'.tif')) 
            shutil.copyfile(os.path.join(img_root, vid.replace('.tif','T.tif')), os.path.join(dead_dir, vid2+'_CNNsays_'+str(TApoBD)+'_Tmarker.tif'))
            shutil.copyfile(os.path.join(img_root, vid.replace('.tif','E.tif')), os.path.join(dead_dir, vid2+'_CNNsays_'+str(TApoBD)+'_Emarker.tif'))
            shutil.copyfile(os.path.join(img_root, vid.replace('.tif','D.tif')), os.path.join(dead_dir, vid2+'_CNNsays_'+str(TApoBD)+'_Dmarker.tif'))                
        
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="get ApoBD time and GradCamMap")
    parser.add_argument('--size', default=281, type=int)
    parser.add_argument('--home_dir', default='/', type=str)
    parser.add_argument('--img_folder', default='0205test/dead_raw', type=str)
    parser.add_argument('--model_name', default='ResNet50', type=str)
    parser.add_argument('--weights_root', default='weights', type=str)
    parser.add_argument('--weights_name', default='step1_weights.pth', type=str)
    parser.add_argument('--frames', default=73, type=int)
    parser.add_argument('--endpoint', default=-1, type=int)
    parser.add_argument('--expname', default='debug', type=str)
    
    args = parser.parse_args()
    pipeline(args)