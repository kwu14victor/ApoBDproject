import os, sys, argparse
import numpy as np
sys.path.append('MRCNN/')
from skimage import io, exposure, filters, color
from math import pi as PI
from datetime import datetime
from itertools import groupby
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn.model as modellib




class CellsConfig(Config):
    def __init__(self, dataset):
        """Set values of computed attributes."""
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

        self.dataset = dataset
    """Configuration for training on the nucleus segmentation dataset."""
    NAME = "Cells"
    IMAGES_PER_GPU = 3
    NUM_CLASSES = 1 + 3
    VAL_IMAGE_IDS = []
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)
    DETECTION_MIN_CONFIDENCE = 0.5
    BACKBONE = "resnet50"
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000
    RPN_NMS_THRESHOLD = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)
    TRAIN_ROIS_PER_IMAGE = 256
    MAX_GT_INSTANCES = 200
    DETECTION_MAX_INSTANCES = 400


class CellsInferenceConfig(CellsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.6    
    
def pipeline(args):
    pfiletype, dshome, frames = args.dtype, args.dhome, args.frames   
    config = CellsInferenceConfig(args.dhome)
    config.DETECTION_MIN_CONFIDENCE, config.DETECTION_NMS_THRESHOLD = 0.5, 0.7
    model = modellib.MaskRCNN(mode="inference",config=config,model_dir='MRCNN/')
    model.load_weights(os.path.join(args.weights_root, args.weight_name), by_name=True)
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
        
        if not os.path.exists(moviehome):
            os.mkdir(moviehome)
        if not os.path.exists(dead_dir):
            os.mkdir(dead_dir)
        imgdir = os.path.join(args.dhome, dsname)
        names = [f for f in os.listdir(imgdir) if 'marker' not in f]
        imgs = [io.imread(os.path.join(imgdir,f)) for f in names]
        c1 = [io.imread(os.path.join(imgdir,f.replace('.tif','_Tmarker.tif'))) for f in names]
        c2 = [io.imread(os.path.join(imgdir,f.replace('.tif','_Emarker.tif'))) for f in names]
        c3 = [io.imread(os.path.join(imgdir,f.replace('.tif','_Dmarker.tif'))) for f in names]
        for index, img in enumerate(imgs):
            generate = False
            areas, nums, cell_nums, mask = MRCNN_ApoBDs(model, list(img))
            if np.amax(nums)>=3:
                generate = True
            
            if generate:
                ABs = np.array(nums)
                ABs = ABs>filters.threshold_otsu(np.array(nums))
                DorA, noF, dead, D_start = [k for k,g in groupby(ABs)], [len(list(g)) for k,g in groupby(ABs)], False, args.frames
                for ind, cla in enumerate(DorA):
                    if cla==True:
                        if noF[ind]>=3: 
                            D_start = sum(noF[:ind])
                            try:
                                D_end = sum(noF[:ind+1])
                            except:
                                D_end = sum(noF[:-1])
                            dead = True
                            break
                if dead:
                    io.imsave(os.path.join(dead_dir,  names[index].replace('.tif','')+'_die_at_frame' + str(D_start) + '_.tif'), img)
                    io.imsave(os.path.join(dead_dir,  names[index].replace('.tif','')+'_die_at_frame' + str(D_start) + '_T.tif'), c1[index])
                    io.imsave(os.path.join(dead_dir,  names[index].replace('.tif','')+'_die_at_frame' + str(D_start) + '_E.tif'), c2[index])
                    io.imsave(os.path.join(dead_dir,  names[index].replace('.tif','')+'_die_at_frame' + str(D_start) + '_D.tif'), c3[index])
                    img = color.gray2rgb(img)
                    img[:,:,:,0] = np.where(mask==True, 255, img[:,:,:,0])
                    img[:,:,:,2] = np.where(mask==True, 255, img[:,:,:,2])
                    io.imsave(os.path.join(dead_dir, names[index].replace('.tif','') + '_die_at_frame' + str(D_start) + '_masked_.tif'), img)
            

 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="screen out image stack with ApoBDs")
    parser.add_argument('--dtype', default='.tif', type=str)
    parser.add_argument('--dhome', default='sample_data/', type=str)
    parser.add_argument('--frames', default=73, type=int)
    parser.add_argument('--weights_root', default='weights', type=str)
    parser.add_argument('--weight_name', default='step0_weights.h5', type=str)
    parser.add_argument('--expname', default='debug_1', type=str)
    parser.add_argument('--ds', '--names-list', nargs='+', default=[])
    parser.add_argument('--endpoint', default=-1, type=int)
    args = parser.parse_args()
    pipeline(args)    
    