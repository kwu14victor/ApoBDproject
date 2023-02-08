from pandas import DataFrame
from skimage import io, transform, exposure
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import models,layers
from sklearn.decomposition import PCA
import matplotlib, os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

detector = models.load_model('trained_model.h5')
detector.summary()
home = '/project/varadarajan/kwu14/proj/getvideo/CNN/'

for fname in [f for f in os.listdir(home) if '.tif' in f]:
    stack = io.imread(home+'/'+fname)
    
    stack = np.expand_dims(stack,-1)
    result = np.argmax(detector.predict(stack),axis=-1)
    
    out = np.tile(stack,(1,1,1,3))
    for k in range(len(out)):
        if result[k]==1:
            out[k,:5,:5, 0], out[k,:5,:5, 1], out[k,:5,:5, 2]=255,255,0
    io.imsave(home+fname+'labeled.tif',out)
        
    