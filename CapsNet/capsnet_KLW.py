"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, optimizers, regularizers, losses
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical


import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import *#CapsuleLayer, PrimaryCap, Length, Mask
import os
from ops import update_routing
from math import floor
from models import *

K.set_image_data_format('channels_last')





def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    # return tf.reduce_mean(tf.square(y_pred))
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    #return tf.reduce_mean(tf.reduce_sum(L, 1))
    
    if args.L2==0:
        return tf.reduce_mean(tf.reduce_sum(L, 1))
    elif args.L2==1:
        loss = tf.reduce_mean(tf.reduce_sum(L, 1))
        with tf.name_scope('l2_loss'):
          l2_loss = tf.reduce_sum(0.5 * tf.stack([tf.nn.l2_loss(v) for v in tf.compat.v1.get_collection('weights')]))
          loss += l2_loss
        return(loss)
'''
def CE(y_true, y_pred):
    try:
        diff = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    except:
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = tf.reduce_mean(diff)
    return loss
'''



def train(model,  # type: models.Model
          data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/'+args.logname+'.csv')
    if args.model == 'alex':
        #checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_loss',
        #                                       save_best_only=True, save_weights_only=True, verbose=1)
        checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/'+args.modelname+'.h5', monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)
    else:
        #checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc', save_best_only=True, save_weights_only=True, verbose=1)
        checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/'+args.modelname+'.h5', monitor='val_capsnet_accuracy', save_best_only=True, save_weights_only=False, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    if args.model == 'original' or args.model =='deep':
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[margin_loss, 'mse'],
                      loss_weights=[1., args.lam_recon],
                      metrics={'capsnet': 'accuracy'})
    elif args.model =='alex':
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss='categorical_crossentropy',
                      loss_weights=1,
                      metrics={'alexnet': 'accuracy'})
    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, verbose=2, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction,
                                           rotation_range=180)  # shift up to 2 pixel for MNIST
                                           #add rotation on 12302021
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield (x_batch, y_batch), (y_batch, x_batch)

    # Training with data augmentation. If shift_fraction=0., no augmentation.
    
    
              #
    if args.model=='alex':
        model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(x_test, y_test), shuffle = True, use_multiprocessing=True, verbose=2, callbacks=[log, checkpoint, lr_decay])
    
    else:
        model.fit(train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                  steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                  epochs=args.epochs, verbose=2,
                  validation_data=((x_test, y_test), (y_test, x_test)), batch_size=args.batch_size,
                  callbacks=[log, checkpoint, lr_decay])
    
    # End: Training with data augmentation -----------------------------------------------------------------------#

    #model.save_weights(args.save_dir + '/trained_model.h5')
    #model.save(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    #from utils import plot_log
    #plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()

def test_mk2(model, data, args):
#generate probability prediction
    x_test, y_test = data
    input_shape, batch_size=x_train.shape[1:], args.batch_size
    I, O = model.layers[0].input, model.layers[-2].output
    model2 = models.Model(I,O)
    out = model2.predict(x_test)
    route1, route2 = os.path.join(os.getcwd(), 'prob_test.xlsx'), os.path.join(os.getcwd(), 'argmax_test.xlsx')
    
def filtering(model, data, args):
    model.summary()
    I, O = model.layers[0].input, model.layers[2].output
    model2 = models.Model(I,O)
    model2.summary()
    data = np.expand_dims(data,-1)
    out = np.zeros((data.shape[0], data.shape[1], 1))
    A = data[0,:args.batch_size,:,:,:]
    print(A.shape)
    print(model2.predict(A).shape)
    
    '''
    for k in range(len(data)):
        for kk in range(floor(len(data[k,:,:,:,:])/args.batch_size)):
            #print(np.argmax(model2.predict(data[k, kk*args.batch_size:(kk+1)*args.batch_size, :,:,:]), axis=-1))
            out[k, kk*args.batch_size:(kk+1)*args.batch_size, :] = np.expand_dims(np.argmax(model2.predict(data[k, kk*args.batch_size:(kk+1)*args.batch_size, :,:,:]), axis=-1), -1)
        #print((model2.predict(data[k,-10:,:,:,:])))
        #print((model2.predict(data[k,-10:,:,:,:])).shape)
        out[k,-3:,:] = np.expand_dims(np.argmax(model2.predict(data[k,-10:,:,:,:]), axis=-1)[-3:], -1)
    #from pandas import DataFrame
    out = out.reshape(out.shape[0], out.shape[1])
    #DataFrame(out).to_excel('/project/varadarajan/kwu14/repo/CapsNet-Keras-tf2/test0121.xlsx')
    np.save('/project/varadarajan/kwu14/repo/CapsNet-Keras-tf2/test.npy', out)
    '''        
    
    
def manipulate_latent(model, data, args):
    print('-' * 30 + 'Begin: manipulate' + '-' * 30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def to_float(inputarr):
    return((inputarr/255).astype(np.float64))

if __name__ == "__main__":
    import os
    import argparse
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import callbacks
    from tensorflow.keras.utils import to_categorical

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help="Initial LR")
    parser.add_argument('--lr_decay', default=0.999, type=float, help="LR decay ratio")
    parser.add_argument('--lam_recon', default=0.392, type=float, help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int, help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float, help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true', help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-m', '--mode', default='train')
    #parser.add_argument('-t', '--testing', default=False, action='store_true', help="Test the trained model on testing dataset")
    #parser.add_argument('-tm2', '--testingmk2', default=False, action='store_true', help="Test the trained model on testing dataset, revised")
    #parser.add_argument('-filter', '--filtering', default=False, action='store_true', help="Test the trained model on testing dataset, revised")                        
    parser.add_argument('--digit', default=0, type=int, help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None, help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--dt', type=str, default='apoptosis')
    parser.add_argument('--size', type=int, default=28)
    parser.add_argument('--model', default='original')
    parser.add_argument('--dmode', type=int, default=1)
    parser.add_argument('--L2', type=int, default=0)
    parser.add_argument('--logname', type=str, default='apoptosis')
    parser.add_argument('--modelname', type=str, default='apop_detector')
    
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    if args.dt =='minst':
        (x_train, y_train), (x_test, y_test) = load_mnist()
    elif args.dt == 'apoptosis':
        import h5py
        h5f = h5py.File(os.path.join(os.getcwd(), 'dataset/data_'+str(args.size)+'.h5'), 'r')
        x_train, y_train, x_test, y_test = h5f['X_train'][:], h5f['Y_train'][:], h5f['X_test'][:], h5f['Y_test'][:]
        if args.size==51:
            x_train, x_test = to_float(x_train), to_float(x_test)
        #print(x_train.shape, y_train.shape)
    
    elif args.dt == 'CART':
        from sklearn.model_selection import train_test_split
        Data = np.load(os.path.join(os.getcwd(), 'dataset/Data.npy'))[:1440,:,:,0]
        A,B,C = Data.shape
        Data, Label = to_float(Data.reshape((A,B,C,1))), to_categorical(np.load(os.path.join(os.getcwd(), 'dataset/Label.npy'))[:1440])
        x_train, x_test, y_train, y_test = train_test_split(Data, Label, test_size=0.2)
        #print(x_train.shape, y_train.shape)
    
    elif args.dt == 'CARTall':
        from sklearn.model_selection import train_test_split
        Data = np.load('/project/varadarajan/kwu/dataset/0719CARALL_Data.npy')[:47520,:,:,0]
        A,B,C = Data.shape
        Data, Label = to_float(Data.reshape((A,B,C,1))), to_categorical(np.load('/project/varadarajan/kwu/dataset/0719CARALL_Label.npy')[:47520])
        x_train, x_test, y_train, y_test = train_test_split(Data, Label, test_size=0.2)
        print(x_train.shape, y_train.shape)
    
    elif args.dt == 'TfromSEQ':
        from sklearn.model_selection import train_test_split
        alive, dead = np.load(os.path.join(os.getcwd(), 'T_alive.npy')), np.load(os.path.join(os.getcwd(), 'T_dead.npy'))
        labelA, labelD = np.zeros((len(alive,))), np.ones((len(dead,)))
        Data, Label = np.concatenate([alive, dead], axis=0), np.concatenate([labelA, labelD], axis=0)
        Data, Label = np.expand_dims(Data, -1), to_categorical(Label)
        Data, Label = to_float(Data[:10400,:,:,:]), Label[:10400,:]
        x_train, x_test, y_train, y_test = train_test_split(Data, Label, test_size=0.3)
        print(x_train.shape, y_train.shape)
    
    elif args.dt == 'EfromSEQ':
        from sklearn.model_selection import train_test_split
        alive, dead = np.load(os.path.join(os.getcwd(), 'E_alive.npy')), np.load(os.path.join(os.getcwd(), 'E_dead.npy'))
        labelA, labelD = np.zeros((len(alive,))), np.ones((len(dead,)))
        Data, Label = np.concatenate([alive, dead], axis=0), np.concatenate([labelA, labelD], axis=0)
        Data, Label = to_float(np.expand_dims(Data, -1)), to_categorical(Label)
        Data, Label = Data[:,:,:,:], Label[:,:]
        x_train, x_test, y_train, y_test = train_test_split(Data, Label, test_size=0.3)
        print(x_train.shape, y_train.shape)

    # define model
    if args.model== 'original':
        model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                      n_class=len(np.unique(np.argmax(y_train, 1))),
                                                      routings=args.routings,
                                                      batch_size=args.batch_size)
    elif args.model=='deep':
        model, eval_model, manipulate_model = DeepCapsNet(input_shape=x_train.shape[1:],
                                                      n_class=len(np.unique(np.argmax(y_train, 1))),
                                                      routings=args.routings,
                                                      batch_size=args.batch_size)
    elif args.model == 'alex':
        model = AlexNet(input_shape=x_train.shape[1:], n_class=len(np.unique(np.argmax(y_train, 1))), batch_size=args.batch_size)
        
    
    #model.summary()
    #for layer in model.layers:
     #   print(layer.output_shape)
    # train or test
    
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.mode=='train':
        model.summary()
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    elif args.mode == 'testing':
        test(model=eval_model, data=(x_test, y_test), args=args)
    elif args.mode == 'testingmk2':
        test_mk2(model=model, data=(x_test, y_test), args=args)
    elif args.mode== 'filtering':
        filtering(model=model, data=x_test, args=args)
        
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)
    
