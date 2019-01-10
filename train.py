import os
import sys
import gc
import numpy as np
from argparse import ArgumentParser

from PIL import Image
import matplotlib.pyplot as plt

# Import backend without the "Using X Backend" message
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras import backend as K
sys.stderr = stderr

from libs.srgan import SRGAN
from libs.util import plot_test_images, DataLoader


# Sample call
"""
# Train 2X SRGAN
python train.py --train C:/Documents/Kaggle/Kaggle-imagenet/input/DET/train --validation C:/Documents/Kaggle/Kaggle-imagenet/input/DET/test --scale 2 --test_path images/samples_2X --stage gan

# Train the 4X SRGAN
python train.py --train C:/Documents/Kaggle/Kaggle-imagenet/input/DET/train --validation C:/Documents/Kaggle/Kaggle-imagenet/input/DET/test --scale 4 --test_path images/samples_4X --scaleFrom 2

# Train the 8X SRGAN
python train.py --train C:/Documents/Kaggle/Kaggle-imagenet/input/DET/train --validation C:/Documents/Kaggle/Kaggle-imagenet/input/DET/test --scale 8 --test_path images/samples_8X --scaleFrom 4
"""

def parse_args():
    parser = ArgumentParser(description='Training script for SRGAN')

    parser.add_argument(
        '-stage', '--stage',
        type=str, default='all',
        help='Which stage of training to run',
        choices=['all', 'mse', 'gan', 'gan-finetune']
    )

    parser.add_argument(
        '-train', '--train',
        type=str,
        help='Folder with training images'
    )
    
    parser.add_argument(
        '-validation', '--validation',
        type=str,
        help='Folder with validation images'
    )

    parser.add_argument(
        '-test', '--test',
        type=str, default='./images/samples_HR',
        help='Folder with testing images'
    )
        
    parser.add_argument(
        '-dataname', '--dataname',
        type=str, default='imagenet',
        help='Dataset name, e.g. \'imagenet\''
    )
        
    parser.add_argument(
        '-scale', '--scale',
        type=int, default=2,
        help='How much should we upscale images'
    )

    parser.add_argument(
        '-scaleFrom', '--scaleFrom',
        type=int, default=None,
        help='Perform transfer learning from lower-upscale model'
    )
        
    parser.add_argument(
        '-workers', '--workers',
        type=int, default=4,
        help='How many workers to user for pre-processing'
    )
        
    parser.add_argument(
        '-batch_size', '--batch_size',
        type=int, default=16,
        help='What batch-size should we use'
    )

    parser.add_argument(
        '-crops_per_image', '--crops_per_image',
        type=int, default=2,
        help='Increase in order to reduce random reads on disk (in case of slower SDDs or HDDs)'
    )
            
    parser.add_argument(
        '-test_path', '--test_path',
        type=str, default='./images/samples_2X/',
        help='Where to output test images during training'
    )
        
    parser.add_argument(
        '-weight_path', '--weight_path',
        type=str, default='./data/weights/',
        help='Where to output weights during training'
    )
        
    parser.add_argument(
        '-log_path', '--log_path',
        type=str, default='./data/logs/',
        help='Where to output tensorboard logs during training'
    )
        
    return  parser.parse_args()

def reset_layer_names(args):
    '''In case of transfer learning, it's important that the names of the weights match
    between the different networks (e.g. 2X and 4X). This function loads the lower-lever
    SR network from a reset keras session (thus forcing names to start from naming index 0),
    loads the weights onto that network, and saves the weights again with proper names'''

    # Find lower-upscaling model results
    BASE_G = os.path.join(args.weight_path, 'SRGAN_'+args.dataname+'_generator_'+str(args.scaleFrom)+'X.h5')
    BASE_D = os.path.join(args.weight_path, 'SRGAN_'+args.dataname+'_discriminator_'+str(args.scaleFrom)+'X.h5')
    assert os.path.isfile(BASE_G), 'Could not find '+BASE_G
    assert os.path.isfile(BASE_D), 'Could not find '+BASE_D
    
    # Load previous model with weights, and re-save weights so that name ordering will match new model
    prev_gan = SRGAN(upscaling_factor=args.scaleFrom)
    prev_gan.load_weights(BASE_G, BASE_D)
    prev_gan.save_weights(args.weight_path+'SRGAN_'+args.dataname)
    del prev_gan
    K.reset_uids()
    gc.collect()
    return BASE_G, BASE_D

def gan_freeze_layers(args, gan):
    '''In case of transfer learning, this function freezes lower-level generator
    layers according to the scaleFrom argument, and recompiles the model so that
    only the top layer is trained in the generator'''

    # Map scalings to layer name
    s2l = {2: '1', 4: '2', 8: '3'}

    # 4X -> 8X block always trainable. 2X -> 4X only if going from 2X.
    up_trainable = ["3", s2l[args.scale]]
    if args.scaleFrom == 2:
        up_trainable.append("2")
    trainable=False
    for layer in gan.generator.layers:
        if 'upSample' in layer.name and any([layer.name.endswith('_'+s) for s in up_trainable]) :
            trainable = True            
        layer.trainable = trainable

    # Compile generator with frozen layers
    gan.compile_generator(gan.generator)

def gan_train(args, gan, common, first_epoch=1000000):
    '''Just a convenience function for training the GAN'''
    gan.train_srgan(
        epochs=100000,
        dataname='SRGAN_'+args.dataname,
        print_frequency=10000,    
        log_weight_frequency=5000,
        log_tensorboard_name='SRGAN_'+args.dataname,
        log_test_frequency=10000,
        first_epoch=1000000,
        **common
    )

def generator_train(args, gan, common, epochs=1):
    '''Just a convenience function for training the GAN'''
    gan.train_generator(
        epochs=1,
        dataname='SRResNet'+args.dataname,        
        steps_per_epoch=100000,        
        log_tensorboard_name='SRResNet'+args.dataname,        
        **common
    )

# Run script
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()
       
    # Common settings for all training stages
    common = {
        "batch_size": args.batch_size, 
        "workers": args.workers,
        "datapath_train": args.train,
        "datapath_validation": args.validation,
        "datapath_test":args.test,
        "steps_per_validation": 5000,
        "log_weight_path": args.weight_path, 
        "log_tensorboard_path": args.log_path,        
        "log_tensorboard_update_freq": 1000,
        "log_test_path": args.test_path,
        "crops_per_image": args.crops_per_image        
    }

    # Generator weight paths
    srresnet_path = os.path.join(args.weight_path, 'SRResNet_'+args.dataname+'_{}X'.format(args.scale))
    srrgan_G_path = os.path.join(args.weight_path, 'SRGAN_'+args.dataname+'_generator_'+str(args.scale)+'X.h5')
    srrgan_D_path = os.path.join(args.weight_path, 'SRGAN_'+args.dataname+'_discriminator_'+str(args.scale)+'X.h5')

    ## FIRST STAGE: TRAINING GENERATOR ONLY WITH MSE LOSS
    ######################################################

    # If we are doing transfer learning, only train top layer of the generator
    # And load weights from lower-upscaling model    
    if args.stage in ['all', 'mse']:
        if args.scaleFrom:

            # Ensure proper layer names
            BASE_G, BASE_D = reset_layer_names(args)

            # Load the properly named weights onto this model and freeze lower-level layers
            gan = SRGAN(upscaling_factor=args.scale)
            gan.load_weights(BASE_G, BASE_D, by_name=True)
            gan_freeze_layers(args, gan)
            generator_train(args, gan, common, 1)

            # Train entire generator for 3 epochs
            gan = SRGAN(upscaling_factor=args.scale)
            gan.load_weights(srresnet_path)
            generator_train(args, gan, common, 3)
        
        else:

            # As in paper - train for 10 epochs
            gan = SRGAN(upscaling_factor=args.scale)    
            generator_train(args, gan, common, 10)        

    ## SECOND STAGE: TRAINING GAN WITH HIGH LEARNING RATE
    ######################################################

    # Re-initialize & train the GAN - load just created generator weights
    if args.stage in ['all', 'gan']:
        gan = SRGAN(upscaling_factor=args.scale)
        gan.load_weights(srresnet_path)
        gan_train(args, gan, common, 1000000)

    ## THIRD STAGE: FINE TUNE GAN WITH LOW LEARNING RATE
    ######################################################
        
    # Re-initialize & fine-tune GAN - load generator & discriminator weights
    if args.stage in ['all', 'gan-finetune']:
        gan = SRGAN(
            gen_lr=1e-5, dis_lr=1e-5,
            upscaling_factor=args.scale
        )
        gan.load_weights(srrgan_G_path, srrgan_D_path)
        gan_train(args, gan, common, 1100000)
        