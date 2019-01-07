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
python train.py --train C:/Documents/Kaggle/Kaggle-imagenet/input/DET/train --validation C:/Documents/Kaggle/Kaggle-imagenet/input/DET/test --scale 2
"""

def parse_args():
    parser = ArgumentParser(description='Training script for SRGAN')

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

    # If we are doing transfer learning, only train top layer of the generator
    # And load weights from lower-upscaling model    
    if args.scaleFrom:

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

        # Load the properly named weights onto this model
        gan = SRGAN(upscaling_factor=args.scale)
        gan.load_weights(BASE_G, BASE_D, by_name=True)

        # 4X -> 8X block always trainable. 2X -> 4X only if going from 2X.
        up_trainable = ["3"]
        if args.scaleFrom == 2:
            up_trainable.append("2")
        trainable=False
        for layer in gan.generator.layers:
            if 'upSample' in layer.name and all([not layer.name.endswith('_'+s) for s in up_trainable]) :
                trainable = True            
            layer.trainable = trainable
        
        # Train with 1 epoch on top layer
        gan.compile_generator(gan.generator)
        gan.train_generator(
            epochs=1,
            dataname='SRResNet_'+args.dataname,        
            steps_per_epoch=100000,        
            log_tensorboard_name='SRResNet_'+args.dataname,        
            **common
        )

        # Train entire generator for 3 epochs
        gan = SRGAN(upscaling_factor=args.scale)
        gan.load_weights(
            os.path.join(args.weight_path, 'SRResNet_'+args.dataname+'_{}X'.format(args.scale)), 
        )
        gan.train_generator(
            epochs=3,
            dataname='SRResNet_'+args.dataname,        
            steps_per_epoch=100000,        
            log_tensorboard_name='SRResNet_'+args.dataname,        
            **common
        )
    
    else:

        # As in paper - train for 10 epochs
        gan = SRGAN(upscaling_factor=args.scale)    
        gan.train_generator(
            epochs=10,
            dataname='SRResNet_'+args.dataname,        
            steps_per_epoch=100000,        
            log_tensorboard_name='SRResNet_'+args.dataname,        
            **common
        )

    # Re-initialize & train the GAN - load just created generator weights
    gan = SRGAN(
        upscaling_factor=args.scale
    )
    gan.load_weights(
        os.path.join(args.weight_path, 'SRResNet_'+args.dataname+'_{}X'.format(args.scale)), 
    )
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
        
    # Re-initialize & fine-tune GAN - load generator & discriminator weights
    gan = SRGAN(
        gen_lr=1e-5, dis_lr=1e-5,
        upscaling_factor=args.scale
    )
    gan.load_weights(
        os.path.join(args.weight_path, 'SRGAN_'+args.dataname+'_generator_'+str(args.scale)+'X.h5'), 
        os.path.join(args.weight_path, 'SRGAN_'+args.dataname+'_discriminator_'+str(args.scale)+'X.h5')
    )
    gan.train_srgan(
        epochs=200000,
        dataname='SRGAN_'+args.dataname,
        print_frequency=10000,    
        log_weight_frequency=5000,
        log_tensorboard_name='SRGAN_'+args.dataname,
        log_test_frequency=10000,
        first_epoch=1100000,
        **common
    )
        