#! /usr/bin/python
import os
import pickle
import datetime

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Activation, Add
from keras.layers import BatchNormalization, LeakyReLU, Conv2D, Dense
from keras.layers import UpSampling2D
from keras.optimizers import Adam
from keras.applications import VGG19

from keras.callbacks import TensorBoard, ReduceLROnPlateau

from .util import DataLoader, plot_test_images



class SRGAN():
    """
    Implementation of SRGAN as described in the paper:
    Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    https://arxiv.org/abs/1609.04802
    """

    def __init__(self, 
        height_lr=64, width_lr=64, channels=3, 
        upscaling_factor=4, 
        gen_lr=1e-4, dis_lr=1e-4, gan_lr=1e-4, 
        loss_weights=[1e-3, 1]
    ):
        """        
        :param int height_lr: Height of low-resolution images
        :param int width_lr: Width of low-resolution images
        :param int channels: Image channels
        :param int upscaling_factor: Up-scaling factor
        :param int gen_lr: Learning rate of generator
        :param int dis_lr: Learning rate of discriminator
        :param int gan_lr: Learning rate of GAN
        """

        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr

        # High-resolution image dimensions
        if upscaling_factor % 2 != 0:
            raise ValueError('Upscaling factor must be a multiple of 2; i.e. 2, 4, 8, etc.')
        self.upscaling_factor = upscaling_factor
        self.height_hr = int(self.height_lr * self.upscaling_factor)
        self.width_hr = int(self.width_lr * self.upscaling_factor)

        # Low-resolution and high-resolution shapes
        self.channels = channels
        self.shape_lr = (self.height_lr, self.width_lr, self.channels)
        self.shape_hr = (self.height_hr, self.width_hr, self.channels)
        
        # Scaling of losses
        self.loss_weights = loss_weights

        # Optimizers used by networks
        optimizer_vgg = Adam(0.0001, 0.9)
        optimizer_generator = Adam(gen_lr, 0.9)
        optimizer_discriminator = Adam(dis_lr, 0.9)
        optimizer_gan = Adam(gan_lr, 0.9)

        # Build the basic networks
        self.vgg = self.build_vgg(optimizer_vgg) # model1
        self.generator = self.build_generator(optimizer_generator) # model2
        self.discriminator = self.build_discriminator(optimizer_discriminator) # model3

        # Build the combined network
        self.srgan = self.build_srgan(optimizer_gan)

    
    def save_weights(self, filepath):
        """Save the generator and discriminator networks"""
        self.generator.save_weights(filepath + "_generator.h5")
        self.discriminator.save_weights(filepath + "_discriminator.h5")


    def load_weights(self, generator_weights=None, discriminator_weights=None):
        if generator_weights:
            self.generator.load_weights(generator_weights)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights)


    def build_vgg(self, optimizer):
        """
        Load pre-trained VGG19 from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        """

        # Input image to extract features from
        img = Input(shape=self.shape_hr)

        # Get the vgg network. Extract features from last conv layer
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]

        # Create model and compile
        model = Model(inputs=img, outputs=vgg(img))
        model.trainable = False
        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        return model


    def build_generator(self, optimizer, residual_blocks=16):
        """
        Build the generator network according to description in the paper.

        :param optimizer: Keras optimizer to use for network
        :param int residual_blocks: How many residual blocks to use
        :return: the compiled model
        """

        def residual_block(input):
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
            x = Activation('relu')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Add()([x, input])
            return x

        def deconv2d_block(input):
            x = UpSampling2D(size=2)(input)
            x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
            x = Activation('relu')(x)
            return x

        # Input low resolution image
        lr_input = Input(shape=(None, None, 3))

        # Pre-residual
        x_start = Conv2D(64, kernel_size=9, strides=1, padding='same')(lr_input)
        x_start = Activation('relu')(x_start)

        # Residual blocks
        r = residual_block(x_start)
        for _ in range(residual_blocks - 1):
            r = residual_block(r)

        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        x = BatchNormalization(momentum=0.8)(x)
        x = Add()([x, x_start])

        # Upsampling (if 4; run twice, if 8; run thrice, etc.)
        for _ in range(int(np.log(self.upscaling_factor) / np.log(2))):
            x = deconv2d_block(x)

        # Generate high resolution output
        hr_output = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(x)

        # Create model and compile
        model = Model(inputs=lr_input, outputs=hr_output)
        model.compile(
            loss='mse',
            optimizer=optimizer
        )
        return model


    def build_discriminator(self, optimizer, filters=64):
        """
        Build the discriminator network according to description in the paper.

        :param optimizer: Keras optimizer to use for network
        :param int filters: How many filters to use in first conv layer
        :return: the compiled model
        """

        def conv2d_block(input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input high resolution image
        img = Input(shape=self.shape_hr)
        x = conv2d_block(img, filters, bn=False)
        x = conv2d_block(x, filters, strides=2)
        x = conv2d_block(x, filters*2)
        x = conv2d_block(x, filters*2, strides=2)
        x = conv2d_block(x, filters*4)
        x = conv2d_block(x, filters*4, strides=2)
        x = conv2d_block(x, filters*8)
        x = conv2d_block(x, filters*8, strides=2)
        x = Dense(filters*16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        # Create model and compile
        model = Model(inputs=img, outputs=x)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        return model


    def build_srgan(self, optimizer):
        """Create the combined SRGAN network"""

        # Input LR images
        img_lr = Input(self.shape_lr)

        # Create a high resolution image from the low resolution one
        generated_hr = self.generator(img_lr)
        generated_features = self.vgg(generated_hr)

        # In the combined model we only train the generator
        self.discriminator.trainable = False

        # Determine whether the generator HR images are OK
        generated_check = self.discriminator(generated_hr)

        # Create model and compile
        model = Model(inputs=img_lr, outputs=[generated_check, generated_features])
        model.compile(
            loss=['binary_crossentropy', 'mse'],
            loss_weights=self.loss_weights,
            optimizer=optimizer
        )
        return model


    def train(self, 
        epochs, 
        dataname, datapath,
        batch_size=1, 
        test_images=None, 
        test_frequency=50,
        test_path="./images/samples/", 
        weight_frequency=None, 
        weight_path='./data/weights/', 
        log_path='./data/logs/',
        print_frequency=1
    ):
        """Train the SRGAN network

        :param int epochs: how many epochs to train the network for
        :param str dataname: name to use for storing model weights etc.
        :param str datapath: path for the image files to use for training
        :param int batch_size: how large mini-batches to use
        :param list test_images: list of image paths to perform testing on
        :param int test_frequency: how often (in epochs) should testing be performed
        :param str test_path: where should test results be saved
        :param int weight_frequency: how often (in epochs) should network weights be saved. None for never
        :param int weight_path: where should network weights be saved
        :param int print_frequency: how often (in epochs) to print progress to terminal
        """

        # Create data loader
        loader = DataLoader(
            datapath,
            self.height_hr, self.width_hr,
            self.height_lr, self.width_lr,
            self.upscaling_factor
        )

        # Shape of output from discriminator
        disciminator_output_shape = list(self.discriminator.output_shape)
        disciminator_output_shape[0] = batch_size
        disciminator_output_shape = tuple(disciminator_output_shape)

        # VALID / FAKE targets for discriminator
        real = np.ones(disciminator_output_shape)
        fake = np.zeros(disciminator_output_shape)        

        # Each epoch == "update iteration" as defined in the paper        
        losses = []
        print_losses = {"G": [], "D": []}
        for epoch in range(epochs):

            # Start epoch time
            if epoch % (print_frequency + 1) == 0:
                start_epoch = datetime.datetime.now()

            # Train discriminator
            imgs_hr, imgs_lr = loader.load_batch(batch_size)
            generated_hr = self.generator.predict(imgs_lr)
            real_loss = self.discriminator.train_on_batch(imgs_hr, real)
            fake_loss = self.discriminator.train_on_batch(generated_hr, fake)
            discriminator_loss = 0.5 * np.add(real_loss, fake_loss)

            # Train generator
            imgs_hr, imgs_lr = loader.load_batch(batch_size)
            features_hr = self.vgg.predict(imgs_hr)
            generator_loss = self.srgan.train_on_batch(imgs_lr, [real, features_hr])

            # Save losses            
            print_losses['G'].append(generator_loss)
            print_losses['D'].append(discriminator_loss)

            # Show the progress
            if epoch % print_frequency == 0:
                g_avg_loss = np.array(print_losses['G']).mean(axis=0)
                d_avg_loss = np.array(print_losses['D']).mean(axis=0)
                losses.append({'generator': g_avg_loss, 'discriminator': d_avg_loss})
                print("Epoch {}/{} | Time: {}s\n>> Generator/GAN: {}\n>> Discriminator: {}\n".format(
                    epoch, epochs,
                    (datetime.datetime.now() - start_epoch).seconds,
                    ", ".join(["{}={:.3f}".format(k, v) for k, v in zip(self.srgan.metrics_names, g_avg_loss)]),
                    ", ".join(["{}={:.3f}".format(k, v) for k, v in zip(self.discriminator.metrics_names, d_avg_loss)])
                ))
                print_losses = {"G": [], "D": []}

            # If test images are supplied, show them to the user
            if test_images and epoch % test_frequency == 0:
                plot_test_images(self, loader, test_images, test_path, epoch)

            # Check if we should save the network weights
            if weight_frequency and epoch % weight_frequency == 0:

                # Save the network weights
                self.save_weights(os.path.join(weight_path, dataname))

                # Save the recorded losses
                pickle.dump(losses, open(os.path.join(weight_path, dataname+'_losses.p'), 'wb'))


# Run the SRGAN network
if __name__ == '__main__':

    # Instantiate the SRGAN object
    print(">> Creating the SRGAN network")
    gan = SRGAN(gen_lr=1e-5)

    # Load previous imagenet weights
    print(">> Loading old weights")
    gan.load_weights('../data/weights/imagenet_generator.h5', '../data/weights/imagenet_discriminator.h5')

    # Train the SRGAN
    gan.train(
        epochs=100000,
        dataname='imagenet',
        datapath='../data/imagenet/train/',
        batch_size=16,
        test_images=[
            '../data/buket.jpg'
            
        ],        
        test_frequency=1000,
        test_path='../images/samples/',
        weight_path='../data/weights/',
        weight_frequency=1000,
        print_frequency=10,
    )
