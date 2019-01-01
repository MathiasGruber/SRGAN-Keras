import os
import gc
import numpy as np
from PIL import Image
from random import choice
from keras.utils import Sequence

try:
    import matplotlib.pyplot as plt
except:
    pass


class DataLoader(Sequence):
    def __init__(self, datapath, batch_size, height_hr, width_hr, scale, crops_per_image):
        """        
        :param string datapath: filepath to training images
        :param int height_hr: Height of high-resolution images
        :param int width_hr: Width of high-resolution images
        :param int height_hr: Height of low-resolution images
        :param int width_hr: Width of low-resolution images
        :param int scale: Upscaling factor
        """

        # Store the datapath
        self.datapath = datapath
        self.batch_size = batch_size
        self.height_hr = height_hr
        self.height_lr = int(height_hr / scale)
        self.width_hr = width_hr
        self.width_lr = int(width_hr / scale)
        self.scale = scale
        self.crops_per_image = crops_per_image
        self.total_imgs = None
        
        # Options for resizing
        self.options = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
        
        # Check data source
        self.img_paths = []
        for dirpath, _, filenames in os.walk(self.datapath):
            for filename in [f for f in filenames if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg'])]:
                self.img_paths.append(os.path.join(dirpath, filename))
        self.total_imgs = len(self.img_paths)
        print(f">> Found {self.total_imgs} images in dataset")
    
    def random_crop(self, img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y+dy), x:(x+dx), :]

    @staticmethod
    def scale_lr_imgs(imgs):
        """Scale low-res images prior to passing to SRGAN"""
        return imgs / 255.
    
    @staticmethod
    def unscale_lr_imgs(imgs):
        """Un-Scale low-res images"""
        return imgs * 255
    
    @staticmethod
    def scale_hr_imgs(imgs):
        """Scale high-res images prior to passing to SRGAN"""
        return imgs / 127.5 - 1
    
    @staticmethod
    def unscale_hr_imgs(imgs):
        """Un-Scale high-res images"""
        return (imgs + 1.) * 127.5
    
    @staticmethod
    def load_img(path):
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    
    def __len__(self):
        return int(self.total_imgs / float(self.batch_size))
    
    def __getitem__(self, idx):
        return self.load_batch(idx=idx)        

    def load_batch(self, idx=0, img_paths=None, training=True, bicubic=False):
        """Loads a batch of images from datapath folder""" 

        # Starting index to look in
        cur_idx = 0
        if not img_paths:
            cur_idx = idx*self.batch_size            
            
        # Scale and pre-process images
        imgs_hr, imgs_lr = [], []
        while True:

            # Check if done with batch
            if img_paths is None:
                if cur_idx >= self.total_imgs:
                    cur_idx = 0
                if len(imgs_hr) >= self.batch_size:
                    break
            if img_paths is not None and len(imgs_hr) == len(img_paths):
                break            
            
            try: 
                # Load image
                img_hr = None
                if img_paths:
                    img_hr = self.load_img(img_paths[cur_idx])
                else:
                    img_hr = self.load_img(self.img_paths[cur_idx])

                # Create HR images to go through
                img_crops = []
                if training:
                    for i in range(self.crops_per_image):
                        #print(idx, cur_idx, "Loading crop: ", i)
                        img_crops.append(self.random_crop(img_hr, (self.height_hr, self.width_hr)))
                else:
                    img_crops = [img_hr]
 
                # Downscale the HR images and save
                for img_hr in img_crops:

                    # TODO: Refactor this so it does not occur multiple times
                    if img_paths is None:
                        if cur_idx >= self.total_imgs:
                            cur_idx = 0
                        if len(imgs_hr) >= self.batch_size:
                            break
                    if img_paths is not None and len(imgs_hr) == len(img_paths):
                        break   

                    # For LR, do bicubic downsampling
                    method = Image.BICUBIC if bicubic else choice(self.options)
                    lr_shape = (int(img_hr.shape[1]/self.scale), int(img_hr.shape[0]/self.scale))           
                    img_lr = Image.fromarray(img_hr.astype(np.uint8))
                    img_lr = np.array(img_lr.resize(lr_shape, method))

                    # Scale color values
                    img_hr = self.scale_hr_imgs(img_hr)
                    img_lr = self.scale_lr_imgs(img_lr)

                    # Store images
                    imgs_hr.append(img_hr)
                    imgs_lr.append(img_lr)
                
            except Exception as e:
                pass
            finally:
                cur_idx += 1

        # Convert to numpy arrays when we are training 
        # Note: all are cropped to same size, which is not the case when not training
        if training:
            imgs_hr = np.array(imgs_hr)
            imgs_lr = np.array(imgs_lr)

        # Return image batch
        return imgs_lr, imgs_hr


def plot_test_images(model, loader, datapath_test, test_output, epoch, name='SRGAN'):
    """        
    :param SRGAN model: The trained SRGAN model
    :param DataLoader loader: Instance of DataLoader for loading images
    :param str datapath_test: path to folder with testing images
    :param string test_output: Directory path for outputting testing images
    :param int epoch: Identifier for how long the model has been trained
    """

    try:
        
        # Get the location of test images
        test_images = [os.path.join(datapath_test, f) for f in os.listdir(datapath_test) if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg'])]
        
        # Load the images to perform test on images
        imgs_lr, imgs_hr = loader.load_batch(img_paths=test_images, training=False, bicubic=True)

        # Create super resolution and bicubic interpolation images
        imgs_sr = []
        imgs_bc = []
        for i in range(len(test_images)):

            # Bicubic interpolation
            pil_img = loader.unscale_lr_imgs(imgs_lr[i]).astype('uint8')
            pil_img = Image.fromarray(pil_img)        
            hr_shape = (imgs_hr[i].shape[1], imgs_hr[i].shape[0])

            imgs_bc.append(
                loader.scale_lr_imgs(
                    np.array(pil_img.resize(hr_shape, resample=Image.BICUBIC))
                )
            )

            # SRGAN prediction
            imgs_sr.append(
                np.squeeze(
                    model.generator.predict(
                        np.expand_dims(imgs_lr[i], 0),
                        batch_size=1
                    ),
                    axis=0
                )
            )

        # Unscale colors values
        imgs_lr = [loader.unscale_lr_imgs(img).astype(np.uint8) for img in imgs_lr]
        imgs_hr = [loader.unscale_hr_imgs(img).astype(np.uint8) for img in imgs_hr]
        imgs_sr = [loader.unscale_hr_imgs(img).astype(np.uint8) for img in imgs_sr]

        # Loop through images
        for img_hr, img_lr, img_bc, img_sr, img_path in zip(imgs_hr, imgs_lr, imgs_bc, imgs_sr, test_images):

            # Get the filename
            filename = os.path.basename(img_path).split(".")[0]

            # Images and titles
            images = {
                'Low Resolution': img_lr, 
                'Bicubic Interpolation': img_bc, 
                name: img_sr, 
                'Original': img_hr
            }

        # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1                    
            fig, axes = plt.subplots(1, 4, figsize=(40, 10))
            for i, (title, img) in enumerate(images.items()):
                axes[i].imshow(img)
                axes[i].set_title("{} - {}".format(title, img.shape))
                axes[i].axis('off')
            plt.suptitle('{} - Epoch: {}'.format(filename, epoch))

            # Save directory                    
            savefile = os.path.join(test_output, "{}-Epoch{}.png".format(filename, epoch))
            fig.savefig(savefile)
            plt.close()
            gc.collect()
    except Exception as e:
        print(">> Could not perform printing. Maybe matplotlib is not installed.")
