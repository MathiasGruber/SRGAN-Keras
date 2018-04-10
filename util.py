import os
import numpy as np
import matplotlib.pyplot as plt


def plot_test_images(model, loader, test_images, test_output, epoch):

    # Load the images to perform test on images
    imgs_hr, imgs_lr = loader.load_batch(batch_size=1, img_paths=test_images, training=False)

    # Create super resolution images
    imgs_sr = []
    for img in imgs_lr:
        imgs_sr.append(
            np.squeeze(
                model.generator.predict(
                    np.expand_dims(img, 0),
                    batch_size=1
                ),
                axis=0
            )
        )

    # Loop through images
    for img_hr, img_lr, img_sr, img_path in zip(imgs_hr, imgs_lr, imgs_sr, test_images):

        # Get the filename
        filename = os.path.basename(img_path).split(".")[0]

        # Images and titles
        images = {
            'Low Resolution': img_lr, 'SRGAN': img_sr, 'Original': img_hr
        }

        # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1                    
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (title, img) in enumerate(images.items()):
            axes[i].imshow(0.5 * img + 0.5)
            axes[i].set_title(title)
            axes[i].axis('off')
        plt.suptitle('{} - Epoch: {}'.format(filename, epoch))

        # Save directory                    
        savefile = os.path.join(test_output, "{}-Epoch{}.png".format(filename, epoch))
        fig.savefig(savefile)
        plt.close()
