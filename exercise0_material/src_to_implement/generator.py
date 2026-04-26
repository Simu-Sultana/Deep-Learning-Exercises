import os, os.path
import json
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path:str, label_path:str, batch_size:int, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path=file_path
        self.label_path=label_path
        
        self.batch_size=batch_size
        self.image_size=image_size
        
        self.mirroring=mirroring
        self.rotation=rotation
        
        self.filenames=os.listdir(file_path)
        
        if shuffle:
            np.random.shuffle(self.filenames)

        self.epoch=0

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        # Load the labels JSON 
        with open(self.label_path, 'r') as l:
            label_dict = json.load(l)

        images = []
        labels = []

        if self.batch_size > len(self.filenames):
            self.epoch = self.current_epoch() + 1
            self.filenames = self.filenames + os.listdir(self.file_path)
        
        for filename in self.filenames[:self.batch_size]:
                
            img = np.load(f'{self.file_path}/{filename}')
            img = self.augment(Image.fromarray(img))
            img = transform.resize(np.asarray(img), self.image_size)

            name, _ = filename.split('.')
            
            label = label_dict[name]
            
            images.append(img)
            labels.append(label)

            self.filenames.pop(0)
        
        return np.asarray(images), np.asarray(labels)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        # Horizontal flip in the image with probability of 0.5
        if self.mirroring is True and np.random.random()<0.5:
            img=img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Rotate image randomly
        rotang=[Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
        if self.rotation is True:
            img=img.transpose(rotang[np.random.randint(0,2)])
            
        return img

    def current_epoch(self):
        # return the current epoch number
        
        epoch=self.epoch
        return epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function

        classname = self.class_dict[x] 
        return classname

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        
        images, labels = self.next()
        
        nrows = int(self.batch_size/3)
        ncols = 3
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6,6))

        for i, axis in enumerate(ax.flat): #nrows*ncols-1
            axis.imshow(images[i])
            classname=self.class_name(labels[i])
            axis.set_title(classname)

        plt.show()
