import skimage
import numpy as np
from skimage import util
# Our Test Images procuder
class Test_Image:
    def __int__(self):
        self.is_one = []
        self.is_not_one = []
        self.augmented_one = []
        self.augmented_non_one = []
    def is_one(self):
        #Image declaration
        img_one = np.zeros([28, 28])
        img_one[ 7:8, 12:16] = 90
        img_one[ 8:9, 10:12] = 80
        img_one[ 9:10, 8:10] = 50
        img_one[ 6:17, 14:16] = 110 - np.random.randint(30, size=(11, 2))
        # To make Image more realistic we performe erosion of dilation of image
        self.img_one = skimage.morphology.erosion(skimage.morphology.dilation(img_one))
        return self.img_one
    def is_not_one(self):
        # Image declaration
        img_not_one = np.zeros([28, 28])
        img_not_one [ 8:18, 8:9] = 100 - np.random.randint(30, size=(10, 1))
        img_not_one [ 8:18, 18:19] = 100 - np.random.randint(30, size=(10, 1))
        img_not_one [ 8:9, 8:19] = 100 - np.random.randint(30, size=(1, 11))
        img_not_one [ 18:19, 8:19] = 100 - np.random.randint(30, size=(1, 11))
        # To make Image more realistic we performe erosion of dilation of image
        self.img_not_one = skimage.morphology.closing(img_not_one)
        return self.img_not_one
    def image_augmentation(self):
        #Augmentation - Colour Inversion
        self.augmented_one = util.invert(self.img_one)
        self.augmented_non_one = util.invert(self.img_not_one)
        return self.augmented_one, self.augmented_non_one

