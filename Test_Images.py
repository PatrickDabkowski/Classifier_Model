import skimage
import numpy as np
# Our Test Images procuder
class Test_Image:
    def is_one(self):
        #Image declaration
        img_one = np.zeros([15, 15])
        img_one[2:3, 7:11] = 90
        img_one[3:4, 5:7] = 80
        img_one[4:5, 3:5] = 50
        img_one[1:12, 9:11] = 110 - np.random.randint(30, size=(11, 2))
        # To make Image more realistic we performe erosion of dilation of image
        img_one = skimage.morphology.erosion(skimage.morphology.dilation(img_one))
        return img_one
    def is_not_one(self):
        # Image declaration
        img_not_one = np.zeros([15, 15])
        img_not_one[2:12, 2:3] = 100 - np.random.randint(30, size=(10, 1))
        img_not_one[2:12, 12:13] = 100 - np.random.randint(30, size=(10, 1))
        img_not_one[2:3, 2:13] = 100 - np.random.randint(30, size=(1, 11))
        img_not_one[12:13, 2:13] = 100 - np.random.randint(30, size=(1, 11))
        # To make Image more realistic we performe erosion of dilation of image
        img_not_one = skimage.morphology.closing(img_not_one)
        return img_not_one
    def image_augmentation(self):

