
import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from random import choice, shuffle
import random
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
import pickle
import os 
from torchvision import transforms

import torch 
# import cv2
from PIL import Image
import numpy as np

from my_transforms import (
    GaussianBlur,
    make_normalize_transform,
    make_normalize_transform_clip,
)



class DataAugmentationDINO_conv(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
    

        self.source_trans = transforms.Compose([
            # transforms.RandomCrop(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            make_normalize_transform(),
        ])

        # self.crop = transforms.Compose([
        #     transforms.RandomCrop(224),
           
        # ])
        self.crop1 = transforms.Compose([
                    transforms.RandomCrop(224),  
                    ])
        

        self.crop2 = transforms.Compose([
                    # transforms.Resize(224),  
                    transforms.Resize(384),
                    transforms.RandomCrop(224),  
                    ])
        

        self.crop3 = transforms.Compose([
                    transforms.Resize(256),  
                    transforms.RandomCrop(224), 
                    ])
        

        self.centercrop = transforms.Compose([
            transforms.CenterCrop(224),
           
        ])


        self.rotate = transforms.RandomRotation(90)

        self.geometric_augmentation_global1 = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1),
            ]
        )

        


    def __call__(self, image):
        output = {}
        output["source"] = []
        output['trans'] = []
        trans = transforms.Lambda(lambda img: data_augment(img))

        scale = 0.3  
        color_jittering1 = transforms.Compose(
            [   # transforms.RandomCrop(224),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4 * scale, contrast=0.4 * scale, saturation=0.2 * scale, hue=0.1 * scale)],
                    p=1, #0.8
                ),
                # transforms.RandomGrayscale(p=1), #0.2
            ]
        )


        if np.array(image).shape[0]<224 or np.array(image).shape[1]<224:
            image = self.centercrop((image))
        
        p = random.random() 
    
        crops_image = self.crop1(trans(image))
        
        output["source"].append(self.source_trans(crops_image))


        self.global_transfo_all = transforms.Compose([self.geometric_augmentation_global1, color_jittering1, GaussianBlur(p=0), self.source_trans])
        output["trans"].append(self.global_transfo_all(crops_image))

        return output
    


class DataAugmentationDINO_conv_test(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
    

        self.source_trans = transforms.Compose([
            # transforms.RandomCrop(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            make_normalize_transform(),
        ])
        self.crop1 = transforms.Compose([
                    transforms.RandomCrop(224), 
                    ])

        self.crop3 = transforms.Compose([
                    transforms.Resize(256),  
                    transforms.RandomCrop(224),  
                    ])
        
        self.crop2 = transforms.Compose([
                    # transforms.Resize(224),  
                    transforms.Resize(384),
                    transforms.RandomCrop(224),  
                    ])
        

        self.centercrop = transforms.Compose([
            transforms.CenterCrop(224),
           
        ])
        self.rotate = transforms.RandomRotation(90)

        self.geometric_augmentation_global1 = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=1),
            ]
        )

        


    def __call__(self, image):
        output = {}
        output["source"] = []
        output['trans'] = []

        scale = 0.3   #best 0.3
        color_jittering1 = transforms.Compose(
            [   # transforms.RandomCrop(224),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4 * scale, contrast=0.4 * scale, saturation=0.2 * scale, hue=0.1 * scale)],
                    p=1, #0.8
                ),
                # transforms.RandomGrayscale(p=1), #0.2
            ]
        )

        crops_image = self.centercrop(image)
        # crops_image = data_augment(crops_image)
        
        output["source"].append(self.source_trans(crops_image))  


        self.global_transfo_all = transforms.Compose([self.geometric_augmentation_global1, color_jittering1, GaussianBlur(p=0), self.source_trans])
        output["trans"].append(self.global_transfo_all(crops_image))

        return output
    
def data_augment(img):
    img = np.array(img)
    # print(np.max(img)) 225
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random.random() < 0.3:
        sig = sample_continuous([0.0, 3.0])
        gaussian_blur(img, sig)

    if random.random() < 0.3:
        method = sample_discrete(['cv2','pil'])
        qual = sample_discrete([30,100])
        img = jpeg_from_key(img, qual, method)
    


    return Image.fromarray(img)



def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]

    if img_cv2 is None or img_cv2.size == 0:
        raise ValueError("图像为空或无效。")
    if img_cv2.dtype != 'uint8':
        img_cv2 = img_cv2.astype('uint8')
    
    compress_val = int(compress_val)
    if not (0 <= compress_val <= 100):
        raise ValueError("压缩质量值必须在 0 到 100 之间")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
