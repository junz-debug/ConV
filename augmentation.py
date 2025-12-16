

import numpy as np
from torchvision import transforms
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from transforms import GaussianBlur, make_normalize_transform


class DataAugmentationDINO(object):
    """Data augmentation class for DINO-based detection."""
    
    def __init__(self, local_crops_number):
        self.local_crops_number = local_crops_number
        
        # Geometric augmentation
        self.geometric_augmentation_global1 = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.8),
        ])
        
        # Color jittering
        scale = 0.3
        color_jittering1 = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.6 * scale, 
                    contrast=0.8 * scale, 
                    saturation=0.4 * scale, 
                    hue=0.1 * scale
                )],
                p=0.8,
            ),
        ])
        
        global_transfo2_extra = transforms.Compose([
            GaussianBlur(p=0),
        ])
        
        # Normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            make_normalize_transform(),
        ])
        
        self.source_trans = transforms.Compose([
            transforms.ToTensor(),
            make_normalize_transform(),
        ])
        
        self.crop = transforms.Compose([
            transforms.RandomCrop(224),
        ])
        
        self.centercrop = transforms.Compose([
            transforms.CenterCrop(224),
        ])
        
        self.global_transfo_all = transforms.Compose([
            self.geometric_augmentation_global1, 
            color_jittering1, 
            global_transfo2_extra, 
            self.normalize
        ])

    def __call__(self, image):
        output = {}
        output["source"] = []
        output["global_crops"] = []
        
        if np.array(image).shape[0] < 224 or np.array(image).shape[1] < 224:
            crops_all = [
                self.centercrop(image) for _ in range(self.local_crops_number)
            ]
            for crops_image in crops_all:
                output["source"].append(self.source_trans(crops_image))
            for crops_image in crops_all:
                output["global_crops"].append(self.global_transfo_all(crops_image))
        else:
            crops_all1 = [
                self.crop(image) for _ in range(self.local_crops_number)
            ]
            
            for crops_image in crops_all1:
                output["source"].append(self.source_trans(crops_image))
                output["global_crops"].append(self.global_transfo_all(crops_image))
        

        return output


