import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate
import config as c

import sklearn.metrics as sk

import numpy as np

from copy import deepcopy



def get_loss_outlier_progan(z1, z2, jac1,jac2, labels, margin = 500):
    positive_mask = (labels == 0).float()
    negative_mask = (labels == 1).float()
    loss_sample1 = 0.5 * torch.sum(z1 ** 2, dim=(1,))
    positive_loss1 =  (loss_sample1 -jac1)* positive_mask
    negative_loss1 = (-loss_sample1 -jac1) * negative_mask* (loss_sample1 <margin).float()
    shape_loss1 = torch.mean(negative_loss1 + positive_loss1)/ z1.shape[1]
    loss_sample2 =  0.5 * torch.sum(z2 ** 2, dim=(1,))  
    positive_loss2 =  (loss_sample2 -jac2)* positive_mask.float()
    negative_loss2 = (-loss_sample2 -jac2) * negative_mask.float()* (loss_sample2 <margin).float()
    shape_loss2 = torch.mean(negative_loss2 + positive_loss2 )/ z2.shape[1]
    shape_loss = (shape_loss1  + shape_loss2)/2

    cosine_similarity = torch.nn.functional.cosine_similarity(z1, z2)
    positive_loss = (1 - cosine_similarity) * positive_mask
    negative_loss = cosine_similarity * 0.1 * (cosine_similarity > 0.1).float() * negative_mask
    consistent_loss = (positive_loss.sum() + negative_loss.sum()) / len(labels)

    total_loss = shape_loss + consistent_loss * 5

    return shape_loss, consistent_loss, total_loss
