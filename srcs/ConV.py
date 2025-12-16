import argparse
import os
import random
import torch
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import Dataset
from PIL import Image
import pickle
from copy import deepcopy
import sklearn.metrics as sk

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from augmentation import DataAugmentationDINO



def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Stable cumulative sum."""
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    """Calculate FPR at a specific recall level."""
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    y_true = (y_true == pos_label)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    thresholds = y_score[threshold_idxs]
    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    
    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]


def get_measures(_pos, _neg, recall_level=0.95):
    """Calculate evaluation measures."""
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr, threshold = fpr_and_fdr_at_recall(labels, examples, recall_level)
    return auroc, aupr, fpr


def find_best_threshold(y_true, y_pred):
    """Find best threshold for binary classification.
    We assume first half is real 0, and the second half is fake 1
    """
    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min():  # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  

        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres


def calculate_acc(y_true, y_pred, thres):
    """Calculate accuracy metrics."""
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def calculate_cosine_similarity(tensor1, tensor2):
    """Calculate cosine similarity."""
    return torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=-1)




def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp", "PNG"]):
    """Recursively read image files."""
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def process_image_to_square(img):
    """Process image to square if it's not already square.
    
    Args:
        img: PIL Image object
        
    Returns:
        PIL Image object (square, 256x256)
    """
    width, height = img.size

    if width == height:
        return img
    
    new_size = min(width, height)
    
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = left + new_size
    bottom = top + new_size
    
    img_cropped = img.crop((left, top, right, bottom))
    
    img_resized = img_cropped.resize((256, 256), Image.BICUBIC)
    
    return img_resized

def get_list(path, must_contain=''):
    """Get list of image paths."""
    print(path)
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class RealFakeDataset(Dataset):
    """Dataset for real and fake images."""
    
    def __init__(self, real_path, fake_path, max_sample, trans=None):
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        self.transform = trans

    def read_path(self, real_path, fake_path, max_sample):
        real_list = get_list(real_path)
        fake_list = get_list(fake_path)

        if max_sample == 0:
            real_list = random.sample(real_list, 1000)
            fake_list = random.sample(fake_list, 1000)
        elif max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]

        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        
        img = process_image_to_square(img)
        
        img = self.transform(img)
        return img, label


def validate(model, loader, crops, fixed_threshold=0, aggregation='mean'):
    """Validate the model.
    
    Args:
        aggregation: 'mean', 'max', 'min', or 'median' - how to aggregate distances across crops
    """
    model.cuda()
    model.eval()

    score_in = []
    score_out = []
    y_true = []
    y_pred = []
    y_prediction = []
    y_labels = []
    
    with torch.no_grad():
        print("Length of dataset: %d" % (len(loader.dataset)))
        for img, labels in loader:
            y_true.append(labels)
            l2_distances = []
            
            for i in range(crops):
                original_embedding = model(img["source"][i].half().cuda())
                original_embedding = original_embedding / original_embedding.norm(dim=-1, keepdim=True)

                noisy_embedding = model(img["global_crops"][i].half().cuda())
                noisy_embedding = noisy_embedding / noisy_embedding.norm(dim=-1, keepdim=True)
                
                l2_distance = calculate_cosine_similarity(original_embedding, noisy_embedding)
                l2_distances.append(l2_distance.detach().cpu().numpy())

            distance = np.stack(l2_distances, axis=1)
        
            if aggregation == 'max':
                original_cosine_similarity = np.max(distance, axis=1)
            elif aggregation == 'min':
                original_cosine_similarity = np.min(distance, axis=1)
            elif aggregation == 'median':
                original_cosine_similarity = np.median(distance, axis=1)
            else:  # default to 'mean'
                original_cosine_similarity = np.mean(distance, axis=1)

            y_prediction.extend(1 - original_cosine_similarity)
            y_labels.extend(labels.flatten().tolist())
            y_pred.append(original_cosine_similarity)

            for i in range(labels.shape[0]):
                if labels[i] == 1:
                    score_in.append(original_cosine_similarity[i])
                else:
                    score_out.append(original_cosine_similarity[i])
    
    y_labels = np.array(y_labels)
    y_prediction = np.array(y_prediction)

    # Find best threshold
    if fixed_threshold == 0:
        best_thres = find_best_threshold(y_labels, y_prediction)
        print(f"Best threshold: {best_thres:.6f}")
        r_acc1, f_acc1, acc1 = calculate_acc(y_labels, y_prediction, best_thres)
    else:
        r_acc1, f_acc1, acc1 = calculate_acc(y_labels, y_prediction, fixed_threshold)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    ap = average_precision_score(1 - y_true, y_pred)
    auroc, aupr, fpr = get_measures(score_out, score_in)
    
    print("AUROC: " + str(auroc))
    print("AP: " + str(ap))
    print("ACC: " + str(acc1))

    return auroc, ap, acc1


def main():
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    parser = argparse.ArgumentParser(description='Simple Fake Image Detection')
    parser.add_argument('--real_path', type=str, required=True, help='Path to real images')
    parser.add_argument('--fake_path', type=str, required=True, help='Path to fake images')
    parser.add_argument('--max_sample', type=int, default=1000, help='Max samples per class')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--crops_num', type=int, default=5, help='Number of crops')
    parser.add_argument('--threshold', type=float, default=0, help='Classification threshold')
    parser.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'max', 'min', 'median'],
                        help='How to aggregate distances across crops: mean, max, min, or median')

    opt = parser.parse_args()

    # Load DINOv2 model
    print("Loading DINOv2 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model = model.half()
    print("Model loaded.")

    # Setup augmentation
    data_transform = DataAugmentationDINO(
        local_crops_number=opt.crops_num
    )

    # Create dataset
    dataset = RealFakeDataset(
        opt.real_path,
        opt.fake_path,
        opt.max_sample,
        trans=data_transform
    )

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers
    )
    
    # Validate
    auroc, ap, acc = validate(model, loader, opt.crops_num, opt.threshold, opt.aggregation)
    
    print("\nFinal Results:")
    print(f"AUROC: {auroc:.4f}")
    print(f"AP: {ap:.4f}")
    print(f"ACC: {acc:.4f}")
    
    import json
    result = {
        "auroc": float(auroc),
        "ap": float(ap),
        "acc": float(acc)
    }
    print(f"RESULT_JSON:{json.dumps(result)}")


if __name__ == '__main__':
    main()

