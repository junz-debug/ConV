"""
Extract DINOv2 features for training and testing datasets
This script pre-extracts features to speed up training and testing
"""

import argparse
import os
import torch
import torch.utils.data
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pickle
import random
from tqdm import tqdm
from torch.cuda.amp import autocast

# Import functions and classes directly (to avoid pickle issues on Windows)
def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp", "PNG"]):
    """Recursively read image files."""
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if len(file.split('.')) > 1 and (file.split('.')[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


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


def get_image_paths_from_subdirs(root_dir):
    """Get image paths from subdirectories containing 0_real and 1_fake folders."""
    real_images = []
    fake_images = []
    
    print(f"Reading images from subdirectories in: {root_dir}")
    
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        real_dir = os.path.join(subdir_path, '0_real')
        fake_dir = os.path.join(subdir_path, '1_fake')
        
        if os.path.isdir(real_dir):
            real_list = get_list(real_dir)
            real_images.extend(real_list)
            print(f"  Found {len(real_list)} real images in {subdir}/0_real")
        
        if os.path.isdir(fake_dir):
            fake_list = get_list(fake_dir)
            fake_images.extend(fake_list)
            print(f"  Found {len(fake_list)} fake images in {subdir}/1_fake")
    
    print(f"Total: {len(real_images)} real images, {len(fake_images)} fake images")
    return real_images, fake_images


class RealFakeDataset(Dataset):
    """Simplified dataset class for real and fake images."""
    
    def __init__(self, real_path, fake_path, data_mode, max_sample, arch, trans=None, path_mode='separate'):
        assert data_mode in ["wang2020", "ours"]
        assert path_mode in ["separate", "subdirs"]
        
        self.path_mode = path_mode
        
        # Read image paths based on mode
        if path_mode == 'subdirs':
            if fake_path is None or fake_path == '':
                root_dir = real_path
            else:
                root_dir = real_path
            real_list, fake_list = get_image_paths_from_subdirs(root_dir)
        else:
            if type(real_path) == str and type(fake_path) == str:
                real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
            else:
                real_list = []
                fake_list = []
                for real_p, fake_p in zip(real_path, fake_path):
                    real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                    real_list += real_l
                    fake_list += fake_l

        # Apply max_sample limit
        if max_sample > 0 and max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = min(len(real_list), len(fake_list))
                if max_sample == 0:
                    max_sample = 1000
                print(f"not enough images, max_sample falling to {max_sample}")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]

        self.total_list = real_list + fake_list

        # Create labels dictionary
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        self.transform = trans

    def read_path(self, real_path, fake_path, data_mode, max_sample):
        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]

        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Use first image as fallback
            img_path = self.total_list[0]
            label = self.labels_dict[img_path]
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            return img, label

# Import augmentation classes
try:
    from augmentations_fconv import DataAugmentationDINO_conv
except ImportError:
    try:
        # Try alternative import path
        sys.path.insert(0, os.path.dirname(__file__))
        from augmentations_fconv import DataAugmentationDINO_conv
    except ImportError:
        print("Error: Could not import DataAugmentationDINO_conv")
        print("Please check if augmentations_fconv.py exists")
        sys.exit(1)



class FeatureExtractor:
    """Extract and save DINOv2 features."""
    
    def __init__(self, model, transform, crops_num=1):
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.transform = transform
        self.crops_num = crops_num
    
    def extract_features_from_dataset(self, dataset, output_path, batch_size=64, num_workers=4):
        """Extract features from dataset and save to file.
        Each image is transformed once, saving one source feature and one transformed feature.
        """
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        all_features_source = []
        all_features_trans = []
        all_labels = []
        
        print(f"Extracting features from {len(dataset)} images...")
        with torch.no_grad():
            for inputs, labels in tqdm(loader):
                with autocast():
                    # Extract features from source and transformed images
                    # Only use the first crop (index 0) since we only transform once
                    source_feat = self.model(inputs["source"][0].half().cuda())
                    # source_feat = source_feat / source_feat.norm(dim=-1, keepdim=True)
                    all_features_source.append(source_feat.cpu().numpy())
                    
                    trans_feat = self.model(inputs["trans"][0].half().cuda())
                    # trans_feat = trans_feat / trans_feat.norm(dim=-1, keepdim=True)
                    all_features_trans.append(trans_feat.cpu().numpy())
                    
                    all_labels.append(labels.numpy())
        
        # Concatenate all features
        features_source = np.concatenate(all_features_source, axis=0)  # Shape: (N, feature_dim)
        features_trans = np.concatenate(all_features_trans, axis=0)      # Shape: (N, feature_dim)
        labels = np.concatenate(all_labels, axis=0)                      # Shape: (N,)
        
        # Save to file
        output_data = {
            'features_source': features_source,
            'features_trans': features_trans,
            'labels': labels
        }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"Features saved to {output_path}")
        print(f"  Source features shape: {features_source.shape}")
        print(f"  Trans features shape: {features_trans.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        return features_source, features_trans, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract DINOv2 features from datasets')
    parser.add_argument('--real_path', type=str, required=True, help='Path to real images (or root path for subdirs mode)')
    parser.add_argument('--fake_path', type=str, default=None, help='Path to fake images (not needed for subdirs mode)')
    parser.add_argument('--data_mode', type=str, default='ours', choices=['wang2020', 'ours'])
    parser.add_argument('--path_mode', type=str, default='separate', choices=['separate', 'subdirs'],
                        help='Path mode: separate or subdirs')
    parser.add_argument('--max_sample', type=int, default=0, help='Max samples per class (0 for all)')
    parser.add_argument('--crops_num', type=int, default=1, help='Number of crops (not used, kept for compatibility)')
    parser.add_argument('--output_path', type=str, required=True, help='Output path to save features (.pkl file)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

    opt = parser.parse_args()

    # Load DINOv2 model
    print("Loading DINOv2 model...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dinov2 = dinov2.cuda()
    dinov2.eval()
    print("Model loaded.")

    # Setup data augmentation
    data_transform = DataAugmentationDINO_conv(
        (0.9, 1),
        (0.05, 0.4),
        opt.crops_num,
        global_crops_size=224,
        local_crops_size=96,
    )

    # Create dataset
    print("Creating dataset...")
    dataset = RealFakeDataset(
        opt.real_path,
        opt.fake_path,
        opt.data_mode,
        opt.max_sample,
        "CLIP:ViT-L/14",
        trans=data_transform,
        path_mode=opt.path_mode
    )
    print(f"Dataset size: {len(dataset)}")

    # Extract features
    extractor = FeatureExtractor(dinov2, data_transform, opt.crops_num)
    extractor.extract_features_from_dataset(
        dataset, 
        opt.output_path,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers
    )
    
    print("Feature extraction completed!")

