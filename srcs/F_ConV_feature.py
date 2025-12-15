
import argparse
import os
import torch
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import Dataset
from PIL import Image
import pickle
from copy import deepcopy
import sklearn.metrics as sk
from tqdm import tqdm
from torch.cuda.amp import autocast
import random

from model import flow_model
from utils import get_loss_outlier_genimage, get_loss_outlier_progan
from augmentations_fconv import DataAugmentationDINO_conv_test


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp", "PNG"]):
    """Recursively read image files."""
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts) and (must_contain in os.path.join(r, file)):
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
    Uses actual labels to find the best threshold that maximizes accuracy.
    """
    N = y_true.shape[0]
    

    unique_preds = np.unique(y_pred)
    # Add midpoints between consecutive unique values for better threshold search
    if len(unique_preds) > 1:
        midpoints = (unique_preds[:-1] + unique_preds[1:]) / 2
        threshold_candidates = np.concatenate([unique_preds, midpoints])
    else:
        threshold_candidates = unique_preds

    best_acc = 0 
    best_thres = threshold_candidates[0] if len(threshold_candidates) > 0 else 0.5
    
    for thres in threshold_candidates:
        # Convert predictions to binary based on threshold
        pred_binary = (y_pred >= thres).astype(int)
        
        # Calculate accuracy
        acc = (pred_binary == y_true).sum() / N  

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


class FeatureDataset(Dataset):
    """Dataset class for loading pre-extracted features from pickle files.
    
    Each image has one source feature and one transformed feature.
    Supports loading from multiple feature files and concatenating them.
    """
    
    def __init__(self, feature_paths):
        """
        Args:
            feature_paths: Single path (str) or list of paths to pickle files containing features
                         Expected format: {
                             'features_source': np.array,  # shape: (N, feature_dim)
                             'features_trans': np.array,   # shape: (N, feature_dim)
                             'labels': np.array            # shape: (N,)
                         }
        """
        # Handle both single path and multiple paths
        if isinstance(feature_paths, str):
            feature_paths = [feature_paths]
        
        all_features_source = []
        all_features_trans = []
        all_labels = []
        
        for feature_path in feature_paths:
            print(f"Loading features from {feature_path}...")
            with open(feature_path, 'rb') as f:
                data = pickle.load(f)
            
            features_source = data['features_source']  # (N, feature_dim)
            features_trans = data['features_trans']    # (N, feature_dim)
            labels = data['labels']                     # (N,)
            
            # Handle both old format (N, crops_num, feature_dim) and new format (N, feature_dim)
            if len(features_source.shape) == 3:
                # Old format: (N, crops_num, feature_dim) -> use first crop only
                print("Warning: Detected old format (N, crops_num, feature_dim), using first crop only")
                features_source = features_source[:, 0, :]  # (N, feature_dim)
                features_trans = features_trans[:, 0, :]    # (N, feature_dim)
            
            all_features_source.append(features_source)
            all_features_trans.append(features_trans)
            all_labels.append(labels)
            
            print(f"  Loaded {len(labels)} samples from {feature_path}")
        
        # Concatenate all features
        self.features_source = np.concatenate(all_features_source, axis=0)  # (Total_N, feature_dim)
        self.features_trans = np.concatenate(all_features_trans, axis=0)    # (Total_N, feature_dim)
        self.labels = np.concatenate(all_labels, axis=0)                     # (Total_N,)
        
        # Convert to torch tensors
        self.features_source = torch.from_numpy(self.features_source).float()
        self.features_trans = torch.from_numpy(self.features_trans).float()
        self.labels = torch.from_numpy(self.labels).long()
        
        print(f"\nTotal loaded: {len(self.labels)} samples")
        print(f"  Source features shape: {self.features_source.shape}")
        print(f"  Trans features shape: {self.features_trans.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Return features in the same format as RealFakeDataset
        # Format: {"source": [feature], "trans": [feature]}, label
        # Note: We wrap in a list to maintain compatibility with code that expects crops
        return {"source": [self.features_source[idx]], "trans": [self.features_trans[idx]]}, self.labels[idx]


def validate_react(dinov2, flow, loader, temperature=None):
    """Validate the model using original images.
    
    Args:
        dinov2: DINOv2 model
        flow: Flow model
        loader: Data loader
        temperature: Temperature parameter for sigmoid (distance/temperature). If None, uses distance directly.
    """
    dinov2.cuda()
    dinov2.eval()
    flow.cuda()
    flow.eval()

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

            with autocast():
                z1 = flow(dinov2(img["source"][0].half().cuda()))
                z2 = flow(dinov2(img["trans"][0].half().cuda()))
                distance = torch.mean(z1 ** 2, dim=1)  + torch.mean(z2 ** 2, dim=1)
                # Use temperature parameter if provided, otherwise use distance directly
                if temperature is not None:
                    prob = torch.sigmoid(distance / temperature)
                else:
                    prob = torch.sigmoid(distance)
                cosine_sim = calculate_cosine_similarity(z1, z2)
                original_cosine_similarity = prob - cosine_sim * 0.5
                y_prediction.extend( original_cosine_similarity.detach().cpu().numpy())
            
            # consistent = calculate_cosine_similarity(z1, z2) * 0.1
            # shape = 0
            # original_cosine_similarity = shape + consistent
            
            original_cosine_similarity = original_cosine_similarity.detach().cpu().numpy()
            y_labels.extend(labels.flatten().tolist())
            y_pred.append(original_cosine_similarity)
            
            for i in range(labels.shape[0]):
                if labels[i] == 0:  # real
                    score_in.append(original_cosine_similarity[i])
                else:  # fake
                    score_out.append(original_cosine_similarity[i])
    
    y_labels, y_prediction = np.array(y_labels), np.array(y_prediction)
    print(y_labels[0:20])
    print(y_prediction[0:20])

    print(distance[0:20])

    print(prob[0:20])
    print(cosine_sim[0:20])

    best_thres = find_best_threshold(y_labels,  y_prediction)
    r_acc1, f_acc1, acc1 = calculate_acc(y_labels, y_prediction, best_thres)


    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    ap = average_precision_score(y_true, y_pred)
    auroc, aupr, fpr = get_measures(score_out, score_in)
    
    print("AUROC: " + str(auroc))
    print("AP: " + str(ap))
    print("ACC: " + str(acc1))
    
    return acc1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='F-ConV: Flow-based Fake Image Detection (Training with Pre-extracted Features, Testing with Original Images)')
    parser.add_argument('--real_path', type=str, required=True, help='Path to real images (or root path for subdirs mode)')
    parser.add_argument('--fake_path', type=str, default=None, help='Path to fake images (not needed for subdirs mode)')
    parser.add_argument('--data_mode', type=str, default='ours', choices=['wang2020', 'ours'])
    parser.add_argument('--path_mode', type=str, default='separate', choices=['separate', 'subdirs'],
                        help='Path mode: separate (real_path and fake_path) or subdirs (single path with 0_real/1_fake subdirs)')
    parser.add_argument('--max_sample', type=int, default=0, help='Max samples per class (0 for all)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--max_epoch', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--crops_num', type=int, default=1, help='Number of crops')
    parser.add_argument('--train_feature_path', type=str, default=None, 
                        help='Path(s) to training feature pickle file(s). Multiple paths can be separated by commas (e.g., "path1.pkl,path2.pkl")')
    parser.add_argument('--model_path', type=str, default=None, help='Path to save/load model weights')
    parser.add_argument('--load_model', action='store_true', help='Load model from model_path instead of training')
    parser.add_argument('--val_real_path', type=str, default=None, help='Path to real images for validation during training (e.g., Imagenet256-BigGAN)')
    parser.add_argument('--val_fake_path', type=str, default=None, help='Path to fake images for validation during training (e.g., Imagenet256-BigGAN)')
    parser.add_argument('--val_path_mode', type=str, default='separate', choices=['separate', 'subdirs'],
                        help='Path mode for validation: separate or subdirs')
    parser.add_argument('--val_max_sample', type=int, default=1000, help='Max samples per class for validation during training (default: 1000)')
    parser.add_argument('--val_interval', type=int, default=500, help='Validate every N batches (default: 500)')
    parser.add_argument('--margin', type=float, default=2000, help='Margin parameter for loss function (default: 2000)')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature parameter for sigmoid in testing (distance/temperature). If None, uses distance directly (default: None)')

    opt = parser.parse_args()

    # Load DINOv2 model (needed for testing with original images)
    print("Loading DINOv2 model...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dinov2 = dinov2.cuda()
    print("Model loaded.")

    # Initialize flow model
    flow = flow_model()
    flow = flow.cuda()
    total_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print(f"Flow model trainable parameters: {total_params}")
    
    # Load model if specified
    if opt.load_model and opt.model_path and os.path.exists(opt.model_path):
        print(f"Loading model from {opt.model_path}...")
        flow.load_state_dict(torch.load(opt.model_path))
        print("Model loaded.")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(flow.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.max_epoch)

    # Setup data augmentation for testing
    data_transform_test = DataAugmentationDINO_conv_test(
        (0.9, 1),
        (0.05, 0.4),
        opt.crops_num,
        global_crops_size=224,
        local_crops_size=96,
    )

    # Training
    if opt.train_feature_path:
        print("Setting up training dataset...")
        # Parse multiple feature paths (comma-separated)
        feature_paths = [path.strip() for path in opt.train_feature_path.split(',')]
        print(f"Loading features from {len(feature_paths)} file(s): {feature_paths}")
        train_dataset = FeatureDataset(feature_paths)
        print(f"Training dataset size: {len(train_dataset)}")

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=opt.batch_size, 
            shuffle=True, 
            num_workers=opt.num_workers
        )

        max_acc = 0
        
        # Setup validation dataset for periodic testing during training
        val_loader = None
        if opt.val_real_path and opt.val_fake_path:
            print(f"\nSetting up validation dataset (Imagenet256-BigGAN) for periodic testing...")
            val_dataset = RealFakeDataset(
                opt.val_real_path,
                opt.val_fake_path,
                opt.data_mode,
                opt.val_max_sample,  # Limit to 1000 samples per class
                "CLIP:ViT-L/14",
                trans=data_transform_test,
                path_mode=opt.val_path_mode
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.num_workers
            )
            print(f"Validation dataset size: {len(val_dataset)} (max {opt.val_max_sample} per class)")
        
        # Setup test dataset (used after each epoch)
        print(f"\nSetting up test dataset...")
        print(f"Creating test dataset with real_path={opt.real_path}, fake_path={opt.fake_path}")
        test_dataset = RealFakeDataset(
            opt.real_path,
            opt.fake_path,
            opt.data_mode,
            opt.max_sample,
            "CLIP:ViT-L/14",
            trans=data_transform_test,
            path_mode=opt.path_mode
        )
        print(f"Test dataset created, size: {len(test_dataset)}")
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=opt.batch_size, 
            shuffle=True, 
            num_workers=opt.num_workers
        )
        print(f"Test DataLoader created")
        
        for epoch in range(opt.max_epoch):
            print(f"\nEpoch {epoch+1}/{opt.max_epoch}")
            flow.train()
            for idx, (features, class_l) in enumerate(tqdm(train_loader)):
                # print(idx)
                class_l = class_l.cuda()
                optimizer.zero_grad()
                
                with autocast():
                    # features["source"][0] and features["trans"][0] are already DINOv2 features
                    feature1 = flow(features["source"][0].half().cuda())
                    # Calculate jacobian immediately after feature1 forward pass
                    jac1 = flow.nf.jacobian(run_forward=False)
                    
                    feature2 = flow(features["trans"][0].half().cuda())
                    jac2 = flow.nf.jacobian(run_forward=False)

                    # Calculate jacobian immediately after feature2 forward pass
                    
                    shape_loss, consistent_loss, loss = get_loss_outlier_progan(
                        feature1, feature2, 
                        jac1,
                        jac2, 
                        class_l, 
                        margin=opt.margin
                    )
                # print(idx)

                if idx % 500 == 0:
                    
                    print(f"Shape loss: {shape_loss:.4f}, Consistent loss: {consistent_loss:.4f}, Total loss: {loss:.4f}")

                #     acc = validate_react(dinov2, flow, test_loader)
                #     print(f"\nEpoch {epoch+1} Test Accuracy: {acc:.4f}")
                #     if acc > max_acc:
                #         max_acc = acc
                #         print(f"New best test accuracy: {max_acc:.4f}")
                #         # Save best model
                #         if opt.model_path:
                #             torch.save(flow.state_dict(), opt.model_path)
                #             print(f"Best model saved to {opt.model_path}")

                loss.backward()
                optimizer.step()
                

            scheduler.step()
            
            # Test after each epoch
            print("\n" + "="*50)
            print(f"Testing after Epoch {epoch+1}/{opt.max_epoch}...")
            print("="*50)
            try:
                acc = validate_react(dinov2, flow, test_loader, temperature=opt.temperature)
                print(f"\nEpoch {epoch+1} Test Accuracy: {acc:.4f}")
                if acc > max_acc:
                    max_acc = acc
                    print(f"New best test accuracy: {max_acc:.4f}")
                    # Save best model
                    if opt.model_path:
                        torch.save(flow.state_dict(), opt.model_path)
                        print(f"Best model saved to {opt.model_path}")
            except Exception as e:
                print(f"Error during testing: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("No training feature path provided, skipping training...")
        
        # Testing when only loading model (no training)
        print("\n" + "="*50)
        print("Testing on test set...")
        print("="*50)
        try:
            print(f"Creating test dataset with real_path={opt.real_path}, fake_path={opt.fake_path}")
            test_dataset = RealFakeDataset(
                opt.real_path,
                opt.fake_path,
                opt.data_mode,
                opt.max_sample,
                "CLIP:ViT-L/14",
                trans=data_transform_test,
                path_mode=opt.path_mode
            )
            print(f"Test dataset created, size: {len(test_dataset)}")
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset, 
                batch_size=opt.batch_size, 
                shuffle=False, 
                num_workers=opt.num_workers
            )
            print(f"Test DataLoader created")
            
            acc = validate_react(dinov2, flow, test_loader, temperature=opt.temperature)
            print(f"\nFinal Test Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()
            raise

