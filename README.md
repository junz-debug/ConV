<div align="center">

<h1>Detecting Generated Images by Fitting Natural Image Distributions</h1>
<h3>A Distribution-Based Approach for AI-Generated Image Detection</h3>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
    <figure>
        <br>
        <p><em>A distribution-based framework for detecting AI-generated images by modeling natural image distributions in feature space</em></p>
    </figure>
</div>

</div>

## Abstract

This repository contains the official implementation of our method for detecting AI-generated images by fitting natural image distributions. Our approach leverages the fundamental distributional differences between natural and synthetic images in the feature space, employing Normalizing Flow models to learn the distribution of natural images and identify generated images as out-of-distribution samples.

## Motivation

The rapid advancement of generative AI technologies has led to increasingly sophisticated deepfakes and AI-generated images, posing significant challenges to image authenticity verification. Traditional detection methods often rely on model-specific artifacts, limiting their generalization capability when encountering novel generative models.

<div align="center">
    <figure>
        <img src="framework.png" alt="Framework Architecture" width="800">
        <br>
        <p><em>Architecture of the distribution-based detection framework</em></p>
    </figure>
</div>

Current limitations in generated image detection include:

- **Limited Generalization**: Detectors trained on specific generative models struggle to transfer to novel architectures
- **Feature Representation Constraints**: Conventional feature extraction methods fail to capture the intrinsic differences between natural and generated images
- **Robustness Challenges**: Detection methods are sensitive to post-processing operations such as transformations and compression

### Our Approach

We propose a **distribution-based detection method** that distinguishes between real and AI-generated images by analyzing their distributional properties in the feature space.

- **Distribution Fitting**: Normalizing Flow models are employed to fit the feature distribution of natural images, enabling anomaly detection through likelihood estimation
- **DINOv2 Feature Extraction**: Powerful image feature representations are extracted using DINOv2 vision transformers
- **Transformation Consistency Analysis**: Detection capability is enhanced by analyzing feature consistency across different transformations
- **Training Data Requirements**: The method requires only natural images for training, eliminating the need for generated image samples

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA (recommended for GPU acceleration)

### Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies include:
- `torch>=1.9.0`
- `torchvision>=0.10.0`
- `numpy>=1.21.0`
- `scikit-learn>=0.24.0`
- `Pillow>=8.0.0`

### Development Setup

For development or customization:

```bash
# Clone the repository
git clone https://github.com/yourusername/ConV.git
cd ConV

# Create conda environment (optional)
conda create -n conv python=3.8
conda activate conv

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

#### Basic Detection (DINOv2 + Cosine Similarity)

The simplest approach compares DINOv2 feature similarities between original and augmented images:

```bash
python main.py \
  --real_path /path/to/real/images \
  --fake_path /path/to/fake/images \
  --max_sample 1000 \
  --batch_size 64 \
  --num_workers 2 \
  --crops_num 5 \
  --aggregation mean
```

**Parameters:**
- `--real_path` (required): Path to directory containing real images
- `--fake_path` (required): Path to directory containing generated images
- `--max_sample` (default: 1000): Maximum number of samples per class (0 for random sampling of 1000 images)
- `--batch_size` (default: 64): Batch size for processing
- `--num_workers` (default: 2): Number of data loading threads
- `--crops_num` (default: 5): Number of image crops for multi-view consistency analysis
- `--threshold` (default: 0): Classification threshold (0 for automatic threshold selection)
- `--aggregation` (default: 'mean'): Distance aggregation method (`mean`, `max`, or `min`)

#### F-ConV Method (Flow-based Convolutional)

Detection using Normalizing Flow models to fit natural image distributions. This method supports both training and inference modes.

**Inference Only (using pre-trained model):**

```bash
python F-ConV.py \
  --real_path /path/to/real/images \
  --fake_path /path/to/fake/images \
  --path_mode separate \
  --data_mode ours \
  --max_sample 0 \
  --batch_size 256 \
  --crops_num 1 \
  --model_path /path/to/model.pth \
  --load_model
```

**Training + Inference:**

```bash
python F-ConV.py \
  --real_path /path/to/test/real/images \
  --fake_path /path/to/test/fake/images \
  --train_real_path /path/to/train/real/images \
  --train_fake_path /path/to/train/fake/images \
  --path_mode separate \
  --data_mode ours \
  --max_sample 0 \
  --train_max_sample 0 \
  --batch_size 256 \
  --max_epoch 1 \
  --lr 1e-5 \
  --crops_num 1 \
  --model_path /path/to/save/model.pth
```

**Subdirectory Mode (for datasets with 0_real/1_fake structure):**

```bash
python F-ConV.py \
  --real_path /path/to/dataset/root \
  --path_mode subdirs \
  --data_mode ours \
  --max_sample 0 \
  --batch_size 256 \
  --crops_num 1 \
  --model_path /path/to/model.pth \
  --load_model
```

**F-ConV Parameters:**
- `--real_path` (required): Path to real images or root path for subdirs mode
- `--fake_path` (optional): Path to fake images (not needed for subdirs mode)
- `--data_mode` (default: 'ours'): Data mode (`wang2020` or `ours`)
- `--path_mode` (default: 'separate'): Path mode (`separate` for separate real/fake paths, `subdirs` for 0_real/1_fake structure)
- `--max_sample` (default: 0): Maximum samples per class for testing (0 for all)
- `--batch_size` (default: 256): Batch size
- `--num_workers` (default: 4): Number of data loading threads
- `--max_epoch` (default: 1): Number of training epochs
- `--lr` (default: 1e-5): Learning rate
- `--crops_num` (default: 1): Number of crops
- `--train_real_path` (optional): Training real image path
- `--train_fake_path` (optional): Training fake image path
- `--train_path_mode` (default: 'separate'): Training path mode
- `--train_max_sample` (default: 0): Maximum samples per class for training (0 for all)
- `--model_path` (optional): Path to save/load model weights
- `--load_model` (flag): Load model from model_path instead of training

#### Feature Extraction Mode

For large-scale datasets, features can be pre-extracted for efficient training:

**Extract Features:**

```bash
python extract_features.py \
  --real_path /path/to/real/images \
  --fake_path /path/to/fake/images \
  --path_mode separate \
  --data_mode ours \
  --max_sample 0 \
  --output_path features.pkl \
  --batch_size 64 \
  --num_workers 4
```

**Training with Pre-extracted Features:**

```bash
python F-ConV-feature.py \
  --real_path /path/to/test/real/images \
  --fake_path /path/to/test/fake/images \
  --train_feature_path features.pkl \
  --path_mode separate \
  --data_mode ours \
  --max_sample 0 \
  --batch_size 256 \
  --max_epoch 1 \
  --lr 1e-5 \
  --model_path /path/to/save/model.pth
```

**F-ConV-feature Additional Parameters:**
- `--train_feature_path`: Path to pre-extracted feature pickle file
- `--val_real_path` (optional): Real image path for validation during training
- `--val_fake_path` (optional): Fake image path for validation during training
- `--val_path_mode` (default: 'separate'): Validation path mode
- `--val_max_sample` (default: 1000): Maximum samples per class for validation
- `--val_interval` (default: 500): Validate every N batches
- `--margin` (default: 2000): Margin parameter for loss function
- `--temperature` (optional): Temperature parameter for sigmoid in testing

### Batch Evaluation

Evaluate on multiple datasets using the provided script:

```bash
bash run_all_datasets.sh
```

This script automatically iterates through configured dataset paths and aggregates results.

## Configuration

### Model Configuration (config.py)

Key configuration parameters:

```python
# Device settings
device = 'cuda'  # or 'cpu'

# Image dimensions
img_size = (448, 448)

# Network hyperparameters
n_scales = 3              # Number of scales for multi-scale feature extraction
clamp_alpha = 3           # Clamp parameter for Normalizing Flow
n_coupling_blocks = 2    # Number of coupling blocks in Flow model
fc_internal = 4096        # Hidden units in fully connected layers
n_feat = 256 * n_scales   # Feature dimensionality

# Training parameters
batch_size = 24
meta_epochs = 24
sub_epochs = 8
lr_init = 2e-4
```

### Data Format

The project supports multiple data organization formats:

1. **Separate Mode** (`--path_mode separate`): Separate directories for real and generated images
   ```
   dataset/
   ├── real/     # Real images
   └── fake/     # Generated images
   ```
   Usage: `--real_path /path/to/real --fake_path /path/to/fake --path_mode separate`

2. **Subdirectory Mode** (`--path_mode subdirs`): Single root directory with subdirectories containing `0_real` and `1_fake` folders
   ```
   dataset/
   ├── subset1/
   │   ├── 0_real/  # Real images
   │   └── 1_fake/  # Generated images
   ├── subset2/
   │   ├── 0_real/
   │   └── 1_fake/
   └── ...
   ```
   Usage: `--real_path /path/to/dataset/root --path_mode subdirs`

3. **Pickle Files**: Pickle files containing lists of image paths (for `--real_path` or `--fake_path`)
   ```python
   # File contains a list of image paths
   image_list = ['/path/to/image1.jpg', '/path/to/image2.jpg', ...]
   ```

4. **Supported Image Formats**: PNG, JPG, JPEG, BMP

## Evaluation Metrics

The following evaluation metrics are provided:

- **AUROC** (Area Under ROC Curve): Area under the receiver operating characteristic curve, measuring classification performance
- **AUPR** (Area Under Precision-Recall Curve): Area under the precision-recall curve
- **FPR@95%TPR**: False positive rate at 95% true positive rate
- **Accuracy**: Classification accuracy based on optimal threshold selection

Example output:
```
AUROC: 0.9523
AP: 0.9456
ACC: 0.9234
Best threshold: 0.013800
```

## Methodology

### Core Principle

Natural and AI-generated images exhibit distinct distributional properties in the feature space. Our method exploits this difference through:

1. **Feature Extraction**: Deep feature representations are extracted using DINOv2
2. **Distribution Modeling**: Normalizing Flow models are used to fit the feature distribution of natural images
3. **Anomaly Detection**: Generated images are identified as out-of-distribution samples with low likelihood values
4. **Transformation Consistency**: Real images maintain feature consistency across transformations, whereas generated images do not

### Technical Details

#### DINOv2 Feature Extraction

- Pre-trained DINOv2 ViT-L/14 model is employed
- Normalized feature vectors are extracted
- Multi-crop view feature extraction is supported

#### Normalizing Flow Model

- Reversible neural network based on Glow architecture
- Feature distribution is learned through coupling layers
- Likelihood is computed using Jacobian determinants

#### Loss Functions

The F-ConV method employs the following loss functions:

- **Shape Loss**: Distinguishes feature distributions between real and generated images
- **Consistency Loss**: Ensures feature consistency across transformations
- **Total Loss**: Combined loss for end-to-end training

## Experimental Results

### Supported Generative Models

The method has been evaluated on the following generative models:

- **Diffusion Models**: Stable Diffusion, ADM, LDM, DiT
- **GAN Models**: BigGAN, StyleGAN-XL, GigaGAN
- **Autoregressive Models**: Mask-GIT, RQ-Transformer
- **Others**: VQDM, SDv4, SDv5, Midjourney, etc.

### Performance

The method demonstrates superior detection performance across multiple datasets, particularly in cross-model generalization scenarios.

## Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{
  your2025detecting,
  title={Detecting Generated Images by Fitting Natural Image Distributions},
  author={Your Name and Co-authors},
  booktitle={Conference Name},
  year={2025},
  url={https://openreview.net/forum?id=27xTIAFbc6}
}
```

## Frequently Asked Questions

**Q1: How should training data be prepared?**  
A: For F-ConV method, only natural (real) images are required for training. Place real images in a directory and specify `--train_real_path`. The system will automatically learn the natural image distribution. For basic detection (`main.py`), no training is required.

**Q2: What image dimensions are supported?**  
A: Images are automatically processed to square format (center-cropped) and resized. The default DINOv2 model expects 224×224 input, but the code handles various input sizes by automatic resizing.

**Q3: What is the difference between `separate` and `subdirs` path modes?**  
A: `separate` mode requires two separate paths (`--real_path` and `--fake_path`). `subdirs` mode uses a single root directory where each subdirectory contains `0_real` and `1_fake` folders. Use `subdirs` mode for datasets organized with multiple subsets.

**Q4: How can detection accuracy be improved?**  
A: Consider the following:
- Increase the `crops_num` parameter (more views for consistency analysis)
- Adjust the `aggregation` method (mean/max/min) in basic detection
- Use the F-ConV method with proper training instead of the basic approach
- Increase the training data volume and training epochs
- Fine-tune hyperparameters (learning rate, margin, etc.)

**Q5: What is the detection speed?**  
A: With GPU acceleration, single image detection time is approximately 10-50ms (depending on model and image dimensions). Batch processing significantly improves efficiency. Feature extraction can be pre-computed for faster training iterations.

**Q6: Is GPU required?**  
A: GPU is strongly recommended for optimal performance. CPU mode is supported but significantly slower, especially for DINOv2 feature extraction and Flow model training.

**Q7: How do I use a pre-trained model?**  
A: For F-ConV, use `--load_model` flag with `--model_path` pointing to the saved model weights. The model will skip training and directly perform inference.

**Q8: What does `--max_sample 0` mean?**  
A: `--max_sample 0` means using all available images. For non-zero values, it limits the number of samples per class. In `main.py`, `0` specifically triggers random sampling of 1000 images.

## File Structure

```
ConV/
├── main.py                 # Basic detection method (DINOv2 + cosine similarity)
├── F-ConV.py              # F-ConV method (Flow-based)
├── F-ConV-feature.py      # F-ConV with pre-extracted features
├── model.py               # Normalizing Flow model definition
├── config.py              # Configuration file
├── utils.py               # Utility functions and loss functions
├── augmentation.py        # Data augmentation (basic method)
├── augmentations_fconv.py # Data augmentation (F-ConV method)
├── extract_features.py    # Feature extraction script
├── extract_features.sh    # Feature extraction shell script
├── run_all_datasets.sh   # Batch evaluation script
├── requirements.txt       # Dependency list
└── README.md             # This file
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions, technical support, or collaboration inquiries:

- Issues: [GitHub Issues](https://github.com/yourusername/ConV/issues)
- Paper: [OpenReview](https://openreview.net/forum?id=27xTIAFbc6)
