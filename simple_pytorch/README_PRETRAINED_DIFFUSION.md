# Pretrained Diffusion Transformer for CIFAR-10

This project implements a diffusion transformer model that leverages a pretrained ResNet backbone to generate high-quality images from the CIFAR-10 dataset.

## Overview

The architecture combines the strengths of:
- **Pretrained ResNet-18** backbone for feature extraction (with frozen weights)
- **Vision Transformer (ViT)** architecture for modeling dependencies between patches
- **Diffusion process** for high-quality image generation
- **Adaptable architecture** that automatically configures for different feature map sizes

This model significantly improves image quality compared to training a diffusion model from scratch.

## Key Features

1. **Pretrained Feature Extraction**: Leverages knowledge from ImageNet pretraining to extract meaningful features from images.
2. **Frozen Backbone**: The pretrained layers remain frozen to preserve the knowledge while allowing fine-tuning of other components.
3. **Auto-configuration**: The model dynamically adapts its positional embeddings and decoder layers based on the actual output shape of the backbone.
4. **Class Token**: Uses a class token similar to ViT to improve conditioning and overall generation quality.
5. **Increased Timesteps**: Uses 1000 timesteps for the diffusion process (compared to 50 in the original model) for better quality.

## How to Use

### Training

To train the model from scratch:

```bash
python pretrained_diffusion_transformer.py
```

The training process:
1. Downloads CIFAR-10 if not already present
2. Downloads pretrained ResNet-18 weights
3. Configures the model based on the actual feature map size
4. Trains for 10 epochs by default
5. Saves sample images every 2 epochs
6. Saves the final model to `pretrained_diffusion_model.pth`

### Comparing with Original Model

You can compare the results of this pretrained diffusion model with the original diffusion transformer:

```bash
python compare_diffusion_models.py
```

This will:
1. Load both models (original and pretrained)
2. Generate samples from each model using the same random seed
3. Create a side-by-side comparison visualization

## Model Architecture Details

### Backbone

The pretrained backbone is a ResNet-18 with:
- First convolutional layer modified for CIFAR-10 image sizes
- Initial layers (layer1 and layer2) with frozen weights
- An adapter layer to match the embedding dimension

### Transformer

The transformer part consists of:
- Dynamic positional embeddings based on the feature map size
- A class token similar to ViT for improved conditioning
- Multi-head self-attention layers
- Normalization and feed-forward layers

### Diffusion Process

The diffusion process uses:
- 1000 timesteps for fine-grained denoising
- Sinusoidal position embeddings for time steps
- Linear beta schedule from 1e-4 to 0.02
- MSE loss for noise prediction

## Hyperparameters

- `img_size`: 32 (CIFAR-10 image size)
- `channels`: 3 (RGB images)
- `batch_size`: 128
- `learning_rate`: 1e-4 (lower for fine-tuning)
- `embed_dim`: 256 (increased from original 128)
- `num_heads`: 4
- `num_layers`: 4 (increased from original 3)
- `timesteps`: 1000 (increased from original 50)

## Results

The pretrained model produces noticeably better image quality compared to the original model:
- Improved structure and coherence in generated images
- Better color and texture reproduction
- More recognizable objects
- Less noise and artifacts

Check the generated comparison visualizations in the `visualizations_comparison` directory after running both models.

## Extending the Model

To adapt this model for other datasets:

1. Modify the image size and channels in the hyperparameters
2. Adjust the learning rate and training schedule
3. Consider using different pretrained backbones for different domains

For conditional generation, add class labels or other conditioning signals to the class token embeddings.

## Requirements

- PyTorch >= 1.12.0
- torchvision
- matplotlib
- numpy
- tqdm 