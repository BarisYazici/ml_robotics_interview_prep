# Ray-Powered PyTorch Model for CIFAR-10

This example demonstrates how to use Ray ecosystem for distributed machine learning with PyTorch on the CIFAR-10 dataset.

## Overview

This implementation showcases two key Ray functionalities:

1. **Ray Core** - For efficient parallel data preprocessing
2. **Ray Tune** - For distributed hyperparameter optimization

The example uses a simple CNN architecture on the CIFAR-10 dataset as a proof of concept.

## Requirements

To run this example, you'll need the following packages:

```
torch>=1.12.0
torchvision>=0.13.0
ray>=1.0.0
matplotlib>=3.5.0
numpy>=1.20.0
tqdm>=4.62.0
```

Install these requirements using:

```bash
pip install -r requirements_ray.txt
```

## Usage

To run the full example showing all Ray functionality:

```bash
python ray_distributed_training.py
```

This will:
1. Initialize Ray
2. Demonstrate parallel data preprocessing with Ray Core
3. Perform hyperparameter tuning with Ray Tune (using ASHA scheduler)
4. Train a final model with the best hyperparameters found
5. Save the trained model to `best_cifar_model.pth`

## Code Structure

- `SimpleCNN`: A basic CNN architecture for CIFAR-10 classification
- `parallel_preprocess_data()`: Shows Ray Core for parallel processing
- `tune_cifar()`: Demonstrates hyperparameter tuning with Ray Tune
- `train_cifar()`: Training function used by Ray Tune
- `train_with_best_params()`: Uses the best hyperparameters to train a final model
- `main()`: Orchestrates the demonstration of all Ray functionalities

## Customization

You can customize the example:

- Adjust hyperparameter search spaces in `tune_cifar()`
- Modify the CNN architecture in `SimpleCNN`
- Change resource allocation in `resources_per_trial`
- Increase `num_samples` and `max_num_epochs` for more thorough tuning

## GPU Acceleration

To use GPUs (if available):

1. Set `gpus_per_trial` in the `tune_cifar()` function call
2. Adjust `resources_per_trial` to include GPU allocation

## Results

Results from Ray Tune will be saved to the `./ray_results` directory, including:
- Training metrics
- Hyperparameter configurations
- Checkpoints

## Additional Resources

- [Ray Documentation](https://docs.ray.io/)
- [Ray Tune Guide](https://docs.ray.io/en/latest/tune/index.html)
- [Ray Core Programming Guide](https://docs.ray.io/en/latest/ray-core/walkthrough.html) 