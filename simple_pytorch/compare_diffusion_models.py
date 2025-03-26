import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Import the model definitions
from train_diffusion_transformer import DiffusionTransformer, linear_beta_schedule, get_diffusion_params
from pretrained_diffusion_transformer import PretrainedDiffusionTransformer

# Create directory for comparison visualizations
os.makedirs('visualizations_comparison', exist_ok=True)

# Device configuration
def get_device():
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps"), "Apple Silicon GPU (MPS)"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "CUDA GPU"
    else:
        return torch.device("cpu"), "CPU"

device, device_name = get_device()
print(f"Using {device_name}")

# Parameters
img_size = 32
channels = 3
patch_size = 4
n_samples = 8

# Original model parameters
orig_timesteps = 50
orig_embed_dim = 128
orig_num_heads = 4
orig_num_layers = 3
orig_mlp_dim = orig_embed_dim * 4
orig_beta_start = 1e-4
orig_beta_end = 0.02

# Pretrained model parameters
pretrained_timesteps = 1000
pretrained_embed_dim = 256
pretrained_num_heads = 4
pretrained_num_layers = 4
pretrained_mlp_dim = pretrained_embed_dim * 4
pretrained_beta_start = 1e-4
pretrained_beta_end = 0.02

# Load the original model
def load_original_model():
    # Set up diffusion parameters
    betas = linear_beta_schedule(orig_timesteps, beta_start=orig_beta_start, beta_end=orig_beta_end)
    diffusion_params = get_diffusion_params(betas)
    
    # Move diffusion parameters to the device
    for key in diffusion_params:
        diffusion_params[key] = diffusion_params[key].to(device)
    
    # Create the model
    model = DiffusionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=channels,
        embed_dim=orig_embed_dim,
        num_heads=orig_num_heads,
        mlp_dim=orig_mlp_dim,
        num_layers=orig_num_layers,
        timesteps=orig_timesteps,
        dropout=0.1
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load('diffusion_transformer_model.pth', map_location=device))
    model.eval()
    
    return model, diffusion_params

# Load the pretrained model
def load_pretrained_model():
    # Set up diffusion parameters
    betas = linear_beta_schedule(pretrained_timesteps, beta_start=pretrained_beta_start, beta_end=pretrained_beta_end)
    diffusion_params = get_diffusion_params(betas)
    
    # Move diffusion parameters to the device
    for key in diffusion_params:
        diffusion_params[key] = diffusion_params[key].to(device)
    
    # Create the model
    model = PretrainedDiffusionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=channels,
        embed_dim=pretrained_embed_dim,
        num_heads=pretrained_num_heads,
        mlp_dim=pretrained_mlp_dim,
        num_layers=pretrained_num_layers,
        timesteps=pretrained_timesteps,
        dropout=0.1
    ).to(device)
    
    # Check if the pretrained model exists
    if os.path.exists('pretrained_diffusion_model.pth'):
        model.load_state_dict(torch.load('pretrained_diffusion_model.pth', map_location=device))
    else:
        print("Pretrained model weights not found. Using random initialization.")
    
    model.eval()
    
    return model, diffusion_params

# Function to generate samples using the diffusion model
@torch.no_grad()
def p_sample(model, x, t, t_index, diffusion_params):
    """
    Sample from the model at timestep t.
    """
    betas = diffusion_params['betas']
    sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod']
    sqrt_alphas_cumprod = diffusion_params['sqrt_alphas_cumprod']
    
    # Extract the current beta, alpha values
    beta = betas[t_index]
    alpha = 1 - beta
    alpha_cumprod = diffusion_params['alphas_cumprod'][t_index]
    alpha_cumprod_prev = diffusion_params['alphas_cumprod_prev'][t_index] if 'alphas_cumprod_prev' in diffusion_params else (
        diffusion_params['alphas_cumprod'][t_index-1] if t_index > 0 else torch.tensor(1.0, device=device)
    )
    
    # Predict the noise
    predicted_noise = model(x, t)
    
    # Calculate the mean for the reverse process
    sqrt_alpha_cumprod = sqrt_alphas_cumprod[t_index]
    sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alphas_cumprod[t_index]
    
    # Calculate denoised x_0 estimate
    x_0_pred = (x - sqrt_one_minus_alpha_cumprod.reshape(-1, 1, 1, 1) * predicted_noise) / sqrt_alpha_cumprod.reshape(-1, 1, 1, 1)
    x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
    
    # Calculate the square root of alpha_cumprod_prev for posterior mean
    sqrt_alpha_cumprod_prev = torch.sqrt(alpha_cumprod_prev)
    
    # Calculate the mean of the posterior q(x_{t-1} | x_t, x_0)
    mean = sqrt_alpha_cumprod_prev.reshape(-1, 1, 1, 1) * (
        x - sqrt_one_minus_alpha_cumprod.reshape(-1, 1, 1, 1) * predicted_noise
    ) / sqrt_alpha_cumprod.reshape(-1, 1, 1, 1)
    
    variance = ((1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * beta).reshape(-1, 1, 1, 1)
    std = torch.sqrt(variance)
    
    # If t == 0, return the mean (no more noise to add)
    if t_index == 0:
        return mean
    
    # Otherwise, add some noise and return
    epsilon = torch.randn_like(x)
    return mean + std * epsilon

@torch.no_grad()
def p_sample_loop(model, shape, diffusion_params, timesteps, n_samples=4):
    """
    Generate samples from the model starting with pure noise.
    """
    b = n_samples
    # Start from pure noise
    x = torch.randn(shape).to(device)
    
    # Progressively denoise
    for i in tqdm(reversed(range(timesteps)), desc='Sampling', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, i, diffusion_params)
    
    # Return generated samples, scaled back to [0, 1] for visualization
    return (x.clamp(-1, 1) + 1) / 2

# Generate samples from both models and compare
def generate_comparison_samples():
    # Load both models
    original_model, original_diffusion_params = load_original_model()
    pretrained_model, pretrained_diffusion_params = load_pretrained_model()
    
    # Set the same seed for both models to ensure fair comparison
    torch.manual_seed(42)
    
    # Generate samples from the original model
    print("Generating samples from the original model...")
    original_samples = p_sample_loop(
        original_model, 
        (n_samples, channels, img_size, img_size), 
        original_diffusion_params,
        orig_timesteps,
        n_samples=n_samples
    )
    
    # Generate samples from the pretrained model
    print("Generating samples from the pretrained model...")
    torch.manual_seed(42)  # Reset seed for fair comparison
    pretrained_samples = p_sample_loop(
        pretrained_model, 
        (n_samples, channels, img_size, img_size), 
        pretrained_diffusion_params,
        pretrained_timesteps,
        n_samples=n_samples
    )
    
    # Visualize side by side
    compare_visualize_samples(original_samples, pretrained_samples)

# Function to visualize samples side by side
def compare_visualize_samples(original_samples, pretrained_samples):
    # Convert to numpy arrays
    original_samples = original_samples.detach().cpu().numpy()
    original_samples = np.transpose(original_samples, (0, 2, 3, 1))  # [B, H, W, C]
    
    pretrained_samples = pretrained_samples.detach().cpu().numpy()
    pretrained_samples = np.transpose(pretrained_samples, (0, 2, 3, 1))  # [B, H, W, C]
    
    # Create side-by-side comparison
    plt.figure(figsize=(20, 10))
    
    for i in range(min(original_samples.shape[0], 8)):
        # Original model samples
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(original_samples[i])
        plt.title(f"Original Model")
        plt.axis('off')
        
        # Pretrained model samples
        plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(pretrained_samples[i])
        plt.title(f"Pretrained Model")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations_comparison/model_comparison.png')
    plt.close()
    print("Saved comparison visualization to visualizations_comparison/model_comparison.png")

# Main function
if __name__ == "__main__":
    print("Comparing original diffusion model with pretrained diffusion model...")
    generate_comparison_samples()
    print("Comparison complete!") 