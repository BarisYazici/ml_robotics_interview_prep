import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm

# Create directory for saving visualizations
os.makedirs('visualizations_diffusion', exist_ok=True)

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

# Hyperparameters
img_size = 32  # CIFAR-10 image size
channels = 3   # RGB images
batch_size = 128
learning_rate = 2e-4  # Slightly increased for faster convergence
weight_decay = 1e-5
num_epochs = 10  # Reduced for quicker toy example
patch_size = 4  # Size of image patches
embed_dim = 128  # Embedding dimension (smaller for speed)
num_heads = 4   # Number of attention heads
num_layers = 3  # Reduced number of transformer layers for speed
mlp_dim = embed_dim * 4  # MLP hidden dimension

# Diffusion model parameters
timesteps = 50  # Reduced number of diffusion steps for the toy example
beta_start = 1e-4
beta_end = 0.02

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False
)

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# Linear Beta Schedule for the Diffusion Process
def linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end):
    """
    Linear beta schedule from beta_start to beta_end
    """
    return torch.linspace(beta_start, beta_end, timesteps)

# Get the pre-computed values for the forward diffusion process
def get_diffusion_params(beta_schedule):
    """
    Returns parameters needed for the diffusion model from the beta schedule
    """
    # The beta values control how much noise is added at each step
    betas = beta_schedule
    
    # The alpha values are used to compute the means in the forward diffusion
    alphas = 1. - betas
    
    # Cumulative product of alpha for computing closed-form means
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # sqrt of alphas_cumprod and 1-alphas_cumprod for computations
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
    }

# Set up the diffusion parameters
betas = linear_beta_schedule(timesteps)
diffusion_params = get_diffusion_params(betas)

# Move diffusion parameters to the same device as the model
for key in diffusion_params:
    diffusion_params[key] = diffusion_params[key].to(device)

# Function to add noise to images according to the forward diffusion process
def q_sample(x_start, t, diffusion_params, noise=None):
    """
    Forward diffusion process - adds noise to the input according to the given timestep.
    
    Args:
        x_start: starting clean image [B, C, H, W]
        t: timestep to add noise for [B,]
        diffusion_params: pre-computed diffusion values
        noise: optionally provide noise, otherwise it will be generated
    
    Returns:
        noisy image at the specified timestep
    """
    # Get the needed diffusion parameters
    sqrt_alphas_cumprod = diffusion_params['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod']
    
    # Generate noise if not provided
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # Get the scaling factors for the mean and noise
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    
    # Compute the noisy image according to the diffusion SDE
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

# Patch Embedding class for Transformer
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [batch_size, channels, img_size, img_size]
        batch_size, _, _, _ = x.shape
        x = self.proj(x)  # [batch_size, embed_dim, n_patches_h, n_patches_w]
        x = x.flatten(2)  # [batch_size, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [batch_size, n_patches, embed_dim]
        return x

# Multi-head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Define query, key, value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head attention
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Calculate weighted values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x = self.proj(x)
        x = self.dropout(x)
        return x

# MLP block
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, embed_dim, dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.attn(self.norm1(x))
        x = x + attn_output
        
        # MLP with residual connection
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        return x

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Fills the input Tensor with values drawn from a truncated
    normal distribution. The implementation is compatible with MPS backend.
    """
    # Ensure tensor is on CPU for initialization
    is_mps = tensor.device.type == "mps"
    if is_mps:
        tensor_cpu = tensor.cpu()
    else:
        tensor_cpu = tensor
    
    # Calculate values
    with torch.no_grad():
        mean = torch.tensor(mean, device="cpu")
        std = torch.tensor(std, device="cpu")
        a = torch.tensor(a, device="cpu")
        b = torch.tensor(b, device="cpu")
        
        normal = torch.distributions.Normal(mean, std)
        cdf_a = normal.cdf(a)
        cdf_b = normal.cdf(b)
        
        # Sample from uniform distribution and transform to truncated normal
        u = torch.rand_like(tensor_cpu)
        u = cdf_a + u * (cdf_b - cdf_a)
        
        # Apply inverse CDF
        tensor_cpu.copy_(normal.icdf(u))
        
        # Clamp to ensure values are within bounds
        tensor_cpu.clamp_(min=a, max=b)
    
    # Copy back to original device if needed
    if is_mps:
        tensor.copy_(tensor_cpu.to(tensor.device))
    
    return tensor

# Transformer-based U-Net for Diffusion Model
class DiffusionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, 
                 mlp_dim, num_layers, timesteps, dropout=0.0):
        super().__init__()
        
        # Time embedding - to provide timestep information to the model
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Image patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection - convert back to image space
        self.norm = nn.LayerNorm(embed_dim)
        
        # Final layer to reconstruct the image (noise prediction)
        # Instead of using Unflatten which causes dimension issues, handle reshaping manually
        self.patch_output = nn.Linear(embed_dim, patch_size * patch_size * channels)
        
        # Initialize weights
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.trunc_normal_(m.weight, std=0.02)
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
            
    def forward(self, x, t):
        # Embed timestep
        t = t.float().unsqueeze(1) / timesteps  # Normalize timestep to [0, 1]
        t_embed = self.time_embed(t).unsqueeze(1)  # [B, 1, embed_dim]
        
        # Image patch embedding
        batch_size = x.shape[0]
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Add time embedding to each patch
        x = x + t_embed
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)  # [B, n_patches, embed_dim]
        
        # Manual reshape approach to avoid dimension mismatch with unflatten
        x = self.patch_output(x)  # [B, n_patches, patch_size*patch_size*channels]
        
        # Reshape to form patches
        h = w = img_size // patch_size  # Number of patches along height and width
        # Reshape to [B, h, w, patch_size, patch_size, channels]
        x = x.reshape(batch_size, h, w, patch_size, patch_size, channels)
        
        # Permute and reshape to get final image
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(batch_size, channels, img_size, img_size)
        
        return x

# Create the model
model = DiffusionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=channels,
    embed_dim=embed_dim,
    num_heads=num_heads,
    mlp_dim=mlp_dim,
    num_layers=num_layers,
    timesteps=timesteps,
    dropout=0.1
).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

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
    alpha_cumprod_prev = diffusion_params['alphas_cumprod'][t_index-1] if t_index > 0 else torch.tensor(1.0, device=device)
    
    # Predict the noise
    predicted_noise = model(x, t)
    
    # Calculate the mean for the reverse process
    sqrt_alpha_cumprod = sqrt_alphas_cumprod[t_index]
    sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alphas_cumprod[t_index]
    
    # Calculate denoised x_0 estimate
    x_0_pred = (x - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod
    x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
    
    # Calculate the mean of the posterior q(x_{t-1} | x_t, x_0)
    mean = sqrt_alpha_cumprod_prev * (x - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod
    variance = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * beta
    std = torch.sqrt(variance)
    
    # If t == 0, return the mean (no more noise to add)
    if t_index == 0:
        return mean
    
    # Otherwise, add some noise and return
    epsilon = torch.randn_like(x)
    return mean + std * epsilon

@torch.no_grad()
def p_sample_loop(model, shape, diffusion_params, n_samples=4):
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

# Function to visualize generated samples
def visualize_samples(samples, epoch=None):
    """
    Visualize generated samples from the diffusion model.
    """
    # Move samples to CPU and convert to numpy
    samples = samples.detach().cpu().numpy()
    samples = np.transpose(samples, (0, 2, 3, 1))  # [B, H, W, C]
    
    # Create a grid plot
    plt.figure(figsize=(10, 10))
    for i in range(min(samples.shape[0], 16)):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i])
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save with epoch information if provided
    if epoch is not None:
        plt.savefig(f'visualizations_diffusion/samples_epoch_{epoch}.png')
    else:
        plt.savefig('visualizations_diffusion/final_samples.png')
    
    plt.close()
    print(f"Saved generated samples visualization")

# Function to visualize the forward diffusion process
def visualize_forward_diffusion(loader):
    """
    Visualize how an image gets progressively noisier in the forward diffusion process.
    """
    # Get a batch from the loader
    images, _ = next(iter(loader))
    img = images[0:1].to(device)  # Take the first image
    
    # Create timesteps (adjusted for the new timesteps=50)
    vis_timesteps = [0, 5, 10, 20, 35, 49]  # Timesteps to visualize
    
    # Visualize forward process
    plt.figure(figsize=(15, 3))
    
    for i, t_step in enumerate(vis_timesteps):
        t = torch.tensor([t_step], device=device)
        noisy_img, _ = q_sample(img, t, diffusion_params)
        
        # Display the image
        plt.subplot(1, len(vis_timesteps), i+1)
        noisy_img_np = noisy_img[0].detach().cpu().numpy().transpose(1, 2, 0)
        noisy_img_np = (noisy_img_np + 1) / 2  # Scale from [-1, 1] to [0, 1]
        noisy_img_np = np.clip(noisy_img_np, 0, 1)
        plt.imshow(noisy_img_np)
        plt.title(f"t={t_step}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations_diffusion/forward_diffusion.png')
    plt.close()
    print("Saved forward diffusion visualization")

# Training function
def train_diffusion_model():
    model.train()
    train_losses = []
    
    # Show the forward diffusion process on an example before training
    visualize_forward_diffusion(test_loader)
    
    # Simple learning rate warmup
    def get_lr_scale(epoch, warmup_epochs=2):
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        return 1.0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # Apply learning rate warmup
        if epoch < 3:
            lr_scale = get_lr_scale(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_scale
            print(f"Learning rate: {learning_rate * lr_scale:.6f}")
        
        # Progress bar for the training loop
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (images, _) in progress_bar:
            # Move images to device
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            
            # Add noise according to the forward process
            noisy_images, noise = q_sample(images, t, diffusion_params)
            
            # Predict the noise
            predicted_noise = model(noisy_images, t)
            
            # Calculate loss (the model predicts the noise that was added)
            loss = criterion(predicted_noise, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
        
        # Generate and visualize samples every 5 epochs or at the last epoch
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            model.eval()
            # Generate samples - use only 2 samples during training for speed
            samples = p_sample_loop(model, (2, channels, img_size, img_size), diffusion_params, n_samples=2)
            visualize_samples(samples, epoch=epoch+1)
            model.train()
    
    # Save the model
    torch.save(model.state_dict(), 'diffusion_transformer_model.pth')
    print("Model saved to diffusion_transformer_model.pth")
    
    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('visualizations_diffusion/training_loss.png')
    plt.close()
    print("Saved training loss plot")
    
    return train_losses

# Function to generate final samples
def generate_final_samples(n_samples=16):
    print("Generating final samples...")
    model.eval()
    # Generate multiple samples
    samples = p_sample_loop(model, (n_samples, channels, img_size, img_size), diffusion_params, n_samples=n_samples)
    visualize_samples(samples)
    print(f"Generated {n_samples} samples")

# Main function
if __name__ == '__main__':
    print("Setting up diffusion model training...")
    
    # Train the diffusion model
    losses = train_diffusion_model()
    
    # Generate final samples
    generate_final_samples(16)
    
    print("Diffusion model training and sampling complete!") 