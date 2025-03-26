import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Create directory for saving visualizations
os.makedirs('visualizations_pretrained_diffusion', exist_ok=True)

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
learning_rate = 1e-4  # Lower learning rate for fine-tuning
weight_decay = 1e-5
num_epochs = 10
patch_size = 4
embed_dim = 256  # Increased embedding dimension
num_heads = 4
num_layers = 4  # More transformer layers
mlp_dim = embed_dim * 4

# Diffusion model parameters
timesteps = 1000  # Increased timesteps for better quality
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
    return torch.linspace(beta_start, beta_end, timesteps)

# Get the pre-computed values for the forward diffusion process
def get_diffusion_params(beta_schedule):
    betas = beta_schedule
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    # Terms needed for q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'posterior_variance': posterior_variance,
    }

# Set up the diffusion parameters
betas = linear_beta_schedule(timesteps)
diffusion_params = get_diffusion_params(betas)

# Move diffusion parameters to the correct device
for key in diffusion_params:
    diffusion_params[key] = diffusion_params[key].to(device)

# Function to add noise to images according to the forward diffusion process
def q_sample(x_start, t, diffusion_params, noise=None):
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

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, timesteps):
        super().__init__()
        self.embed_dim = embed_dim
        self.timesteps = timesteps
        
        # Create positional embedding lookup table
        pe = torch.zeros(timesteps, embed_dim)
        position = torch.arange(0, timesteps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, t):
        return self.pe[t]

# Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [batch_size, channels, img_size, img_size]
        x = self.proj(x)  # [batch_size, embed_dim, n_patches_h, n_patches_w]
        x = x.flatten(2)  # [batch_size, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [batch_size, n_patches, embed_dim]
        return x

# Modified pretrained ResNet backbone for the diffusion model
class PretrainedBackbone(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Use a pretrained ResNet18 as the backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify the first layer to accept 3-channel CIFAR images
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Take the remaining layers from the pretrained model
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        
        # Adapter layer to match the embedding dimension
        self.adapter = nn.Conv2d(128, embed_dim, kernel_size=1)
        
        # Freeze the pretrained layers
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.adapter(x)
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

# Pretrained Diffusion Transformer
class PretrainedDiffusionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, 
                 mlp_dim, num_layers, timesteps, dropout=0.1):
        super().__init__()
        
        # Time embedding
        self.time_embed = PositionalEmbedding(embed_dim, timesteps)
        
        # Pretrained feature extractor
        self.backbone = PretrainedBackbone(embed_dim)
        
        # Estimate feature map size (will be corrected in first forward pass)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.register_buffer('feature_size_detected', torch.tensor(False))
        
        # Initialize dummy positional embedding that will be replaced in the first forward pass
        self.register_parameter('pos_embed', nn.Parameter(torch.zeros(1, 1, embed_dim)))
        
        # Add a class token similar to ViT, to help with condition learning
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Dropout after position embedding
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output normalization and projection
        self.norm = nn.LayerNorm(embed_dim)
        
        # The output_conv will be initialized in the first forward pass
        # when we know the actual feature map size
        self.output_conv = None
        
    def _init_pos_embed(self, h, w, device):
        """Initialize the positional embedding with the correct shape."""
        n_patches = h * w
        # Create a new parameter with the correct shape
        new_pos_embed = nn.Parameter(torch.zeros(1, n_patches, self.embed_dim, device=device))
        nn.init.trunc_normal_(new_pos_embed, std=0.02)
        
        # Replace the old parameter
        del self._parameters['pos_embed']
        self.register_parameter('pos_embed', new_pos_embed)
        
        print(f"Created positional embedding with shape {new_pos_embed.shape} for feature map size {h}x{w}")
        
        # Calculate how many upsampling layers we need to get back to the original image size
        feature_size = h  # Assuming square feature maps
        target_size = self.img_size
        
        # Create the decoder with appropriate upsampling
        decoder_layers = []
        current_size = feature_size
        channels = self.embed_dim
        
        print(f"Creating decoder to upsample from {feature_size}x{feature_size} to {target_size}x{target_size}")
        
        # Keep track of how many times we need to upsample by 2x
        while current_size < target_size:
            out_channels = max(channels // 2, 32)  # Reduce channels gradually, but not below 32
            decoder_layers.extend([
                nn.ConvTranspose2d(channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            ])
            current_size *= 2
            channels = out_channels
        
        # If we've overshot the target size, downsample with a stride-1 conv to exact size
        if current_size > target_size:
            decoder_layers.append(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
            )
        
        # Final layer to get to the target number of channels
        decoder_layers.append(
            nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1)
        )
        
        self.output_conv = nn.Sequential(*decoder_layers).to(device)
        print(f"Created decoder with {len(decoder_layers)} layers")
        
        self.feature_size_detected.fill_(True)
        
    def forward(self, x, t):
        # Extract features using the pretrained backbone
        batch_size = x.shape[0]
        
        # Get time embedding
        t_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, embed_dim]
        
        # Extract features with the pretrained backbone
        features = self.backbone(x)  # [B, embed_dim, h, w]
        
        # Get feature map dimensions
        h, w = features.shape[2], features.shape[3]
        
        # Initialize positional embedding and output decoder if this is the first forward pass
        if not self.feature_size_detected.item():
            self._init_pos_embed(h, w, x.device)
        
        # Convert to sequence of patches
        features = features.permute(0, 2, 3, 1)  # [B, h, w, embed_dim]
        features = features.reshape(batch_size, h * w, -1)  # [B, h*w, embed_dim]
        
        # Add positional embedding (should match the shape)
        features = features + self.pos_embed
        
        # Append class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat((cls_token, features), dim=1)
        
        # Add time embedding to each patch
        features = features + t_emb
        
        # Apply position dropout
        features = self.pos_drop(features)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            features = block(features)
        
        # Apply final normalization
        features = self.norm(features)
        
        # Remove class token for reconstruction
        features = features[:, 1:]
        
        # Reshape back to feature map
        features = features.reshape(batch_size, h, w, -1)
        features = features.permute(0, 3, 1, 2)  # [B, embed_dim, h, w]
        
        # Generate output image through decoder
        output = self.output_conv(features)
        
        return output

# Create the model
model = PretrainedDiffusionTransformer(
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

# Count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {count_parameters(model):,}")

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
    alphas = diffusion_params['alphas']
    alphas_cumprod = diffusion_params['alphas_cumprod']
    alphas_cumprod_prev = diffusion_params['alphas_cumprod_prev']
    sqrt_recip_alphas = diffusion_params['sqrt_recip_alphas']
    
    # Extract the current beta value
    beta = betas[t_index]
    
    # Predict the noise
    predicted_noise = model(x, t)
    
    # The formula for x_{t-1} given x_t
    # 1. Get the predicted x_0: (x_t - sqrt(1-αcum_t) * noise) / sqrt(αcum_t)
    # 2. Get the coefficient for x_0: sqrt(αcum_{t-1}) * beta / (1 - αcum_t)
    # 3. Get the coefficient for x_t: sqrt(α_t) * (1 - αcum_{t-1}) / (1 - αcum_t)
    # 4. Get the variance: (1 - αcum_{t-1}) * beta / (1 - αcum_t)
    
    alpha_cumprod_t = alphas_cumprod[t_index]
    alpha_cumprod_prev_t = alphas_cumprod_prev[t_index]
    sqrt_one_minus_alpha_cumprod_t = diffusion_params['sqrt_one_minus_alphas_cumprod'][t_index]
    
    # Calculate the mean of the posterior distribution
    pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t.reshape(-1, 1, 1, 1) * predicted_noise) / \
              torch.sqrt(alpha_cumprod_t).reshape(-1, 1, 1, 1)
    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
    
    # Calculate the mean for the reverse diffusion step
    posterior_mean = torch.sqrt(alpha_cumprod_prev_t).reshape(-1, 1, 1, 1) * pred_x0 + \
                    torch.sqrt(1 - alpha_cumprod_prev_t - beta).reshape(-1, 1, 1, 1) * x
    
    # Get the variance
    posterior_variance = ((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * beta).reshape(-1, 1, 1, 1)
    posterior_log_variance = torch.log(posterior_variance)
    
    # At t=0, just return the mean (no noise to add)
    if t_index == 0:
        return posterior_mean
    
    # Add some noise
    noise = torch.randn_like(x)
    return posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise

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
    samples = samples.detach().cpu().numpy()
    samples = np.transpose(samples, (0, 2, 3, 1))  # [B, H, W, C]
    
    plt.figure(figsize=(10, 10))
    for i in range(min(samples.shape[0], 16)):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i])
        plt.axis('off')
    
    plt.tight_layout()
    
    if epoch is not None:
        plt.savefig(f'visualizations_pretrained_diffusion/samples_epoch_{epoch}.png')
    else:
        plt.savefig('visualizations_pretrained_diffusion/final_samples.png')
    
    plt.close()
    print(f"Saved generated samples visualization")

# Function to visualize the forward diffusion process
def visualize_forward_diffusion(loader):
    """
    Visualize how an image gets progressively noisier in the forward diffusion process.
    """
    images, _ = next(iter(loader))
    img = images[0:1].to(device)  # Take the first image
    
    # Create timesteps
    vis_timesteps = [0, 100, 200, 400, 600, 999]  # Timesteps to visualize
    
    plt.figure(figsize=(15, 3))
    
    for i, t_step in enumerate(vis_timesteps):
        t = torch.tensor([t_step], device=device)
        noisy_img, _ = q_sample(img, t, diffusion_params)
        
        plt.subplot(1, len(vis_timesteps), i+1)
        noisy_img_np = noisy_img[0].detach().cpu().numpy().transpose(1, 2, 0)
        noisy_img_np = (noisy_img_np + 1) / 2  # Scale from [-1, 1] to [0, 1]
        noisy_img_np = np.clip(noisy_img_np, 0, 1)
        plt.imshow(noisy_img_np)
        plt.title(f"t={t_step}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('visualizations_pretrained_diffusion/forward_diffusion.png')
    plt.close()
    print("Saved forward diffusion visualization")

# Training function for the diffusion model
def train_diffusion_model():
    model.train()
    train_losses = []
    
    # Show the forward diffusion process on an example before training
    visualize_forward_diffusion(test_loader)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        
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
            
            # Gradient clipping
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
        
        # Generate and visualize samples
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            model.eval()
            # Generate samples
            samples = p_sample_loop(model, (4, channels, img_size, img_size), diffusion_params, n_samples=4)
            visualize_samples(samples, epoch=epoch+1)
            model.train()
    
    # Save the model
    torch.save(model.state_dict(), 'pretrained_diffusion_model.pth')
    print("Model saved to pretrained_diffusion_model.pth")
    
    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('visualizations_pretrained_diffusion/training_loss.png')
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
    print("Setting up pretrained diffusion model training...")
    
    # Train the diffusion model
    losses = train_diffusion_model()
    
    # Generate final samples
    generate_final_samples(16)
    
    print("Pretrained diffusion model training and sampling complete!") 