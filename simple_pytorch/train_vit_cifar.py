import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from scipy.ndimage import zoom

# Create a directory for saving visualizations
os.makedirs('visualizations_vit', exist_ok=True)

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
num_epochs = 10  # Vision Transformers might need more epochs
batch_size = 128
learning_rate = 0.001
img_size = 32  # CIFAR-10 image size
patch_size = 4  # Size of image patches
num_patches = (img_size // patch_size) ** 2  # Number of patches
embed_dim = 192  # Embedding dimension
num_heads = 4  # Number of attention heads
num_layers = 4  # Number of transformer layers
mlp_dim = embed_dim * 4  # MLP hidden dimension
num_classes = 10  # CIFAR-10 has 10 classes
dropout_rate = 0.1  # Dropout rate

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

# Multi-head Self-Attention (MSA) module
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Define query, key, value projections for all heads in one go
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For storing attention weights
        self.attention_weights = None
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head attention
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        self.attention_weights = attn  # Store for visualization
        attn = self.dropout(attn)
        
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    
    def get_attention_weights(self):
        """Return the attention weights for visualization"""
        return self.attention_weights

# MLP module
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

# Transformer Encoder Block
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
    
    def get_attention_weights(self):
        """Return the attention weights from the attention module"""
        return self.attn.get_attention_weights()

# Patch Embedding module
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

# Vision Transformer (ViT) model
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, 
                 mlp_dim, num_layers, dropout=0.0):
        super().__init__()
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Positional embedding and class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
        # For storing attention weights
        self.attention_maps = []
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        
    def forward(self, x):
        # Store attention weights for visualization
        self.attention_maps = []
        
        # Get patch embeddings
        batch_size = x.shape[0]
        x = self.patch_embed(x)  # [batch_size, n_patches, embed_dim]
        
        # Add positional embeddings and class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, n_patches + 1, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            # Store attention weights
            self.attention_maps.append(block.get_attention_weights())
        
        # Classification using the [CLS] token
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Take the [CLS] token
        x = self.head(cls_token_final)
        
        return x
    
    def get_attention_maps(self):
        """Return attention maps from all transformer blocks for visualization"""
        return self.attention_maps

# Initialize the model
model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=3,
    num_classes=num_classes,
    embed_dim=embed_dim,
    num_heads=num_heads,
    mlp_dim=mlp_dim,
    num_layers=num_layers,
    dropout=dropout_rate
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

# Function to display some examples
def show_random_images():
    try:
        # Set a non-interactive backend for matplotlib
        import matplotlib
        matplotlib.use('Agg')  # Use the 'Agg' backend which is good for saving files
        
        # Get random training images
        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        
        # Create figure
        plt.figure(figsize=(10, 4))
        
        # Create and save individual images as well
        print("Saving individual sample images...")
        for i in range(5):
            # Create a separate figure for each image
            plt.figure(figsize=(3, 3))
            
            # Make sure image tensor is on CPU before converting to numpy
            img = images[i].cpu().numpy().transpose((1, 2, 0))
            img = img * 0.5 + 0.5  # denormalize
            
            plt.imshow(img)
            plt.title(classes[labels[i]])
            plt.axis('off')
            plt.tight_layout()
            
            # Save each individual image
            individual_filename = f'vit_cifar_sample_{i}_{classes[labels[i]]}.png'
            plt.savefig(individual_filename)
            plt.close()  # Close the figure to free memory
            print(f"  Saved {individual_filename}")
        
        # Now create a plot with all 5 images
        plt.figure(figsize=(15, 3))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            img = images[i].cpu().numpy().transpose((1, 2, 0))
            img = img * 0.5 + 0.5  # denormalize
            plt.imshow(img)
            plt.title(classes[labels[i]])
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('vit_cifar_examples.png')
        print("Saved combined samples to vit_cifar_examples.png")
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        print("Continuing without visualization...")

# Training function
def train_model():
    model.train()
    total_step = len(train_loader)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress
            if (i+1) % 50 == 0:  # Print more frequently because batches are smaller
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Accuracy: {100 * correct / total:.2f}%')
        
        # Print epoch statistics
        epoch_loss = running_loss / total_step
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Training Loss: {epoch_loss:.4f}, '
              f'Training Accuracy: {epoch_acc:.2f}%')
        
        # Evaluate the model after each epoch
        evaluate_model()
    
    # Save the model
    torch.save(model.state_dict(), 'vit_cifar_model.pth')
    print("Model saved to vit_cifar_model.pth")

# Evaluation function
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Function to visualize predictions
def visualize_predictions(num_images=10):
    """Visualize model predictions on random test images, showing both correct and incorrect predictions."""
    try:
        model.eval()
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        
        # Get predictions
        images_device = images.to(device)
        outputs = model(images_device)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Convert to probabilities
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        
        # Move data back to CPU for visualization
        images = images.cpu().numpy()
        labels = labels.numpy()
        probabilities = probabilities.cpu().numpy()
        
        # Create a figure
        plt.figure(figsize=(12, 12))
        
        # Plot images with predictions
        for i in range(min(num_images, len(images))):
            plt.subplot(4, 5, i+1)
            img = np.transpose(images[i], (1, 2, 0))
            img = img * 0.5 + 0.5  # denormalize
            plt.imshow(img)
            
            # Green text for correct predictions, red for incorrect
            color = "green" if predicted[i] == labels[i] else "red"
            pred_class = classes[predicted[i]]
            true_class = classes[labels[i]]
            confidence = probabilities[i, predicted[i]] * 100
            
            title = f"Pred: {pred_class}\nTrue: {true_class}\nConf: {confidence:.1f}%"
            plt.title(title, color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations_vit/predictions.png')
        print("Saved prediction visualization to visualizations_vit/predictions.png")
        
    except Exception as e:
        print(f"Error in prediction visualization: {str(e)}")

# Function to create a confusion matrix
def plot_confusion_matrix():
    """Create and plot a confusion matrix to visualize model performance across classes."""
    try:
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('visualizations_vit/confusion_matrix.png')
        print("Saved confusion matrix to visualizations_vit/confusion_matrix.png")
        
        # Analyze where the model commonly fails
        print("\nCommon misclassifications:")
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and cm[i, j] > 10:  # Threshold for "common" misclassification
                    print(f"  {classes[i]} misclassified as {classes[j]}: {cm[i, j]} times")
    
    except Exception as e:
        print(f"Error in confusion matrix plotting: {str(e)}")

# Function to visualize attention maps
def visualize_attention(image_idx=0):
    """Visualize self-attention maps from the Vision Transformer."""
    try:
        # Get a single image
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        
        # Select specific image
        img = images[image_idx:image_idx+1].to(device)
        label = labels[image_idx]
        
        # Forward pass to get attention maps
        model.eval()
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
        
        # Get attention maps from all transformer blocks
        attention_maps = model.get_attention_maps()
        
        # Convert the image to numpy for reference
        img_np = images[image_idx].cpu().numpy().transpose((1, 2, 0))
        img_np = img_np * 0.5 + 0.5  # denormalize
        
        # Visualize attention maps for the CLS token
        plt.figure(figsize=(15, 5 * len(attention_maps)))
        
        for block_idx, attn_block in enumerate(attention_maps):
            # Attention map shape: [batch_size, num_heads, seq_len, seq_len]
            # We'll visualize attention from CLS token (first token) to all patches
            
            # Get mean attention across all heads
            cls_attn = attn_block[0, :, 0, 1:].mean(0).cpu().numpy()  # Mean across heads
            
            # Reshape attention to match the image patches
            patches_per_side = img_size // patch_size
            attn_map = cls_attn.reshape(patches_per_side, patches_per_side)
            
            # Resize attention map to image size for overlay
            resized_attn = zoom(attn_map, img_size / patches_per_side)
            
            # Plot attention map and overlay
            plt.subplot(len(attention_maps), 3, block_idx * 3 + 1)
            plt.imshow(img_np)
            plt.title(f"Original: {classes[label]}\nPredicted: {classes[predicted.item()]}")
            plt.axis('off')
            
            plt.subplot(len(attention_maps), 3, block_idx * 3 + 2)
            plt.imshow(resized_attn, cmap='hot')
            plt.title(f"Block {block_idx + 1} Attention")
            plt.axis('off')
            
            plt.subplot(len(attention_maps), 3, block_idx * 3 + 3)
            plt.imshow(img_np)
            plt.imshow(resized_attn, cmap='hot', alpha=0.5)
            plt.title(f"Block {block_idx + 1} Overlay")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'visualizations_vit/attention_maps_image_{image_idx}.png')
        print(f"Saved attention map visualization for image {image_idx}")
        
    except Exception as e:
        print(f"Error in attention visualization: {str(e)}")

# Main function
if __name__ == '__main__':
    try:
        print("Showing sample CIFAR-10 images...")
        show_random_images()
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        print("Continuing with training...")
    
    print("\nStarting training...")
    train_model()
    
    print("\nFinal evaluation...")
    evaluate_model()
    
    print("\nGenerating visualizations...")
    try:
        # Visualize predictions and confusion matrix
        visualize_predictions(num_images=20)
        plot_confusion_matrix()
        
        # Visualize attention maps for multiple images
        print("\nVisualizing attention maps...")
        for i in range(5):  # Analyze 5 different images
            visualize_attention(image_idx=i)
            
        print("\nAll visualizations completed and saved to the 'visualizations_vit' directory.")
        print("Check these images to understand what the Vision Transformer has learned.")
    except Exception as e:
        print(f"Error in visualization generation: {str(e)}") 