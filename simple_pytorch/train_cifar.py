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
os.makedirs('visualizations', exist_ok=True)

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
num_epochs = 5
batch_size = 64
learning_rate = 0.001

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

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes for CIFAR-10
        self.relu = nn.ReLU()
        
        # Store activations
        self.activations = {}
    
    def forward(self, x):
        # First conv block
        x = self.relu(self.conv1(x))
        self.activations['conv1'] = x.detach()
        x = self.pool(x)
        
        # Second conv block
        x = self.relu(self.conv2(x))
        self.activations['conv2'] = x.detach()
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 64 * 8 * 8)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_activations(self):
        return self.activations

# Initialize the model
model = SimpleCNN().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
            individual_filename = f'cifar_sample_{i}_{classes[labels[i]]}.png'
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
        plt.savefig('cifar_examples.png')
        print("Saved combined samples to cifar_examples.png")
        
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
            if (i+1) % 100 == 0:
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
    torch.save(model.state_dict(), 'cifar_model.pth')
    print("Model saved to cifar_model.pth")

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
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        
        # Move data back to CPU for visualization
        images = images.cpu().numpy()
        labels = labels.numpy()
        
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
            title = f"Pred: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}"
            plt.title(title, color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/predictions.png')
        print("Saved prediction visualization to visualizations/predictions.png")
        
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
        plt.savefig('visualizations/confusion_matrix.png')
        print("Saved confusion matrix to visualizations/confusion_matrix.png")
        
        # Analyze where the model commonly fails
        print("\nCommon misclassifications:")
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and cm[i, j] > 10:  # Threshold for "common" misclassification
                    print(f"  {classes[i]} misclassified as {classes[j]}: {cm[i, j]} times")
    
    except Exception as e:
        print(f"Error in confusion matrix plotting: {str(e)}")

# Function to visualize CNN activations
def visualize_feature_maps(image_idx=0):
    """Visualize CNN activations to see what features the network is focusing on."""
    try:
        # Get a single image
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        
        # Select specific image
        img = images[image_idx:image_idx+1].to(device)
        label = labels[image_idx]
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
        
        # Get activations
        activations = model.get_activations()
        
        # Save the original image for reference
        plt.figure(figsize=(6, 6))
        img_np = images[image_idx].cpu().numpy().transpose((1, 2, 0))
        img_np = img_np * 0.5 + 0.5  # denormalize
        plt.imshow(img_np)
        plt.title(f"Original Image: {classes[label]}, Predicted: {classes[predicted.item()]}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'visualizations/original_image_{image_idx}.png')
        
        # Visualize first conv layer activations (shows what low-level features are detected)
        conv1_activations = activations['conv1'].cpu().squeeze(0)
        
        # Plotting first 16 feature maps from conv1
        plt.figure(figsize=(10, 10))
        for i in range(min(16, conv1_activations.size(0))):
            plt.subplot(4, 4, i + 1)
            plt.imshow(conv1_activations[i], cmap='viridis')
            plt.axis('off')
        plt.suptitle(f"First Convolutional Layer Activations - Low-level features")
        plt.tight_layout()
        plt.savefig(f'visualizations/conv1_activations_image_{image_idx}.png')
        
        # Visualize second conv layer activations (shows what higher-level features are detected)
        conv2_activations = activations['conv2'].cpu().squeeze(0)
        
        # Plotting first 16 feature maps from conv2
        plt.figure(figsize=(10, 10))
        for i in range(min(16, conv2_activations.size(0))):
            plt.subplot(4, 4, i + 1)
            plt.imshow(conv2_activations[i], cmap='viridis')
            plt.axis('off')
        plt.suptitle(f"Second Convolutional Layer Activations - Higher-level features")
        plt.tight_layout()
        plt.savefig(f'visualizations/conv2_activations_image_{image_idx}.png')
        
        print(f"Saved feature map visualizations for image {image_idx}")
        
    except Exception as e:
        print(f"Error in feature map visualization: {str(e)}")

# Function to create a heatmap overlay showing where the CNN is focusing
def visualize_activation_heatmap(image_idx=0):
    """Create a heatmap overlay showing where the CNN is focusing its attention."""
    try:
        # Get a single image
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        
        # Select specific image
        img = images[image_idx:image_idx+1].to(device)
        label = labels[image_idx]
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
        
        # Get activations
        activations = model.get_activations()
        
        # Convert the image to numpy for visualization
        img_np = images[image_idx].cpu().numpy().transpose((1, 2, 0))
        img_np = img_np * 0.5 + 0.5  # denormalize
        
        # Get the second layer activations (higher level features)
        conv2_activations = activations['conv2'].cpu().squeeze(0)
        
        # Sum over all feature channels to get a single heatmap
        heatmap = torch.mean(conv2_activations, dim=0).numpy()
        
        # Resize heatmap to match image dimensions
        zoom_factor = img_np.shape[0] / heatmap.shape[0]
        heatmap = zoom(heatmap, zoom_factor)
        
        # Create a figure with 3 subplots
        plt.figure(figsize=(15, 5))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title(f"Original: {classes[label]}\nPredicted: {classes[predicted.item()]}")
        plt.axis('off')
        
        # Plot heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='hot')
        plt.title("Activation Heatmap")
        plt.axis('off')
        
        # Plot overlay
        plt.subplot(1, 3, 3)
        plt.imshow(img_np)
        plt.imshow(heatmap, cmap='hot', alpha=0.5)  # Overlay with transparency
        plt.title("Activation Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'visualizations/heatmap_overlay_image_{image_idx}.png')
        print(f"Saved heatmap visualization for image {image_idx}")
        
    except Exception as e:
        print(f"Error in heatmap visualization: {str(e)}")

# Function to visualize filters/weights
def visualize_filters():
    """Visualize the filters/weights in the convolutional layers."""
    try:
        # Get the weights for the first conv layer (shape: out_channels, in_channels, height, width)
        conv1_weights = model.conv1.weight.data.cpu().numpy()
        
        # Plotting filters
        plt.figure(figsize=(10, 10))
        for i in range(min(16, conv1_weights.shape[0])):
            # For each output channel, visualize the filter for all input channels
            filter_i = conv1_weights[i]
            
            # Normalize filter for better visualization
            filter_i = (filter_i - filter_i.min()) / (filter_i.max() - filter_i.min() + 1e-8)
            
            # Plot the filter as an RGB image (3 input channels)
            plt.subplot(4, 4, i + 1)
            plt.imshow(np.transpose(filter_i, (1, 2, 0)))
            plt.axis('off')
            
        plt.suptitle("First Convolutional Layer Filters")
        plt.tight_layout()
        plt.savefig('visualizations/conv1_filters.png')
        print("Saved filter visualizations")
        
    except Exception as e:
        print(f"Error in filter visualization: {str(e)}")

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
        
        # Visualize network features and activations for multiple images
        print("\nVisualizing network features and activations...")
        visualize_filters()
        
        for i in range(5):  # Analyze 5 different images
            visualize_feature_maps(image_idx=i)
            visualize_activation_heatmap(image_idx=i)
            
        print("\nAll visualizations completed and saved to the 'visualizations' directory.")
        print("Check these images to understand what features the network has learned.")
    except Exception as e:
        print(f"Error in visualization generation: {str(e)}")