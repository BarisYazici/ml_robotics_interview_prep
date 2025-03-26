# PyTorch & Ray Cheat Sheet

## Table of Contents
- [PyTorch Basics](#pytorch-basics)
  - [Tensor Operations](#tensor-operations)
  - [Neural Network Building Blocks](#neural-network-building-blocks)
  - [Loss Functions & Optimizers](#loss-functions--optimizers)
  - [Data Loading](#data-loading)
  - [Training Loop](#training-loop)
  - [GPU Usage](#gpu-usage)
  - [Saving & Loading Models](#saving--loading-models)
- [Advanced PyTorch](#advanced-pytorch)
  - [Custom Modules](#custom-modules)
  - [Hooks & Debugging](#hooks--debugging)
  - [Distributed Training](#distributed-training)
- [Ray Integration](#ray-integration)
  - [Ray Basics](#ray-basics)
  - [Distributed Training with Ray](#distributed-training-with-ray)
  - [Hyperparameter Tuning with Ray Tune](#hyperparameter-tuning-with-ray-tune)
  - [Model Serving with Ray Serve](#model-serving-with-ray-serve)

## PyTorch Basics

### Tensor Operations

```python
import torch

# Creating tensors
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 4)  # 3x4 tensor of zeros
z = torch.ones(2, 3, 4)  # 2x3x4 tensor of ones
a = torch.rand(3, 4)  # Random values from uniform distribution [0,1)
b = torch.randn(3, 4)  # Random values from normal distribution (mean=0, std=1)
c = torch.arange(10)  # tensor([0, 1, 2, ..., 9])
d = torch.linspace(0, 1, 5)  # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

# Tensor attributes
shape = x.shape  # Size of tensor
dtype = x.dtype  # Data type
device = x.device  # CPU/GPU

# Common operations
add = x + y  # Element-wise addition
sub = x - y  # Element-wise subtraction
mul = x * y  # Element-wise multiplication
div = x / y  # Element-wise division
dot = torch.dot(x, x)  # Dot product
mm = torch.mm(a, b.t())  # Matrix multiplication (2D only)
matmul = torch.matmul(a, b.t())  # Generalized matrix multiplication

# Reshaping operations
view = x.view(1, 3)  # Reshape tensor (shares memory with original)
reshape = x.reshape(1, 3)  # Reshape tensor (may create new tensor)
squeeze = z.squeeze(0)  # Remove dimensions of size 1
unsqueeze = x.unsqueeze(0)  # Add dimension of size 1
flat = x.flatten()  # Flatten tensor to 1D
transpose = a.transpose(0, 1)  # Swap dimensions
permute = z.permute(2, 0, 1)  # Reorder dimensions

# Indexing and slicing
first_elem = x[0]  # First element
slice_tensor = x[1:3]  # Elements 1 through 2
adv_index = x[torch.tensor([0, 2])]  # Select specific indices
bool_mask = x[x > 1]  # Boolean masking
```

### Neural Network Building Blocks

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic linear layer
linear = nn.Linear(in_features=10, out_features=5)

# Activation functions
relu = F.relu
sigmoid = torch.sigmoid
tanh = torch.tanh
leaky_relu = F.leaky_relu

# Common layers
conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
dropout = nn.Dropout(p=0.5)
batch_norm = nn.BatchNorm2d(num_features=16)
layer_norm = nn.LayerNorm(normalized_shape=[16, 32, 32])

# Creating a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

model = SimpleNN()
summary = str(model)  # Get model summary
```

### Loss Functions & Optimizers

```python
import torch.optim as optim

# Common loss functions
mse_loss = nn.MSELoss()  # Mean squared error (regression)
ce_loss = nn.CrossEntropyLoss()  # Cross entropy (classification)
bce_loss = nn.BCELoss()  # Binary cross entropy
bce_with_logits = nn.BCEWithLogitsLoss()  # BCE with sigmoid
nll_loss = nn.NLLLoss()  # Negative log likelihood
l1_loss = nn.L1Loss()  # Mean absolute error
smooth_l1 = nn.SmoothL1Loss()  # Huber loss

# Optimizers
sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
rmsprop = optim.RMSprop(model.parameters(), lr=0.01)

# Learning rate schedulers
step_lr = optim.lr_scheduler.StepLR(optimizer=adam, step_size=30, gamma=0.1)
cosine_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer=adam, T_max=100)
reduce_on_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=adam, mode='min', factor=0.1, patience=10
)
```

### Data Loading

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Custom dataset example
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

# Common transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Loading built-in datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                download=True, transform=transform)

# Creating DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)

# Iterating through DataLoader
for batch_idx, (data, target) in enumerate(train_loader):
    # Training loop code here
    pass
```

### Training Loop

```python
# Basic training loop
def train(model, train_loader, optimizer, criterion, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for data, target in train_loader:
        # Move data to device (CPU/GPU)
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
    
    # Return average loss
    return running_loss / len(train_loader)

# Basic evaluation loop
def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # Return average loss and accuracy
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Training for multiple epochs
def train_model(model, train_loader, test_loader, optimizer, 
               criterion, scheduler, num_epochs, device):
    for epoch in range(num_epochs):
        # Train
        train_loss = train(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()
        
        # Print stats
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
```

### GPU Usage

```python
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create tensor on specific device
x_cpu = torch.tensor([1, 2, 3])
x_gpu = torch.tensor([1, 2, 3], device=device)  # Direct creation on device
x_gpu_alt = torch.tensor([1, 2, 3]).to(device)  # Move to device

# Move model to device
model = SimpleNN().to(device)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

# Initialize gradient scaler for mixed precision
scaler = GradScaler()

# Mixed precision training loop
def train_mixed_precision(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with autocast (mixed precision)
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        
        # Optimizer step with unscaled gradients
        scaler.step(optimizer)
        
        # Update scaler
        scaler.update()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)
```

### Saving & Loading Models

```python
# Saving and loading a model (recommended)
def save_model(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_model(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

# Quick saving (only model parameters)
torch.save(model.state_dict(), 'model_params.pth')

# Quick loading (only model parameters)
model.load_state_dict(torch.load('model_params.pth'))

# Saving entire model (not recommended)
torch.save(model, 'entire_model.pth')

# Loading entire model (not recommended)
model = torch.load('entire_model.pth')

# Saving models for production (TorchScript)
scripted_model = torch.jit.script(model)
scripted_model.save('scripted_model.pt')

# Loading TorchScript model
loaded_model = torch.jit.load('scripted_model.pt')
```

## Advanced PyTorch

### Custom Modules

```python
# Custom layer with learnable parameters
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

# Custom activation function
class SwishActivation(nn.Module):
    def __init__(self):
        super(SwishActivation, self).__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)

# Custom loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability of being correct
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

### Hooks & Debugging

```python
# Registering forward hooks
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.conv1.register_forward_hook(get_activation('conv1'))
output = model(input_tensor)
conv1_output = activation['conv1']  # Access stored activation

# Gradient hooks for debugging
def print_grad_hook(grad):
    print('Gradient:', grad)

x = torch.randn(5, requires_grad=True)
y = x * 2
z = y.sum()
y.register_hook(print_grad_hook)
z.backward()

# Using torchinfo for model summary
from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))
```

### Distributed Training

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use 'nccl' for GPU, 'gloo' for CPU
        init_method='tcp://localhost:12355',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()

def train_worker(rank, world_size):
    # Initialize process group
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = SimpleNN().to(rank)
    
    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    
    # Create sampler for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    # Create DataLoader with sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    # ...
    
    # Cleanup
    cleanup()

# Launch distributed training
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
```

## Ray Integration

### Ray Basics

```python
import ray

# Initialize Ray
ray.init()  # Local initialization
# ray.init(address='auto')  # Connect to existing Ray cluster

# Define a remote function
@ray.remote
def remote_function(x):
    return x * x

# Execute the remote function
future = remote_function.remote(4)
result = ray.get(future)  # result = 16

# Define a remote class
@ray.remote
class Counter:
    def __init__(self):
        self.value = 0
        
    def increment(self):
        self.value += 1
        return self.value
        
    def get_value(self):
        return self.value

# Create a remote object
counter = Counter.remote()
future = counter.increment.remote()
value = ray.get(future)  # value = 1

# Parallel processing with Ray
@ray.remote
def process_batch(batch_data):
    # Process batch here
    return result

# Process multiple batches in parallel
futures = [process_batch.remote(batch) for batch in batches]
results = ray.get(futures)  # Wait for all tasks to complete
```

### Distributed Training with Ray

```python
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

# Define a training function
def train_func(config):
    # Standard PyTorch training setup
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    # Get the DataLoader for the trainer
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    
    # Prepare model for distributed training
    model = ray.train.torch.prepare_model(model)
    
    # Training loop
    for epoch in range(config["num_epochs"]):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Report metrics to Ray
        ray.train.report({"loss": train_loss / len(train_loader)})

# Create a TorchTrainer
trainer = TorchTrainer(
    train_func,
    train_loop_config={"lr": 0.001, "num_epochs": 10},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)

# Start training
result = trainer.fit()
```

### Hyperparameter Tuning with Ray Tune

```python
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Define a trainable function
def train_model(config):
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        # Training code here
        train_loss = ...
        test_loss = ...
        
        # Report metrics to Ray Tune
        tune.report(loss=test_loss, accuracy=test_acc)

# Configure hyperparameter search space
config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "momentum": tune.uniform(0.1, 0.9)
}

# Configure scheduler
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=10,
    grace_period=1,
    reduction_factor=2
)

# Run hyperparameter tuning
tuner = tune.Tuner(
    train_model,
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=10
    ),
    param_space=config
)

results = tuner.fit()

# Get best configuration
best_result = results.get_best_result("loss", "min")
best_config = best_result.config
print(f"Best config: {best_config}")
```

### Model Serving with Ray Serve

```python
import ray
from ray import serve
from starlette.requests import Request
import json
import torch
import numpy as np

# Initialize Ray Serve
ray.init()
serve.start()

# Define a deployable model
@serve.deployment(route_prefix="/predict")
class TorchPredictor:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    async def __call__(self, request: Request):
        # Parse input data
        data = await request.json()
        input_data = np.array(data["input"])
        
        # Preprocess
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        predictions = output.cpu().numpy().tolist()
        
        return {"predictions": predictions}

# Deploy model
predictor = TorchPredictor.bind("model.pt")
predictor_handle = serve.run(predictor)

# Query endpoint (from client)
import requests

prediction = requests.post(
    "http://localhost:8000/predict",
    json={"input": [[1.0, 2.0, 3.0, 4.0]]}
)
print(prediction.json())

# Shutdown service
serve.shutdown()
```