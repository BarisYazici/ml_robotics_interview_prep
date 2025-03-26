import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm


# Define a simple CNN model for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self, l1=64, l2=128):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, l1, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(l1, l2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(l2 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self.fc1.in_features)  # Flatten
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x


# Function to load and preprocess CIFAR-10 data
def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download and load training dataset
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    
    # Download and load test dataset
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)
    
    return trainset, testset


# Training function for Ray Tune
def train_cifar(config, checkpoint_dir=None):
    # Get data loaders
    batch_size = config["batch_size"]
    trainset, testset = load_data()
    
    # Create data loaders for the distributed training
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model, loss function, and optimizer based on the hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleCNN(
        l1=config["layer1_size"], 
        l2=config["layer2_size"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    
    # Load checkpoint if provided
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint_path)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # Training loop
    for epoch in range(config["epochs"]):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward, backward, and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # Print every 200 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}")
                running_loss = 0.0
        
        # Test the model
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Accuracy after epoch {epoch+1}: {accuracy:.2f}%")
        
        # Report metrics to Ray Tune - using dict style that works with older Ray versions
        result = {"accuracy": accuracy, "loss": running_loss}
        
        # Compatible with older versions of Ray
        if hasattr(tune, "report"):
            tune.report(**result)
        else:
            # For older ray versions
            from ray import tune as ray_tune
            ray_tune.track.log(**result)
        
        # Save a checkpoint
        if hasattr(tune, "checkpoint_dir"):
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                torch.save(
                    (model.state_dict(), optimizer.state_dict()),
                    os.path.join(checkpoint_dir, "checkpoint")
                )
        else:
            # For older ray versions
            checkpoint_path = os.path.join("./checkpoints", f"checkpoint_epoch_{epoch}")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                (model.state_dict(), optimizer.state_dict()),
                checkpoint_path
            )
            ray_tune.track.log(checkpoint_path=checkpoint_path)


# Function to run distributed hyperparameter tuning with Ray Tune
def tune_cifar(num_samples=5, max_num_epochs=5, gpus_per_trial=0):
    # Define the hyperparameter search space
    config = {
        "layer1_size": tune.choice([32, 64, 128]),
        "layer2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([64, 128, 256]),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "epochs": max_num_epochs
    }
    
    # Make directory for checkpoints
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Use ASHA scheduler with parameters compatible with older Ray versions
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    
    # Configure resources per trial
    resources_per_trial = {"cpu": 2}
    if gpus_per_trial > 0:
        resources_per_trial["gpu"] = gpus_per_trial
    
    # Run tuning with compatibility for older Ray versions
    try:
        # For newer Ray versions
        analysis = tune.run(
            train_cifar,
            resources_per_trial=resources_per_trial,
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            checkpoint_at_end=True,
            local_dir="./ray_results",
            name="cifar_tune_asha"
        )
        best_trial = analysis.get_best_trial("accuracy", "max", "last")
    except Exception as e:
        print(f"Error running with newer Ray API: {e}")
        print("Falling back to older Ray Tune API...")
        
        # For older Ray versions
        analysis = tune.run(
            train_cifar,
            stop={"training_iteration": max_num_epochs},
            resources_per_trial=resources_per_trial,
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            local_dir="./ray_results",
            name="cifar_tune_asha"
        )
        best_trial = analysis.get_best_trial("accuracy", "max")
    
    print("Best trial config:", best_trial.config)
    try:
        print("Best trial final accuracy:", best_trial.last_result["accuracy"])
    except:
        print("Best trial final accuracy:", best_trial.metric_analysis["accuracy"]["max"])
    
    return best_trial


# Function to parallelize data preprocessing with Ray
def parallel_preprocess_data():
    print("Performing parallel data preprocessing with Ray...")
    
    # Create data transformation function
    @ray.remote
    def preprocess_batch(indices, dataset):
        # Process one batch of indices
        batch_images = []
        batch_labels = []
        
        for idx in indices:
            img, label = dataset[idx]
            # Convert tensor to numpy
            img_np = img.numpy()
            batch_images.append(img_np)
            batch_labels.append(label)
        
        # Stack into numpy arrays
        batch_images = np.stack(batch_images)
        batch_labels = np.array(batch_labels)
        
        # Simple normalization (additional to the transforms already applied)
        batch_images = (batch_images - batch_images.mean(axis=(0, 2, 3), keepdims=True)) / \
                       (batch_images.std(axis=(0, 2, 3), keepdims=True) + 1e-7)
        
        return {"images": batch_images, "labels": batch_labels}
    
    # Load data
    trainset, testset = load_data()
    
    # Create indices for batching
    batch_size = 1000
    train_indices = [list(range(i, min(i + batch_size, len(trainset)))) 
                    for i in range(0, len(trainset), batch_size)]
    test_indices = [list(range(i, min(i + batch_size, len(testset)))) 
                   for i in range(0, len(testset), batch_size)]
    
    # Process in parallel using Ray
    print("Processing training data in parallel...")
    train_futures = [preprocess_batch.remote(indices, trainset) for indices in train_indices]
    train_processed = ray.get(train_futures)
    
    print("Processing test data in parallel...")
    test_futures = [preprocess_batch.remote(indices, testset) for indices in test_indices]
    test_processed = ray.get(test_futures)
    
    print(f"Train dataset size: {len(trainset)}")
    print(f"Test dataset size: {len(testset)}")
    
    return train_processed, test_processed


# Function to train with the best hyperparameters found by Ray Tune
def train_with_best_params(best_trial):
    print(f"Training with best parameters: {best_trial.config}")
    
    # Extract the best configuration
    config = best_trial.config
    
    # Initialize a model with the best parameters
    model = SimpleCNN(
        l1=config["layer1_size"],
        l2=config["layer2_size"]
    )
    
    # Load data
    trainset, testset = load_data()
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config["batch_size"], shuffle=False)
    
    # Set up training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Save the final model
    torch.save(model.state_dict(), "best_cifar_model.pth")
    print("Model saved to best_cifar_model.pth")
    
    return model


# Main function to demonstrate different Ray functionalities
def main():
    # Initialize Ray
    ray.init()
    
    # Demonstrate Ray for parallel data preprocessing
    print("\n===== Demonstrating Parallel Data Preprocessing =====")
    train_data, test_data = parallel_preprocess_data()
    
    # Demonstrate Ray Tune for hyperparameter optimization with a smaller sample
    print("\n===== Demonstrating Hyperparameter Tuning with Ray Tune =====")
    
    # Smaller run initially to test compatibility
    print("Running a small test with 2 samples and 2 epochs...")
    best_trial = tune_cifar(num_samples=2, max_num_epochs=2)
    
    # If the smaller run works, run with more samples
    print("\n===== Training with the Best Parameters =====")
    final_model = train_with_best_params(best_trial)
    
    # Show summary of what was demonstrated
    print("\n===== Ray Functionality Demonstration Summary =====")
    print("1. Ray Core: Parallel data preprocessing")
    print("2. Ray Tune: Hyperparameter optimization")
    print("3. Model Training with Best Parameters")
    print("\nAll Ray functionalities have been demonstrated successfully!")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main() 