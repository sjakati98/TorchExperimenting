"""
Example PyTorch Experiment

A simple example to get you started with PyTorch experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class SimpleNet(nn.Module):
    """A simple neural network for demonstration"""
    
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


def create_dummy_data(n_samples=1000, input_dim=10):
    """Create dummy data for testing"""
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, 2, (n_samples,))
    return TensorDataset(X, y)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(dataloader), 100.0 * correct / total


def main():
    """Main experiment function"""
    print("🔥 PyTorch Experiment Example\n")
    
    # Setup
    device = get_device()
    print(f"📱 Using device: {device}")
    
    # Hyperparameters
    input_dim = 10
    hidden_dim = 64
    output_dim = 2
    batch_size = 32
    learning_rate = 0.001
    epochs = 15
    
    # Model
    model = SimpleNet(input_dim, hidden_dim, output_dim).to(device)
    print(f"\n🧠 Model: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data
    dataset = create_dummy_data(1000, input_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\n🚀 Training for {epochs} epochs...\n")
    for epoch in range(epochs):
        loss, acc = train_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()

