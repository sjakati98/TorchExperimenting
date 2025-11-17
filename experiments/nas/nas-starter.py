from collections import defaultdict
import random
import time
from typing import Dict, List, Optional, Tuple, Any, Union

import matplotlib.pyplot as plt
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# set the mac accelerator
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU :(")

print(f"Pytorch Version: {torch.__version__}")

# ====

# DynamicCNN is the architecture template
class DynamicCNN(nn.Module):
    
    def __init__(self, architecture: Dict[str, Union[int, bool]]) -> None:
        super(DynamicCNN, self).__init__()
        self.arch = architecture

        ## add the conv layers dynamically
        self.conv_layers = nn.ModuleList() # ModuleList allows for variable number of layers
        in_channels = 1 # MNIST is grayscale

        for i in range(architecture['num_conv_layers']):
            self.conv_layers.append(nn.Conv2d(
                in_channels,
                architecture['filters'],
                kernel_size=architecture['kernel_size'],
                padding=architecture['kernel_size']//2,
            ))
            in_channels = architecture['filters']

        # Calculate how often to pool to avoid spatial dimensions going to 0
        # For MNIST (28x28), we can pool at most 4 times (28 -> 14 -> 7 -> 3 -> 1)
        num_conv = architecture['num_conv_layers']
        max_pools = min(4, num_conv)  # At most 4 pooling operations for 28x28 images
        
        # Determine which layers get pooling (spread evenly)
        self.pool_indices = set()
        if max_pools > 0:
            pool_every = max(1, num_conv // max_pools)
            for i in range(0, num_conv, pool_every):
                if len(self.pool_indices) < max_pools:
                    self.pool_indices.add(i)
        
        # Calculate final spatial size
        spatial_size = 28 // (2 ** len(self.pool_indices))
        flattened_size = architecture['filters'] * spatial_size * spatial_size

        self.fc1 = nn.Linear(flattened_size, architecture['dense_units'])
        self.fc2 = nn.Linear(architecture['dense_units'], 10)

        # Activations and regularization
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5) if architecture['use_dropout'] else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.conv_layers):
            x = self.relu(conv(x))
            # Only pool at specified indices
            if i in self.pool_indices:
                x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# training data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    device: torch.device
) -> Tuple[float, float]:
    """ Train model for a single epoch """
    model.train()
    total_loss: float = 0.0
    correct: int = 0
    total: int = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward Pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward Pass
        loss.backward()
        optimizer.step()

        # Track accuracy
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    return avg_loss, accuracy

def evaluate_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on test set"""
    model.eval()  # Set to evaluation mode
    test_loss: float = 0.0
    correct: int = 0
    
    with torch.no_grad():  # Don't compute gradients during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# ====

# define the search space for the architetures
SEARCH_SPACE: Dict[str, List[Union[int, bool]]] = {
    'num_conv_layers': [1, 2, 4, 8],
    'filters': [16, 32, 64],
    'kernel_size': [3, 5],
    'use_dropout': [True, False],
    'dense_units': [64, 128, 256]
}

# print the architecture options
print("Search Space: \n")
table = tabulate(
    [[key, ', '.join(map(str, value))] for key, value in SEARCH_SPACE.items()],
    headers=["Variable", "Options"],
    tablefmt='orgtbl'
)
print(table + "\n")

total_architectures = 1
for values in SEARCH_SPACE.values():
    total_architectures *= len(values)
print(f"\nTotal Possible Architectures: {total_architectures}")


# ====
def sample_architecture(search_space: Dict[str, List[Union[int, bool]]]) -> Dict[str, Union[int, bool]]:
    """Sample a random architecture from the search space"""
    return {k: random.choice(v) for k, v in search_space.items()}

def quick_evaluate(
    architecture: Dict[str, Union[int, bool]], 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    device: torch.device, 
    epochs: int = 2
) -> Optional[Dict[str, Any]]:
    """Quickly evaluate an architecture by training for a few epochs"""
    try:
        model = DynamicCNN(architecture).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train for a few epochs
        train_loss: float = 0.0
        train_acc: float = 0.0
        for epoch in range(epochs):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        
        # Final evaluation
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        return {
            'architecture': architecture,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'num_params': num_params,
            'train_accuracy': train_acc
        }
    except Exception as e:
        print(f"Error evaluating architecture {architecture}: {e}")
        return None


def random_search(
    search_space: Dict[str, List[Union[int, bool]]], 
    num_samples: int, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    device: torch.device, 
    epochs: int = 2
) -> List[Dict[str, Any]]:
    """
    Perform random search over the architecture space.
    This is one of the simplest but surprisingly effective NAS methods!
    """
    print(f"\n--- Starting Random Search ---")
    print(f"Will evaluate {num_samples} random architectures")
    print(f"Each will train for {epochs} epochs")
    print(f"Estimated time: ~{num_samples * epochs * 20} seconds\n")
    
    results: List[Dict[str, Any]] = []
    
    for i in range(num_samples):
        print(f"[{i+1}/{num_samples}] ", end='')
        
        # Sample architecture
        architecture = sample_architecture(search_space)
        print(f"Testing: {architecture}")
        
        # Evaluate
        start_time = time.time()
        result = quick_evaluate(architecture, train_loader, test_loader, device, epochs)
        elapsed = time.time() - start_time
        
        if result:
            results.append(result)
            print(f"  ✓ Accuracy: {result['test_accuracy']:.2f}% | "
                  f"Params: {result['num_params']:,} | "
                  f"Time: {elapsed:.1f}s\n")
    
    return results

# Run random search with 15 architectures
NUM_ARCHITECTURES = 15  # Try 15 different architectures
QUICK_EPOCHS = 2  # Train each for only 2 epochs

# For quick demo, use smaller subset
small_train_loader = DataLoader(
    torch.utils.data.Subset(train_dataset, range(10000)),
    batch_size=128, 
    shuffle=True
)

results = random_search(
    SEARCH_SPACE, 
    NUM_ARCHITECTURES, 
    small_train_loader, 
    test_loader, 
    device, 
    QUICK_EPOCHS
)

# Sort by accuracy
results_sorted = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)

print("\n--- Top 5 Architectures ---")
for i, result in enumerate(results_sorted[:5]):
    print(f"\n{i+1}. Accuracy: {result['test_accuracy']:.2f}%")
    print(f"   Architecture: {result['architecture']}")
    print(f"   Parameters: {result['num_params']:,}")

# Best architecture
best_result = results_sorted[0]
print("\n" + "="*70)
print("🏆 BEST ARCHITECTURE FOUND")
print("="*70)
print(f"Accuracy: {best_result['test_accuracy']:.2f}%")
print(f"Architecture: {best_result['architecture']}")
print(f"Parameters: {best_result['num_params']:,}")


print("\n--- Generating Visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Neural Architecture Search Results', fontsize=16, fontweight='bold')


ax1 = axes[0, 0]
accuracies = [r['test_accuracy'] for r in results]
ax1.hist(accuracies, bins=10, edgecolor='black', alpha=0.7)
ax1.axvline(best_result['test_accuracy'], color='red', linestyle='--', linewidth=2, label='Best')
ax1.set_xlabel('Test Accuracy (%)')
ax1.set_ylabel('Count')
ax1.set_title('Accuracy Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)


ax2 = axes[0, 1]
params = [r['num_params'] for r in results]
ax2.scatter(params, accuracies, alpha=0.6, s=100)
ax2.scatter([best_result['num_params']], [best_result['test_accuracy']], 
            color='red', s=200, marker='*', label='Best', zorder=5)
ax2.set_xlabel('Number of Parameters')
ax2.set_ylabel('Test Accuracy (%)')
ax2.set_title('Model Size vs Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)


ax3 = axes[1, 0]
hyperparam_impact = defaultdict(lambda: defaultdict(list))
for result in results:
    for key, value in result['architecture'].items():
        hyperparam_impact[key][str(value)].append(result['test_accuracy'])

# Plot num_conv_layers impact
if 'num_conv_layers' in hyperparam_impact:
    layers_data = hyperparam_impact['num_conv_layers']
    layers = sorted([int(k) for k in layers_data.keys()])
    avg_accs = [sum(layers_data[str(l)]) / len(layers_data[str(l)]) for l in layers]
    ax3.bar([str(l) for l in layers], avg_accs, alpha=0.7)
    ax3.set_xlabel('Number of Conv Layers')
    ax3.set_ylabel('Average Accuracy (%)')
    ax3.set_title('Impact of Conv Layers')
    ax3.grid(True, alpha=0.3, axis='y')

ax4 = axes[1, 1]
if 'filters' in hyperparam_impact:
    filters_data = hyperparam_impact['filters']
    filters = sorted([int(k) for k in filters_data.keys()])
    avg_accs = [sum(filters_data[str(f)]) / len(filters_data[str(f)]) for f in filters]
    ax4.bar([str(f) for f in filters], avg_accs, alpha=0.7, color='orange')
    ax4.set_xlabel('Number of Filters')
    ax4.set_ylabel('Average Accuracy (%)')
    ax4.set_title('Impact of Filter Count')
    ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('nas_results.png', dpi=150, bbox_inches='tight')
print("Saved visualization to 'nas_results.png'")
plt.show()