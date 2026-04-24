import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # Initialize gate scores to 0.5 (sigmoid ~ 0.62)
        nn.init.constant_(self.gate_scores, 0.5)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

class SelfPruningNet(nn.Module):
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_sparsity_loss(self):
        sparsity_loss = 0.0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores)
                sparsity_loss += torch.sum(gates)
        return sparsity_loss

    def get_sparsity_level(self, threshold=0.1):
        total_weights = 0
        pruned_weights = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores)
                total_weights += gates.numel()
                pruned_weights += torch.sum(gates < threshold).item()
        return (pruned_weights / total_weights) * 100.0 if total_weights > 0 else 0.0

    def get_all_gates(self):
        gates_list = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores).detach().cpu().numpy()
                gates_list.append(gates.flatten())
        return np.concatenate(gates_list)

def load_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def train_and_evaluate(lam, trainloader, testloader, device, epochs=5):
    print(f"\n--- Training with lambda = {lam} ---")
    model = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        cls_running_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            cls_loss = criterion(outputs, labels)
            sparsity_loss = model.get_sparsity_loss()
            
            total_loss = cls_loss + lam * (sparsity_loss / 1000.0)
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            cls_running_loss += cls_loss.item()
            
        print(f"Epoch {epoch+1} - Total Loss: {running_loss/len(trainloader):.4f} - Cls Loss: {cls_running_loss/len(trainloader):.4f} - Sparsity Level: {model.get_sparsity_level():.2f}%")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    sparsity = model.get_sparsity_level()
    print(f"Test Accuracy: {accuracy:.2f}%, Final Sparsity: {sparsity:.2f}%")
    return model, accuracy, sparsity

def main():
    parser = argparse.ArgumentParser(description="Train a Self-Pruning Neural Network")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for each lambda')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    trainloader, testloader = load_data(batch_size=args.batch_size)
    
    lambdas = [0.0, 0.01, 0.1, 1.0, 5.0] 
    results = {}
    best_model = None
    best_lam = None
    highest_sparsity_at_good_acc = 0
    
    for lam in lambdas:
        model, acc, sparsity = train_and_evaluate(lam, trainloader, testloader, device, epochs=args.epochs)
        results[lam] = {'accuracy': acc, 'sparsity': sparsity}
        
        # Determine "best" model as the one with high sparsity and decent accuracy
        if sparsity > highest_sparsity_at_good_acc and sparsity > 10.0:
            highest_sparsity_at_good_acc = sparsity
            best_model = model
            best_lam = lam
            
    # If no model got decent sparsity, just use the last one that had some sparsity
    if best_model is None:
        best_model = model
        best_lam = lam
        
    print("\\nFinal Results:")
    print("Lambda | Test Accuracy | Sparsity Level (%)")
    print("------------------------------------------")
    for lam in lambdas:
        print(f"{lam:<6} | {results[lam]['accuracy']:<13.2f} | {results[lam]['sparsity']:.2f}")

    # Plot distribution for the best model
    gates = best_model.get_all_gates()
    plt.figure(figsize=(10, 6))
    plt.hist(gates, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Gate Values Distribution (lambda={best_lam})')
    plt.xlabel('Gate Value (Sigmoid Output)')
    plt.ylabel('Frequency')
    plt.yscale('log') # Log scale helps see both peaks if there is an imbalance
    plt.savefig('gate_distribution.png')
    print("\\nSaved gate_distribution.png for best model.")

if __name__ == '__main__':
    main()
