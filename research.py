import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score

#research for question 5
# Define MLP
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# Define simple CNN
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # (28x28) -> (28x28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # -> (14x14)
            nn.Conv2d(32, 64, 3, padding=1), # -> (14x14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # -> (7x7)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64*7*7)
        return self.fc(x)

def train_model(model, train_loader, val_loader, device, epochs=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    acc = accuracy_score(all_labels, all_preds)
    return model, acc

def find_best_model(data_dir="Data", save_path="best_model.pth"):
    # Load and normalize data
    X = np.load(os.path.join(data_dir, "mnist_train_data.npy")).astype(np.float32) / 255.0
    y = np.load(os.path.join(data_dir, "mnist_train_labels.npy")).astype(np.int64)

    X = torch.tensor(X).unsqueeze(1)  # Add channel dimension
    y = torch.tensor(y)

    dataset = TensorDataset(X, y)

    # Train/val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        "MLP": MLPNet(),
        "CNN": CNNNet()
    }

    best_score = 0
    best_model_name = None
    best_model_instance = None

    for name, model in models.items():
        print(f"🚀 Training {name}...")
        trained_model, val_acc = train_model(model, train_loader, val_loader, device)
        print(f"✅ Validation accuracy for {name}: {val_acc:.4f}")

        if val_acc > best_score:
            best_score = val_acc
            best_model_name = name
            best_model_instance = trained_model

    # Save best model
    torch.save(best_model_instance.state_dict(), save_path)
    print(f"\n🏆 Best model: {best_model_name} with val acc: {best_score:.4f} → saved to: {save_path}")

    return {
        "model_name": best_model_name,
        "score": best_score,
        "path": save_path,
        "model": best_model_instance
    }
result = find_best_model(data_dir="Data", save_path="best_mnist_model.pth")
print(result)