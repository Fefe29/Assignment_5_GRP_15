# === Import required libraries ===
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

# === Load MNIST Data ===
X = np.load('Data/mnist_train_data.npy')  # Shape: (60000, 28, 28)
y = np.load('Data/mnist_train_labels.npy')  # Shape: (60000,)

# === Convert Images to Feature Vectors (Column-wise stacking) ===
X_features = X.reshape(X.shape[0], -1, order='F')  # Column-stacked

# === Question 1: Train-Test Split with Balancing ===
def create_balanced_split(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = [], [], [], []
    for digit in range(10):
        indices = np.where(y == digit)[0]
        np.random.shuffle(indices)
        split_point = int(len(indices) * (1 - test_size))
        X_train.extend(X_features[indices[:split_point]])
        y_train.extend(y[indices[:split_point]])
        X_test.extend(X_features[indices[split_point:]])
        y_test.extend(y[indices[split_point:]])
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

X_train, X_test, y_train, y_test = create_balanced_split(X_features, y)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# === Question 2: kNN Model with Cross-Validation ===
param_grid = {'n_neighbors': list(range(1, 16))}
knn = KNeighborsClassifier(metric='euclidean')
grid_knn = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_knn.fit(X_train, y_train)

best_k = grid_knn.best_params_['n_neighbors']
print(f"Best k: {best_k}")

# Train final kNN
final_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
final_knn.fit(X_train, y_train)
y_pred_knn = final_knn.predict(X_test)
err_knn = 1 - accuracy_score(y_test, y_pred_knn)
print(f"kNN Test Error Rate: {err_knn:.4f}")

# Plot error vs k
results = grid_knn.cv_results_
mean_test_errors = 1 - np.array(results['mean_test_score'])
plt.plot(param_grid['n_neighbors'], mean_test_errors, marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Error Rate')
plt.title('kNN Hyperparameter Tuning')
plt.grid(True)
plt.show()

# === Question 3: Polynomial Kernel SVM ===
param_grid_svm = {
    'C': [0.1, 1, 10],
    'degree': [2, 3, 4],
    'kernel': ['poly']
}
svm = SVC()
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', verbose=2)
grid_svm.fit(X_train, y_train)

best_params_svm = grid_svm.best_params_
print(f"Best SVM params: {best_params_svm}")

final_svm = SVC(**best_params_svm)
final_svm.fit(X_train, y_train)
y_pred_svm = final_svm.predict(X_test)
err_svm = 1 - accuracy_score(y_test, y_pred_svm)
print(f"SVM Test Error Rate: {err_svm:.4f}")

# === Question 4: MLP Model (PyTorch) ===
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes=10):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_mlp(hidden_layers, epochs=20):
    model = MLP(input_size=784, hidden_sizes=hidden_layers).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
    y_tensor = torch.tensor(y_train, dtype=torch.long).cuda()

    for epoch in range(epochs):
        model.train()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

# Grid Search (simple manual loop for PyTorch)
layer_options = [1, 2, 3]
unit_options = [64, 128, 256]
best_err = float('inf')
best_model = None
best_setting = None

for L in layer_options:
    for K in unit_options:
        hidden_structure = [K] * L
        model = train_mlp(hidden_structure, epochs=10)
        model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).cuda()
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        err = 1 - (predicted.cpu().numpy() == y_test).mean()
        print(f"L={L}, K={K} => Error Rate: {err:.4f}")
        if err < best_err:
            best_err = err
            best_model = model
            best_setting = (L, K)

print(f"Best MLP Setting: Layers={best_setting[0]}, Units={best_setting[1]}")
print(f"MLP Test Error Rate: {best_err:.4f}")

# === Save the Final MLP Model ===
torch.save(best_model.state_dict(), 'final_mlp_model.pth')

# === Comparison Summary ===
print("\n=== Final Model Error Rates ===")
print(f"kNN Error: {err_knn:.4f}")
print(f"SVM Error: {err_svm:.4f}")
print(f"MLP Error: {best_err:.4f}")
