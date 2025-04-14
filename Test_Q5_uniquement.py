import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# Même structure que ton modèle sauvegardé
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 64 * 7 * 7)
        return self.fc(x)


def classifyHandwrittenDigits(Xtest, data_dir, model_path):
    """
    Returns a vector of predictions with elements "0", "1", ..., "9",
    corresponding to each of the N_test test images in Xtest

    Parameters:
    - Xtest: N_test x 28 x 28 numpy array (test images)
    - data_dir: path to folder containing training data (not used here)
    - model_path: path to .pth model file

    Returns:
    - ytest: numpy array of shape (N_test,) with predicted labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing: normalize and add channel dimension
    Xtest = Xtest.astype(np.float32) / 255.0
    Xtest_tensor = torch.tensor(Xtest).unsqueeze(1).to(device)  # shape: (N, 1, 28, 28)

    # Load model
    model = CNNNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        outputs = model(Xtest_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    return predictions




Xtest = np.load("Data/mnist_train_data.npy")  # si dispo
Xtest_labels = np.load("Data/mnist_train_labels.npy")  # si dispo




# # Afficher les premières lignes (ou éléments) du tableau
# print(Xtest)

# # Voir les dimensions du tableau
# print(Xtest.shape)

# # Obtenir des informations sur le type de données
# print(Xtest.dtype)

# print(Xtest[:10])

ytest = classifyHandwrittenDigits(Xtest, data_dir="Data", model_path="best_mnist_model.pth")
print(ytest)  # Affiche les 10 premières prédictions
# Calculer l'accuracy en comparant les prédictions avec les labels réels
accuracy = accuracy_score(Xtest_labels, ytest)  # Compare les 10 premiers labels
print(f"Accuracy on test set: {accuracy:.4f}")
