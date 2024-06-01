import torchvision
import torchvision.transforms as transforms

# Définir les transformations des données
transform = transforms.Compose([
    transforms.ToTensor(),  # Convertit les images en tensors PyTorch
    transforms.Normalize((0.1307,), (0.3081,))  # Normalise les données MNIST
])

# Télécharger et charger le dataset d'entraînement
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Télécharger et charger le dataset de test
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
