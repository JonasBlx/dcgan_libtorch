import argparse
import matplotlib.pyplot as plt
import torch

# Définir les arguments de la ligne de commande
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--sample-file", required=True, help="Path to the saved sample file")
parser.add_argument("-o", "--out-file", default="out.png", help="Output image file")
parser.add_argument("-d", "--dimension", type=int, default=3, help="Dimension of the grid (d x d)")
options = parser.parse_args()

# Charger les échantillons générés
module = torch.jit.load(options.sample_file)
samples = list(module.parameters())[0]

# Déterminer la taille de l'image en fonction des dimensions
grid_size = options.dimension
fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))

# Afficher les images dans une grille
for i in range(grid_size):
    for j in range(grid_size):
        index = i * grid_size + j
        if index < samples.size(0):
            image = samples[index].detach().cpu().squeeze()
            axes[i, j].imshow(image, cmap="gray")
            axes[i, j].axis('off')

# Sauvegarder l'image
plt.tight_layout()
plt.savefig(options.out_file)
print("Saved", options.out_file)
