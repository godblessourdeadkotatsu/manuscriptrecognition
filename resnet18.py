import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


# --- DEFINIZIONE DEGLI ARGOMENTI ---
parser = argparse.ArgumentParser(description="Training resnet18 su grayscale")
parser.add_argument('--epochs', type=int, default=20, help='Numero di epoche da eseguire')

args = parser.parse_args()

num_epochs = args.epochs
print(f"Numero di epoche: {num_epochs}")

# Creating a custom dataset class
class ImageDataset(torch.utils.data.Dataset):
    
    def __init__(self, dir, transform=None):
        self.data_dir = dir
        self.images = []
        self.labels = []
        self.transform = transform

        # classi : mi baso sulle sottocartelle
        self.classes = sorted(os.listdir(dir))  
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_folder = os.path.join(dir, cls_name)
            for fname in os.listdir(cls_folder):
                self.images.append(os.path.join(cls_folder, fname))
                self.labels.append(self.class_to_idx[cls_name])

    # Defining the length of the dataset
    def __len__(self):
        return len(self.images)

    # Defining the method to get an item from the dataset
    def __getitem__(self, index):
        image_np = np.array(Image.open(self.images[index])) # forza grayscale
        image = torch.from_numpy(image_np).unsqueeze(0) / 255.0

        # Applying the transform
        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        
        return image, label
    
class SaltAndPepper(object):

    def __init__(self, generator, amount=0.05):
        self.amount = amount
        self.generator = generator

    #definisco la trasformazione

    def __call__(self, image):

        #dimensioni
        if image.ndim == 3:
            _, h, w = image.shape
        else:
            h, w = image.shape


        number_of_pixels = int(h * w * self.amount)

        # Pick a random y coordinate
        y_coord=torch.randint(0, h, (number_of_pixels,), generator=self.generator)
        
        # Pick a random x coordinate
        x_coord=torch.randint(0, w, (number_of_pixels,), generator=self.generator)
        
        # Color that pixel to white
        image[:, y_coord, x_coord] = 1.0
            
        # Randomly pick some pixels in
        # the image for coloring them black
        # Pick a random number between 300 and 10000 
        
        # Pick a random y coordinate
        y_coord=torch.randint(0, h, (number_of_pixels,), generator=self.generator)
        
        # Pick a random x coordinate
        x_coord=torch.randint(0, w, (number_of_pixels,), generator=self.generator)
        
        # Color that pixel to black
        image[:, y_coord, x_coord] = 0.0

        return image

rng = torch.Generator().manual_seed(2025)  # crea un generatore

# Replace the path with the path to your dataset
data_path = 'data/patches/gsc'

my_transform = v2.Compose([
    v2.RandAugment(2,4)
])

# Creating a dataset object with the path to the dataset
dataset = ImageDataset(data_path, transform=my_transform)

# Getting the length of the dataset
dataset_length = len(dataset)

# Printing the length of the dataset
print('Number of training examples:',dataset_length)

dataset = ImageDataset(data_path, transform=my_transform)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, val_size], generator=rng)

train_dataloader = torch.utils.data.DataLoader(
    dataset = train_dataset,

    batch_size = 200,

    shuffle = True,

    num_workers= 2 
)

print('Number of batches:',len(train_dataloader))

test_dataloader = torch.utils.data.DataLoader(
    dataset = test_dataset,

    batch_size = 200,

    shuffle = False,

    num_workers= 2 
)

print('Number of batches:',len(test_dataloader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === MODEL ===
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Adattiamo la prima conv per grayscale
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Adattiamo il classificatore finale per 3 classi
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

model = model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# === TRAINING SETUP ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# preparo la merda per il grafico
train_losses = []
train_accuracies = []
# === LOOP ===

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(train_dataloader)

    for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    epoch_loss = running_loss / total_batches
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {running_loss/len(train_dataloader):.4f}  Acc: {train_acc:.2f}%")

print("Training completed.")

model_path = "pytorch/checkpoints/resnet18_full.pth"

torch.save(model, model_path)

# Salva il grafico
plt.figure(figsize=(10,6))
plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Loss')
plt.plot(range(1, num_epochs+1), train_accuracies, marker='s', label='Accuracy')
plt.title("Training Metrics per Epoca")
plt.xlabel("Epoca")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.savefig("training_metrics.png")  # <- Salva il PNG
plt.close()  # chiude la figura per liberare memoria