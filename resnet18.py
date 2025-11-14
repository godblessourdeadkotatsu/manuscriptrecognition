'''
resnet50: funzionerà o NO?

chi lo sa!

this is the dark age of love
'''
import os
import numpy as np
from PIL import Image
import torch
from collections import Counter
from torchvision.transforms import v2
from torch.utils.data import random_split, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# svuoto la cache della GPU
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# sono un ragazzo semplice, voglio solo chiamare le epoche quando lancio il programma perchè ho visto gente farlo da htop
parser = argparse.ArgumentParser(description="Training resnet18 su grayscale")
parser.add_argument('--epochs', type=int, default=20, help='Numero di epoche da eseguire')

args = parser.parse_args()

num_epochs = args.epochs
print(f"Numero di epoche: {num_epochs}")

# creo la mia dataset class custom per una maggiore flessibilità :)
class ImageDataset(torch.utils.data.Dataset):
    
    def __init__(self, dir, transform=None):
        # nell'init metto il percorso e preparo la lista di percorsi di immagini e la lista di etichette
        self.data_dir = dir
        self.images = []
        self.labels = []
        self.transform = transform

        # per le classi mi baso sulle sottocartelle
        self.classes = sorted(os.listdir(dir))  
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}
        
        # ciclo tra le sottocartelle
        for cls_name in self.classes:
            cls_folder = os.path.join(dir, cls_name)
            for fname in sorted(os.listdir(cls_folder)):
                self.images.append(os.path.join(cls_folder, fname))
                self.labels.append(self.class_to_idx[cls_name])

    # la lunghezza di questo ImageDataset sarà ovviamente il numero di immagini
    def __len__(self):
        return len(self.images)

    # quando voglio ottenere l'elemento i dal dataset, apri l'immagine come PIL in grayscale
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("L") 

        # se le ho definite, applica le trasformazioni
        if self.transform:
            image = self.transform(image)

        image = image.repeat(3, 1, 1) #RESNET ASPETTA 3 CANALI  

        label = self.labels[index]
        
        return image, label
    

'''
passiamo ora a caricare le varie immagini.
quello che farò è usare questa classe ImageDataset per caricare tutta la cartella dentro un ImageDataset e applicare trasformazioni (l'ultima delle quali è la conversione a tensore)

Poi dividerò questa classe in due sottodataset (train e test) e li caricherò in dei dataloader. 
Siccome le tre classi hanno diverse numerosità userò un weighted sampler per equilibrarle.
'''

# come ben insegna il prof. Balbo, settiamo un seme
rng = torch.Generator().manual_seed(2025) 

# i nostri dati 
data_path = 'data/patches/gsc'

# le nostre trasformazioni preferite
transform_bundle = v2.Compose([
        v2.Resize((224,224)),
        v2.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
])

# carico dati e mi salvo la lunghezza e le etichette
dataset = ImageDataset(data_path, transform=transform_bundle)
dataset_size = len(dataset)
labels = dataset.labels

# ora lo divido on the fly in training (80%) e test (20%)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=rng)

# --- TRAIN SUBSET ---
train_labels_subset = [dataset.labels[i] for i in train_dataset.indices]

# conto quanti esempi per classe nel subset
train_counts_subset = Counter(train_labels_subset)

# calcolo i pesi inversi per classe
train_weights_per_class = {cls: 1.0 / count for cls, count in train_counts_subset.items()}

# assegno a ogni esempio il suo peso
train_sample_weights = [train_weights_per_class[label] for label in train_labels_subset]

# creo il WeightedRandomSampler
train_sampler = WeightedRandomSampler(
    weights=train_sample_weights,
    num_samples=len(train_sample_weights),
    replacement=True
)

# normalmente non serve sampler pesato per il test, basta shuffle=False
test_sampler = None

# carico tutto nel dataloader
dataloaders = {
    'train': torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=600,
        sampler=train_sampler,
        num_workers=16
    ),
    'test': torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16
    )
}

# mi assicuro di star usando la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'usiamo il dispositivo {device}')

# carico il modello e lo inizializzo
model = models.resnet50(weights="ResNet50_Weights.DEFAULT").to(device)

for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 3)).to(device) # perchè ho 3 classi
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

# definisco la funzione di training
def train_model(model, criterion, optimizer, dataloaders, num_epochs = 3):

    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-HL3soon-' * 10)

        for phase in ['train','test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(
                dataloaders[phase], 
                desc=f'fase di {phase}, epoca {epoch+1}',
                leave = False
                ):
                # porto etichette e tensori sulla GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # faccio il feedforward della rete con gli input e ottengo il vettore (tensore) con 3 probabilità assegnate alle 3 classi
                outputs = model(inputs)

                # calcolo la loss media per batch
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad() # calcolo gradiente
                    loss.backward() # backpropagation
                    optimizer.step() # step dell'ottimizzatore

                _, preds =torch.max(outputs, 1) # trova il valore massimo tra le classi

                # accumulo loss media x dimensione totale della batch per OGNI batch
                running_loss += loss.item() * inputs.size(0) 
                # faccio lo stesso accumulando quante volte ho azzeccato la previsione
                running_corrects += torch.sum(preds == labels)

            # stessa cosa ma con le epoche (salvo le loss e le accuracies per un grafico)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)

            print(f'in epoch {epoch} we have {phase} loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.4f}')
    
    return model, epoch_losses, epoch_accuracies

model_trained, train_losses, train_accuracies = train_model(model, criterion, optimizer, dataloaders, num_epochs)

# salvo lo stato del modello
torch.save(model_trained.state_dict(), 'pytorch/resnet50weights.pth')

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