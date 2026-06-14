"""
Gestione del dataset PetImages: pulizia, trasformazioni e DataLoader.
"""
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from PIL import Image

def clean_corrupted_files(data_path):
    print("Pulizia file corrotti in corso...")
    num_skipped = 0
    for folder_name in ["Cat", "Dog"]:
        folder_path = data_path / folder_name
        if folder_path.exists():
            for fname in os.listdir(folder_path):
                fpath = folder_path / fname
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                except Exception:
                    num_skipped += 1
                    os.remove(fpath)
    print(f"Completato! Eliminati {num_skipped} file non validi.")

# Classe per applicare trasformazioni diverse ai subset di random_split
class PetDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_pet_dataloaders(batch_size=64, img_size=128):
    base_path = Path(__file__).resolve().parent
    data_path = base_path / "PetImages"
    
    clean_corrupted_files(data_path)

    # 1. TRASFORMAZIONI DI TRAINING (con Data Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),    # Riflessione speculare
        transforms.RandomRotation(degrees=10),     # Leggera rotazione
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. TRASFORMAZIONI DI TEST (Solo Resize, Tensor e Normalize)
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    full_dataset = datasets.ImageFolder(root=str(data_path))

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_subset, test_subset = random_split(
    full_dataset, 
    [train_size, test_size], 
    generator=torch.Generator().manual_seed(42) # Garantisce lo stesso identico split ogni volta
)

    train_dataset = PetDataset(train_subset, transform=train_transform)
    test_dataset = PetDataset(test_subset, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader