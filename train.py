from pathlib import Path
from time import time
import torch
import torch.nn as nn
import torch.optim as optim

from pet_model import CustomCNN, get_transfer_model
from pet_dataset import get_pet_dataloaders

# SETUP 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_pet_dataloaders(batch_size=64, img_size=128)

models_to_train = [
    {"name": "Custom_CNN", "model": CustomCNN()},
    {"name": "Transfer_ResNet", "model": get_transfer_model()}
]

for item in models_to_train:
    model_name = item["name"]
    model = item["model"].to(device)
    
    print(f"\n>>> Inizio Training: {model_name} su {device}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCELoss()
    epochs = 5
    
    start_time = time()
    for epoch in range(epochs):
        
        # FASE DI TRAINING 
        model.train() # Imposta il modello in modalità addestramento
        epoch_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # FASE DI VALUTAZIONE (Monitoraggio sul Test Set)
        model.eval() # Imposta il modello in modalità valutazione (disabilita dropout, batchnorm, ecc.) 
        correct, total = 0, 0
        
        with torch.no_grad(): # Non calcolo i gradienti (risparmia RAM e CPU/GPU)
            for test_images, test_labels in test_loader:
                test_images = test_images.to(device)
                test_labels = test_labels.to(device).float().unsqueeze(1)
                
                test_outputs = model(test_images)
                predicted = (test_outputs > 0.5).float()
                
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
                
        test_acc = 100 * correct / total
        
        print(f"[{model_name}] Epoca {epoch+1}/{epochs} completata. "
              f"Train Loss: {avg_train_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    # Salvataggio
    save_path = Path(__file__).resolve().parent / f"{model_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Modello {model_name} salvato in: {save_path}")
    print(f"Tempo totale per {model_name}: {time() - start_time:.2f}s")