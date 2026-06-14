from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from pet_model import CustomCNN, get_transfer_model
from pet_dataset import get_pet_dataloaders

# Carichiamo solo il test loader
_, test_loader = get_pet_dataloaders(batch_size=64, img_size=128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definiamo quali modelli vogliamo valutare
models_to_eval = [
    {"name": "Custom_CNN", "model": CustomCNN(), "file": "Custom_CNN.pth"},
    {"name": "Transfer_ResNet", "model": get_transfer_model(), "file": "Transfer_ResNet.pth"}
]

for item in models_to_eval:
    model = item["model"].to(device)
    model_path = Path(__file__).resolve().parent / item["file"]
    
    # Carica i pesi salvati
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds, all_labels = [], []
    correct, total = 0, 0

    print(f"Valutando: {item['name']}...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Mostra i risultati
    acc = 100 * correct / total
    print(f"Accuratezza {item['name']}: {acc:.2f}%")
    
    # Matrice di Confusione (come in laboratorio)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matrice: {item['name']} (Acc: {acc:.2f}%)")
    plt.show()