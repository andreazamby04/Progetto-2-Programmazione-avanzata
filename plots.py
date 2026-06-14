import matplotlib.pyplot as plt

# Numero di epoche
epochs = [1, 2, 3, 4, 5]

# --- INSERISCI QUI I TUOI VALORI PRECISI DI LOSS SE DIVERSI ---
custom_loss = [0.6176, 0.5008, 0.4381, 0.3940, 0.3593]  # Sostituisci con la tua Train Loss reale
resnet_loss = [0.3015, 0.2343, 0.2233, 0.2191, 0.2178] # Sostituisci con la tua Train Loss reale

# Accuratezze corrette dal tuo log
custom_acc = [74.16, 77.72, 79.34, 82.70, 83.82] # Epoca 1: 75.24% -> Epoca 5: 84.16%
resnet_acc = [91.76, 92.60, 92.64, 92.58, 92.64] # Epoca 1: 94.88% -> Epoca 5: 95.42%

# Configurazione della figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. Grafico della Loss
ax1.plot(epochs, custom_loss, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Custom CNN')
ax1.plot(epochs, resnet_loss, marker='s', linestyle='--', color='#ff7f0e', linewidth=2, label='Transfer ResNet18')
ax1.set_title('Andamento della Training Loss', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoca', fontsize=10)
ax1.set_ylabel('Loss', fontsize=10)
ax1.set_xticks(epochs)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=10)

# 2. Grafico della Test Accuracy
ax2.plot(epochs, custom_acc, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Custom CNN')
ax2.plot(epochs, resnet_acc, marker='s', linestyle='--', color='#ff7f0e', linewidth=2, label='Transfer ResNet18')
ax2.set_title('Andamento della Test Accuracy', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoca', fontsize=10)
ax2.set_ylabel('Accuracy (%)', fontsize=10)
ax2.set_xticks(epochs)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('training_trends.png', dpi=300)
plt.show()