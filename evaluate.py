from sklearn.metrics import classification_report, confusion_matrix
from model import NoiseClassifier, MelClassifier, TinyMelClassifier
from dataset import LDTH2025Dataset, LDTH2025DatasetMel, LDTH2025DatasetRaw
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def scores(y_true, y_pred):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    
    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'confusion_matrix_{RUN_NAME}_epoch_{EPOCH}.png')
    plt.close()
    
    # Print metrics
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RUN_NAME = "desert-voice-8"
    EPOCH = 100

    model = MelClassifier().to(device)
    model.load_state_dict(load_file(f"model/{RUN_NAME}/model_{EPOCH}.safetensors"))
    model.to(device)
    model.eval()
    test_dataset = LDTH2025DatasetMel(data_path="data/raw", split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            mel, label = batch
            mel = mel.to(device)
            output = model(mel)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(output.argmax(dim=1).cpu().numpy())
    
    scores(y_true, y_pred)

    RUN_NAME = "likely-grass-9"
    EPOCH = 30

    model = NoiseClassifier().to(device)
    model.load_state_dict(load_file(f"model/{RUN_NAME}/model_{EPOCH}.safetensors"))
    model.to(device)
    model.eval()
    test_dataset = LDTH2025Dataset(data_path="data/raw", split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            audio, mask, label = batch
            audio = audio.to(device)
            mask = mask.to(device)
            output = model(audio, mask)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(output.argmax(dim=1).cpu().numpy())
    
    scores(y_true, y_pred)

    RUN_NAME = "stellar-vortex-13"
    EPOCH = 100

    model = TinyMelClassifier().to(device)
    model.load_state_dict(load_file(f"model/{RUN_NAME}/model_{EPOCH}.safetensors"))
    model.to(device)
    model.eval()
    test_dataset = LDTH2025DatasetRaw(data_path="data/raw", split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            mel, label = batch
            mel = mel.to(device)
            output = model(mel)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(output.argmax(dim=1).cpu().numpy())
    
    scores(y_true, y_pred)

