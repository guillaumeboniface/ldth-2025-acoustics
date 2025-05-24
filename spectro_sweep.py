from model import TinyMelClassifier
from dataset import LDTH2025DatasetRaw
from torch.utils.data import DataLoader
import torch
import wandb

project = "ldth-2025-acoustics-sweep"

sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "n_fft": {"values": [200, 400, 800, 1600]},
        "n_mels": {"values": [32, 64, 128]},
    },
}

def main():
    torch.manual_seed(0)

    batch_size = 16
    epochs = 100
    learning_rate = 1e-4

    run = wandb.init(project=project, config={
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate
    })

    n_fft = run.config.n_fft
    hop_length = n_fft // 2
    n_mels = run.config.n_mels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMelClassifier(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels).to(device)
    train_dataset = LDTH2025DatasetRaw(data_path="data/raw", split="train")
    test_dataset = LDTH2025DatasetRaw(data_path="data/raw", split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    
    for epoch in range(epochs):
        epoch_loss = []
        epoch_test_loss = []
        epoch_accuracy = []
        for i, batch in enumerate(train_loader):
            mel, label = batch
            mel = mel.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(mel)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            
        epoch_loss = sum(epoch_loss) / len(epoch_loss)

        for i, batch in enumerate(test_loader):
            mel, label = batch
            mel = mel.to(device)
            label = label.to(device)
            output = model(mel)
            loss = criterion(output, label)
            accuracy = (output.argmax(dim=1) == label).float().mean().item()
            epoch_test_loss.append(loss.item())
            epoch_accuracy.append(accuracy)

        epoch_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
        epoch_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)

        wandb.log({
            "loss": epoch_loss,
            "test_loss": epoch_test_loss,
            "accuracy": epoch_accuracy
        })

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

    # Start the sweep job
    wandb.agent(sweep_id, function=main, count=4)
