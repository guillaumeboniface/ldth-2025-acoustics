from model import TinyMelClassifier
from dataset import LDTH2025DatasetRaw
from torch.utils.data import DataLoader
import torch
import itertools
import wandb
from safetensors.torch import save_file
import os

if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMelClassifier().to(device)
    train_dataset = LDTH2025DatasetRaw(data_path="data/raw", split="train")
    test_dataset = LDTH2025DatasetRaw(data_path="data/raw", split="test")

    batch_size = 16
    epochs = 100
    learning_rate = 1e-4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    wandb.init(project="ldth-2025-acoustics", config={
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "model": str(model)
    })
    for epoch in range(epochs):
        epoch_loss = []
        epoch_test_loss = []
        epoch_accuracy = []
        for i, (batch, test_batch) in enumerate(zip(train_loader, itertools.cycle(test_loader))):
            mel, label = batch
            mel = mel.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(mel)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                test_mel, test_label = test_batch
                test_mel = test_mel.to(device)
                test_label = test_label.to(device)
                test_output = model(test_mel)
                test_loss = criterion(test_output, test_label)
                accuracy = (test_output.argmax(dim=1) == test_label).float().mean().item()
            epoch_loss.append(loss.item())
            epoch_test_loss.append(test_loss.item())
            epoch_accuracy.append(accuracy)
            print(f"\r[Epoch {epoch+1}/{epochs}], [Batch {i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy:.4f}", end="")
        
            wandb.log({
                "loss": loss.item(),
                "test_loss": test_loss.item(),
                "accuracy": accuracy
            })
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
        epoch_accuracy = sum(epoch_accuracy) / len(epoch_accuracy)
        print(f"\r[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Accuracy: {epoch_accuracy:.4f}                                       ")
        model_path = os.path.join("model", wandb.run.name, f"model_{epoch+1}.safetensors")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_file(model.state_dict(), model_path)