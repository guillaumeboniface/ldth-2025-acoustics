from transformers import EncodecModel
from torch import nn
import torch
from torchaudio import transforms as T
import math

class NoiseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.batch_norm = nn.BatchNorm1d(375 * 2)
        self.fc = nn.Linear(375 * 2 * 1200, 3)

    def forward(self, audio, mask):
        x = self.encoder(audio, mask)
        x = nn.functional.one_hot(x["audio_codes"], num_classes=1200).to(torch.float32)
        x = x.view(audio.shape[0], -1)
        x = self.fc(x)
        return x
    
class MelClassifier(nn.Module):
    def __init__(self, height=64, width=601):
        super().__init__()
        self.height = height
        self.width = width
        conv_dim = (height // 2) * (width // 2) * 8
        fc_dim = 32
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.batch_norm = nn.BatchNorm1d(conv_dim)
        self.attention = torch.nn.MultiheadAttention(embed_dim=fc_dim, num_heads=8, batch_first=True)
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3)
        )

    def forward(self, mel):
        x = self.conv_layer(mel)
        x = x.transpose(2, 3)
        x = x.reshape(mel.shape[0], -1)
        x = self.batch_norm(x)
        x = x.reshape(mel.shape[0], -1, 32)
        attn_output, attn_output_weights = self.attention(x, x, x)
        x = attn_output.mean(dim=1)
        x = self.fc_layer(x)
        return x

class TinyMelClassifier(nn.Module):
    def __init__(self, audio_length=220500, n_fft=400, hop_length=200, n_mels=64):
        super().__init__()
        self.height = n_mels
        self.width = math.ceil(audio_length / hop_length)
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=44100, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.log_mel_spectrogram = T.AmplitudeToDB()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * (self.height // 32) * (self.width // 32), 3)
        )
    
    def forward(self, audio):
        x = self.mel_spectrogram(audio)
        x = self.log_mel_spectrogram(x)
        x = self.conv_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layer(x)
        return x
    
    
if __name__ == "__main__":
    # model = NoiseClassifier()
    # dummy_audio = torch.randn(8, 1, 120000)
    # dummy_mask = torch.ones(8, 1, 120000)
    # print(model(dummy_audio, dummy_mask).shape)

    # model = MelClassifier()
    # dummy_mel = torch.randn(8, 1, 64, 601)
    # print(torch.tensor([param.numel() for param in model.parameters()]).sum())
    # print(model(dummy_mel).shape)

    model = TinyMelClassifier()
    dummy_mel = torch.randn(8, 1, 220500)
    print(torch.tensor([param.numel() for param in model.parameters()]).sum())
    print(model(dummy_mel).shape)

    model = TinyMelClassifier(n_fft=2048, hop_length=1024, n_mels=128)
    dummy_mel = torch.randn(8, 1, 220500)
    print(torch.tensor([param.numel() for param in model.parameters()]).sum())
    print(model(dummy_mel).shape)