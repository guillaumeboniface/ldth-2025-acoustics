from torch.utils.data import Dataset
import os
import torch
import torchaudio
from transformers import AutoProcessor
import torchaudio.transforms as T
from functools import lru_cache

def transform_wav(waveform, sample_rate=16000, n_fft=400, hop_length=200, n_mels=64):
    """
    Loads a WAV file and transforms it into a Mel spectrogram.

    Args:
        audio_path (str): Path to the WAV file.
        sample_rate (int): The sample rate to which the audio will be resampled.
                           UrbanSound8K dataset typically has varying sample rates,
                           so resampling to a consistent rate is common.
        n_fft (int): Number of FFT components.
        hop_length (int): The hop length for the STFT.
        n_mels (int): Number of Mel filterbanks.

    Returns:
        torch.Tensor: The Mel spectrogram.
    """

    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    mel_spectrogram = mel_spectrogram_transform(waveform)

    # (Optional) Convert to log scale (dB)
    log_mel_spectrogram = T.AmplitudeToDB()(mel_spectrogram)

    return log_mel_spectrogram

class LDTH2025Dataset(Dataset):
    def __init__(self, data_path: str, split: str = "train"):
        self.data_path = os.path.join(data_path, split)
        self.split = split
        self.classes = os.listdir(self.data_path)
        self.classes.sort()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.sample_rate = 24000
        self.dataset = []
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz", use_fast=True)
        for class_name in os.listdir(self.data_path):
            for file in os.listdir(os.path.join(self.data_path, class_name)):
                self.dataset.append((os.path.join(self.data_path, class_name, file), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.dataset)
    
    @lru_cache()
    def __getitem__(self, idx):
        audio_path, label = self.dataset[idx]
        audio, sr = torchaudio.load(audio_path, normalize=True)
        #resample audio to 24khz with torchaudio
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = resampler(audio)
        inputs = self.processor(raw_audio=audio.squeeze(), sampling_rate=self.sample_rate, return_tensors="pt")
        processed_audio = inputs["input_values"][0]
        padding_mask = inputs["padding_mask"][0]
        return processed_audio, padding_mask, torch.tensor(label)
    
class LDTH2025DatasetMel(Dataset):
    def __init__(self, data_path: str, split: str = "train"):
        self.data_path = os.path.join(data_path, split)
        self.split = split
        self.classes = os.listdir(self.data_path)
        self.classes.sort()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.sample_rate = 24000
        self.dataset = []
        for class_name in os.listdir(self.data_path):
            for file in os.listdir(os.path.join(self.data_path, class_name)):
                self.dataset.append((os.path.join(self.data_path, class_name, file), self.class_to_idx[class_name]))
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=self.sample_rate, n_fft=400, hop_length=200, n_mels=64)
        self.log_mel_spectrogram = T.AmplitudeToDB()

    def __len__(self):
        return len(self.dataset)
    
    @lru_cache()
    def __getitem__(self, idx):
        audio_path, label = self.dataset[idx]
        audio, sr = torchaudio.load(audio_path, normalize=True)
        #resample audio to 24khz with torchaudio
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = resampler(audio)
        mel_spectrogram = self.mel_spectrogram(audio)
        log_mel_spectrogram = self.log_mel_spectrogram(mel_spectrogram)
        return log_mel_spectrogram, torch.tensor(label)
    
if __name__ == "__main__":
    dataset = LDTH2025Dataset(data_path="data/raw")
    print(len(dataset))
    print(dataset[0])
    print("Audio: ", dataset[0][0].shape)
    print("Mask: ", dataset[0][1].shape)
    
    mel_dataset = LDTH2025DatasetMel(data_path="data/raw")
    print(len(mel_dataset))
    print(mel_dataset[0])
    print("Mel: ", mel_dataset[0][0].shape)
    
    
        