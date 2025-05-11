import streamlit as st
from safetensors.torch import load_file
from model import TinyMelClassifier
import torchaudio
import torch

st.title("Acoustic Classification")

@st.cache_resource()
def load_model(run_name: str, epoch: int):
    torch_model = TinyMelClassifier()
    torch_model.load_state_dict(load_file(f"model/{run_name}/model_{epoch}.safetensors"))
    torch_model.eval()
    return torch_model

model = load_model("revived-breeze-15", 100)

audio_value = st.audio_input("Record sound around you")

if audio_value is not None:
    torch_audio, sr = torchaudio.load(audio_value, normalize=True)
    print(sr)
    # resample to 44100
    torch_audio = torchaudio.functional.resample(torch_audio, sr, 44100)[:, :220500]
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=512, hop_length=256, n_mels=64, f_min=0, f_max=44100 / 2)(torch_audio)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel_spec = mel_spec.unsqueeze(0)
    output = model(mel_spec)
    label = torch.argmax(output, dim=1)
    st.write(["background", "drone", "helicopter"][label.item()])

