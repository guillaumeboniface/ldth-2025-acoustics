import sounddevice as sd
import numpy as np
import torch
from collections import deque

# Parameters
SAMPLE_RATE = 16000  # 16kHz is common for speech/audio ML
CHANNELS = 1         # Mono
DURATION = 5         # seconds

# Buffer to hold the last 5 seconds of audio
buffer_size = SAMPLE_RATE * DURATION
audio_buffer = deque(maxlen=buffer_size)

def audio_callback(indata, frames, time, status):
    # Flatten and append to buffer
    audio_buffer.extend(indata[:, 0])

# Start the stream
with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, callback=audio_callback):
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            if len(audio_buffer) == buffer_size:
                # Convert buffer to numpy array, then to torch tensor
                np_audio = np.array(audio_buffer, dtype=np.float32)
                torch_audio = torch.from_numpy(np_audio)
                # torch_audio now contains the last 5 seconds of audio as a 1D tensor
                # You can now use torch_audio for further processing
    except KeyboardInterrupt:
        print("Stopped.")