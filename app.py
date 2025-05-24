import sounddevice as sd
import numpy as np
import torch
from collections import deque
from model import TinyMelClassifier
from safetensors.torch import load_file
import tkinter as tk
from tkinter import ttk

# Parameters
SAMPLE_RATE = 44100  # 16kHz is common for speech/audio ML
CHANNELS = 1         # Mono
DURATION = 5         # seconds

RUN_NAME = "bumbling-deluge-19"
EPOCH = 91
# Buffer to hold the last 5 seconds of audio
buffer_size = SAMPLE_RATE * DURATION
audio_buffer = deque(maxlen=buffer_size*2)

# Define colors for different classes (add more colors as needed)
COLORS = {
    0: "#00FF00",  # Green
    1: "#FF0000",  # Red
    2: "#0000FF",  # Blue
}

CLASSES = {
    0: "Background",
    1: "Drone",
    2: "Helicopter?",
}

def update_color(class_id):
    color = COLORS.get(class_id, "#808080")  # Default to gray if class not found
    color_label.configure(bg=color)
    class_label.configure(text=f"Class: {CLASSES[class_id]}")

def audio_callback(indata, frames, time, status):
    # Flatten and append to buffer
    audio_buffer.extend(indata[:, 0])

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    root.title("Audio Classification Display")
    
    # Create a frame to hold the color display
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Create a label for the color display
    color_label = tk.Label(frame, width=120, height=40)  # 4x larger (30*4=120, 10*4=40)
    color_label.grid(row=0, column=0, padx=5, pady=5)
    
    # Create a label for the class number
    class_label = tk.Label(frame, text="Class: -", font=("Arial", 14))
    class_label.grid(row=1, column=0, padx=5, pady=5)
    
    # Initialize the model
    model = TinyMelClassifier()
    model.load_state_dict(load_file(f"model/{RUN_NAME}/model_{EPOCH}.safetensors"))
    model.eval()

    # Start the stream
    with sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, callback=audio_callback):
        print("Listening... Press Ctrl+C to stop.")
        try:
            while True:
                if len(audio_buffer) >= buffer_size:
                    # Convert buffer to numpy array, then to torch tensor
                    np_audio = np.array(audio_buffer, dtype=np.float32)
                    torch_audio = torch.from_numpy(np_audio)[-buffer_size:].unsqueeze(0).unsqueeze(0)
                    # torch_audio now contains the last 5 seconds of audio as a 1D tensor
                    # You can now use torch_audio for further processing
                    with torch.no_grad():
                        output = model(torch_audio)
                        class_id = torch.argmax(output, dim=1).item()
                        # Update the window with the new color
                        root.after(0, update_color, class_id)
                        print(class_id)
                
                # Update the tkinter window
                root.update()
                
        except KeyboardInterrupt:
            print("Stopped.")
            root.destroy()