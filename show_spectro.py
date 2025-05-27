from dataset import LDTH2025DatasetMel
import matplotlib.pyplot as plt
from tqdm import tqdm

dataset = LDTH2025DatasetMel(data_path="data/raw", split="train")

if __name__ == "__main__":
    for i in tqdm(range(len(dataset))):
        spectrogram, label = dataset[i]

        label_name = dataset.classes[label]
    
        # Convert to numpy and transpose to get time on x-axis
        spectrogram_np = spectrogram.squeeze(0).numpy()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram_np, aspect='auto', origin='lower')
        plt.colorbar(label='dB')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.title('Mel Spectrogram')
        plt.gca().set_facecolor('none')
        plt.gcf().set_facecolor('none')
        plt.tight_layout()
        plt.savefig(f"spectrograms/{label_name}/spectrogram_{i}.png", transparent=True)
        plt.close()