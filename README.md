# LDTH 2025 - Helsing Drone Acoustics

> Machine learning on drone acoustics for the London Defense Tech Hackathon, May 2025. Designed and run by Helsing.

## Data

Sourced from: https://github.com/DroneDetectionThesis/Drone-detection-dataset (audio + video dataset)  
Paper: [A dataset for multi-sensor drone detection](https://www.sciencedirect.com/science/article/pii/S2352340921007976#!)

# Running the repo
1. Load the data
> uv run src.ldth_drone_acoustics.setup.download_raw_data

2. Log in Wandb
3. Export your huggingface token
> export HF_TOKEN=<YOUR_HF_TOKEN>

4. Train one of the models
> uv run train_tiny_mel.py

5. Update <strong>app.py</strong> with your run name and preferred epoch
6. Run the app
> uv run app.py  