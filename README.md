# LDTH 2025 - Helsing Drone Acoustics

> Machine learning on drone acoustics for the London Defense Tech Hackathon, May 2025. Designed and run by Helsing.

## Challenge Prompt

Our AI problem is framed around detecting drones using acoustic data.
Automated detection of threats is essential in facilitating early warning and situational awareness.
Acoustic detection complements other sensor modalities; while radar, optical, and infrared sensors can also be
used for this problem, each has limitations such as weather and obstructions.
Given the low infrastructure costs and ability for rapid deployment, acoustic sensing presents a suitable additional
layer of surveillance for modern defense strategies.

The problem is split into two phases.

Phase 1: 3-class prediction. We provide a small curated dataset of open-source acoustic recordings split into three
categories: background, drone, and helicopter. The challenge is to train a model to separate these three class from
their acoustic signatures.

Phase 2: Enhanced prediction. Creating AI to use in the field is not just about model performance. We also need to
consider aspects such as inference time, edge support, and assurance. To this end, we ask contestants to explore the
ways they can enhance their approach for use in the field. This is intentionally left quite open-ended: we
want you to be creative! However, some suggestions include: analysing the interpretability/explainability of your
model, using as lightweight or as fast a model as possible (while maintaining predictive accuracy!), or creating new
synthetic data to explore what happens with really quiet contacts.

## Data

Sourced from: https://github.com/DroneDetectionThesis/Drone-detection-dataset (audio + video dataset)  
Paper: [A dataset for multi-sensor drone detection](https://www.sciencedirect.com/science/article/pii/S2352340921007976#!)