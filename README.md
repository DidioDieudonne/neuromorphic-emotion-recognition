# SEW-ResNet18 for Facial Emotion Recognition

A comparative study implementation of Spiking Neural Networks (SNN) versus Artificial Neural Networks (ANN) for facial emotion recognition using the FER2013 dataset.

## Overview

This repository implements SEW-ResNet18, a spiking neural network architecture, alongside a traditional ResNet18 baseline for fair performance comparison on emotion recognition tasks. The implementation provides robust SpikingJelly integration with automatic fallback mechanisms.

## Architecture

- **SNN Model**: SEW-ResNet18 with temporal processing (8 timesteps)
- **ANN Baseline**: ResNet18 adapted for grayscale emotion images
- **Parameter Balance**: ~11M parameters for both models (fair comparison)
- **Input**: 48×48 grayscale emotion images
- **Output**: 7 emotion classes (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)

## Key Features

- **Robust Implementation**: Automatic fallback from SpikingJelly to standard ResNet18
- **Temporal Processing**: Single-step mode with manual temporal loops
- **Energy Monitoring**: GPU power consumption tracking (optional)
- **Scientific Rigor**: Balanced parameter count for valid comparison
- **Comprehensive Logging**: Detailed training metrics and visualizations


## Installation

```bash
git clone https://github.com/DidioDieudonne/neuromorphic-emotion-recognition.git
cd sew-resnet18-emotion-recognition
pip install -r requirements.txt
```

## Usage

### Quick Test
```python
from snn_model import create_sew_resnet18, test_sew_resnet

# Test model creation and functionality
success = test_sew_resnet()

# Create model for training
model = create_sew_resnet18(num_classes=7, num_timesteps=8)
```

### Training
```python
# Configure dataset path in notebook
dataset_path = "path/to/fer2013/data"

# Run comparative training (see notebook for full pipeline)
python training_notebook.py
```

## Dataset Structure

```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## Results

The implementation provides comprehensive comparison metrics:
- **Accuracy**: Validation accuracy for both SNN and ANN
- **Energy Efficiency**: Power consumption analysis (if monitoring enabled)
- **Training Dynamics**: Loss/accuracy curves and convergence analysis
- **Confusion Matrices**: Per-class performance visualization




git config --global user.name "DidioDieudonne"
git config --global user.email "dieudobyaombemw@gmail.com"



Dbm@viet9971


pkexec env DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY "/usr/local/Visual_Paradigm_CE_18.0/bin/Visual_Paradigm"




Spécialisation "DSA In Python + Top 130 Leetcode Problems for MAANG"


Madame, Monsieur,

Je vous adresse ma candidature pour le sujet de thèse intitulé «Quantification de l’incertitude prédictive, 
fondée sur la théorie de l’évidence, appliquée à l’estimation des traits de vie des poissons à partir 
d’images d’otolithes 3D ».

Vous trouverez en pièce jointe un fichier PDF unique contenant mon CV, ma lettre de motivation ainsi que mes relevés de notes.

Je reste à votre disposition pour toute information complémentaire.

Cordialement,
Dieudonné M. Byaombe

