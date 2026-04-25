# ALS Digital Twin

A machine learning system for creating digital twins of ALS patients using longitudinal clinical data.

## Overview

This project implements a digital twin model for Amyotrophic Lateral Sclerosis (ALS) patients that:

1. **Predicts next clinical state** - ALSFRS-R total and domain scores
2. **Predicts time-to-event** - Death, gastrostomy, NIV, wheelchair use, speech loss
3. **Updates dynamically** - Absorbs new visit data to refine predictions

## Architecture

The system uses a staged portfolio approach:

### Classical Baselines
- **Gradient Boosting** for next-state prediction
- **Random Survival Forest** for event prediction (when scikit-survival is available)

### Deep Learning Model
- **GRU-D encoder** for temporal sequences with missing data handling
- **Static encoder** for baseline features
- **Multi-task output heads** for state and survival prediction

## Data Requirements

The model expects the ALS Natural History dataset with:
- Demographics
- ALSFRS-R assessments
- ALS Diagnosis with UMN/LMN findings
- Assistive Devices Log
- Non-Invasive Ventilation Log
- Feeding Tube Placement records
- Mortality data

## Installation

```bash
pip install torch numpy pandas scikit-learn
# Optional for survival models
pip install scikit-survival xgboost
```

## Usage

### 1. Clean the data

```bash
python clean_data.py
```

This creates processed datasets in `./cleaned_data/`:
- `training_landmarks.csv` - Main training dataset
- `patients_static.csv` - Static patient features
- `patient_events.csv` - Event outcomes
- `visits_longitudinal.csv` - Longitudinal visits

### 2. Train the models

```bash
python train_digital_twin.py
```

This trains both classical baselines and the deep learning model, saving to `./trained_models/`.

### 3. Run inference

```bash
python inference.py
```

## Project Structure

```
.
├── clean_data.py           # Data cleaning pipeline
├── train_digital_twin.py   # Training script
├── inference.py            # Inference script
├── models/
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── digital_twin.py     # Deep learning model
│   └── classical_baselines.py  # Classical ML models
├── data/                   # Raw data (not in repo)
├── cleaned_data/           # Processed data (not in repo)
└── trained_models/         # Trained models (not in repo)
```

## Model Details

### Features

**Static Features:**
- Age at diagnosis, sex, race, ethnicity
- El Escorial classification
- UMN/LMN burden scores

**Temporal Features:**
- ALSFRS-R total and domain scores
- Individual ALSFRS-R items
- Time since diagnosis
- Visit timing features

### Targets

**State Prediction:**
- Next visit ALSFRS-R total
- Domain scores: Bulbar, Fine Motor, Gross Motor, Walking, Respiratory

**Event Prediction:**
- Time to death
- Time to gastrostomy
- Time to NIV
- Time to wheelchair use
- Time to speech loss

## License

Research use only.
