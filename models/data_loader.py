"""
Data loading and preprocessing for Digital Twin training.
Handles train/val/test splits and creates PyTorch datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings('ignore')


# Feature groups for the model
STATIC_FEATURES = [
    'age_at_diagnosis', 'is_female', 'is_hispanic',
    'race_white', 'race_black', 'race_asian', 'race_other',
    'el_escorial', 'umn_burden', 'lmn_burden', 'emg_burden'
]

TEMPORAL_FEATURES = [
    'alsfrs_total', 'domain_bulbar', 'domain_fine_motor',
    'domain_gross_motor', 'domain_walking', 'domain_respiratory',
    'pct_bulbar', 'pct_fine_motor', 'pct_gross_motor',
    'pct_walking', 'pct_respiratory', 'pct_total',
    'months_since_diagnosis', 'days_since_prev_visit'
]

# Individual ALSFRS-R items
ALSFRS_ITEMS = [
    'alsfrs1', 'alsfrs2', 'alsfrs3', 'alsfrs4', 'alsfrs5',
    'alsfrs6', 'alsfrs7', 'alsfrs8', 'alsfrs9',
    'alsfrsr1', 'alsfrsr2', 'alsfrsr3'
]

# Targets for next-state prediction
STATE_TARGETS = [
    'alsfrs_total', 'domain_bulbar', 'domain_fine_motor',
    'domain_gross_motor', 'domain_walking', 'domain_respiratory'
]

# Event targets (time-to-event in days)
EVENT_TARGETS = [
    'days_to_death', 'days_to_gastrostomy', 'days_to_niv',
    'days_to_wheelchair', 'days_to_speech_loss'
]

EVENT_INDICATORS = [
    'future_death', 'future_gastrostomy', 'future_niv',
    'future_wheelchair', 'future_speech_loss'
]


def load_data(data_dir='./cleaned_data'):
    """Load the cleaned training data."""
    data_dir = Path(data_dir)

    # Load main training dataset
    df = pd.read_csv(data_dir / 'training_landmarks.csv')

    # Parse dates
    date_cols = ['assessment_date', 'diagnosis_date', 'prev_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df


def prepare_features(df, static_features=None, temporal_features=None):
    """Prepare features with proper handling of missing values."""

    if static_features is None:
        static_features = STATIC_FEATURES
    if temporal_features is None:
        temporal_features = TEMPORAL_FEATURES + ALSFRS_ITEMS

    df = df.copy()

    # Fill missing static features with median/mode
    for col in static_features:
        if col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 0)

    # For temporal features, we'll handle missing differently (masking)
    # Fill with -1 as indicator, create mask later
    for col in temporal_features:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(0)

    # Fill days_since_prev_visit for first visits
    if 'days_since_prev_visit' in df.columns:
        df['days_since_prev_visit'] = df['days_since_prev_visit'].fillna(0)

    return df


def create_patient_splits(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Create train/val/test splits grouped by patient.
    Uses time-aware splitting where possible.
    """

    # Get unique patients
    patients = df['SubjectUID'].unique()

    # First split: train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(df, groups=df['SubjectUID']))

    df_train_val = df.iloc[train_val_idx]
    df_test = df.iloc[test_idx]

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(gss2.split(df_train_val, groups=df_train_val['SubjectUID']))

    df_train = df_train_val.iloc[train_idx]
    df_val = df_train_val.iloc[val_idx]

    print(f"Train: {len(df_train)} visits, {df_train['SubjectUID'].nunique()} patients")
    print(f"Val: {len(df_val)} visits, {df_val['SubjectUID'].nunique()} patients")
    print(f"Test: {len(df_test)} visits, {df_test['SubjectUID'].nunique()} patients")

    return df_train, df_val, df_test


class ALSDataset(Dataset):
    """
    Dataset for ALS Digital Twin training.
    Each sample is a patient's visit history up to a landmark point.
    """

    def __init__(self, df, static_features=None, temporal_features=None,
                 state_targets=None, event_targets=None, event_indicators=None,
                 scaler_static=None, scaler_temporal=None, fit_scalers=False,
                 max_seq_len=50):

        self.static_features = static_features or STATIC_FEATURES
        self.temporal_features = temporal_features or (TEMPORAL_FEATURES + ALSFRS_ITEMS)
        self.state_targets = state_targets or STATE_TARGETS
        self.event_targets = event_targets or EVENT_TARGETS
        self.event_indicators = event_indicators or EVENT_INDICATORS
        self.max_seq_len = max_seq_len

        # Prepare features
        self.df = prepare_features(df, self.static_features, self.temporal_features)

        # Get available features (those that exist in df)
        self.static_features = [f for f in self.static_features if f in self.df.columns]
        self.temporal_features = [f for f in self.temporal_features if f in self.df.columns]

        # Fit or use provided scalers
        if fit_scalers:
            self.scaler_static = StandardScaler()
            self.scaler_temporal = StandardScaler()

            static_data = self.df[self.static_features].values
            self.scaler_static.fit(static_data)

            temporal_data = self.df[self.temporal_features].values
            self.scaler_temporal.fit(temporal_data)
        else:
            self.scaler_static = scaler_static
            self.scaler_temporal = scaler_temporal

        # Group by patient
        self.patient_ids = self.df['SubjectUID'].unique()
        self.patient_data = {
            pid: self.df[self.df['SubjectUID'] == pid].sort_values('assessment_date')
            for pid in self.patient_ids
        }

        # Create index of all valid landmark points
        self.landmarks = []
        for pid in self.patient_ids:
            patient_df = self.patient_data[pid]
            for i in range(len(patient_df)):
                self.landmarks.append((pid, i))

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        pid, visit_idx = self.landmarks[idx]
        patient_df = self.patient_data[pid]

        # Get history up to and including current visit
        history = patient_df.iloc[:visit_idx + 1].tail(self.max_seq_len)
        current = patient_df.iloc[visit_idx]

        # Static features (same for all visits)
        static = history[self.static_features].iloc[0].values.astype(np.float32)
        if self.scaler_static:
            static = self.scaler_static.transform(static.reshape(1, -1)).flatten()

        # Temporal features (sequence)
        temporal = history[self.temporal_features].values.astype(np.float32)
        if self.scaler_temporal:
            temporal = self.scaler_temporal.transform(temporal)

        # Time deltas (for GRU-D style processing)
        if 'days_since_prev_visit' in history.columns:
            time_deltas = history['days_since_prev_visit'].values.astype(np.float32)
            time_deltas = np.clip(time_deltas / 30.0, 0, 12)  # Normalize to months, cap at 12
        else:
            time_deltas = np.zeros(len(history), dtype=np.float32)

        # Mask for missing values
        mask_cols = [f'{f}_missing' for f in self.temporal_features if f'{f}_missing' in history.columns]
        if mask_cols:
            mask = 1 - history[mask_cols].values.astype(np.float32)
        else:
            mask = np.ones((len(history), len(self.temporal_features)), dtype=np.float32)

        # State targets (next visit values if available)
        if visit_idx < len(patient_df) - 1:
            next_visit = patient_df.iloc[visit_idx + 1]
            state_target = np.array([
                next_visit.get(col, np.nan) for col in self.state_targets
            ], dtype=np.float32)
            state_target_mask = (~np.isnan(state_target)).astype(np.float32)
            state_target = np.nan_to_num(state_target, nan=0.0)
        else:
            state_target = np.zeros(len(self.state_targets), dtype=np.float32)
            state_target_mask = np.zeros(len(self.state_targets), dtype=np.float32)

        # Event targets
        event_times = np.array([
            current.get(col, np.nan) for col in self.event_targets
        ], dtype=np.float32)
        event_indicators = np.array([
            current.get(col, 0) for col in self.event_indicators
        ], dtype=np.float32)

        # Cap event times and normalize
        event_times = np.clip(np.nan_to_num(event_times, nan=3650), 0, 3650)  # Cap at 10 years
        event_times = event_times / 365.0  # Convert to years

        return {
            'static': torch.tensor(static, dtype=torch.float32),
            'temporal': torch.tensor(temporal, dtype=torch.float32),
            'time_deltas': torch.tensor(time_deltas, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'seq_len': len(history),
            'state_target': torch.tensor(state_target, dtype=torch.float32),
            'state_target_mask': torch.tensor(state_target_mask, dtype=torch.float32),
            'event_times': torch.tensor(event_times, dtype=torch.float32),
            'event_indicators': torch.tensor(event_indicators, dtype=torch.float32),
        }


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""

    # Get max sequence length in batch
    max_len = max(item['seq_len'] for item in batch)

    # Pad sequences
    static = torch.stack([item['static'] for item in batch])

    temporal_padded = []
    time_deltas_padded = []
    mask_padded = []
    seq_lens = []

    for item in batch:
        seq_len = item['seq_len']
        seq_lens.append(seq_len)

        # Pad temporal
        temporal = item['temporal']
        pad_len = max_len - seq_len
        if pad_len > 0:
            temporal = torch.cat([
                torch.zeros(pad_len, temporal.shape[1]),
                temporal
            ], dim=0)
        temporal_padded.append(temporal)

        # Pad time deltas
        time_deltas = item['time_deltas']
        if pad_len > 0:
            time_deltas = torch.cat([
                torch.zeros(pad_len),
                time_deltas
            ], dim=0)
        time_deltas_padded.append(time_deltas)

        # Pad mask
        mask = item['mask']
        if pad_len > 0:
            mask = torch.cat([
                torch.zeros(pad_len, mask.shape[1]),
                mask
            ], dim=0)
        mask_padded.append(mask)

    return {
        'static': static,
        'temporal': torch.stack(temporal_padded),
        'time_deltas': torch.stack(time_deltas_padded),
        'mask': torch.stack(mask_padded),
        'seq_lens': torch.tensor(seq_lens),
        'state_target': torch.stack([item['state_target'] for item in batch]),
        'state_target_mask': torch.stack([item['state_target_mask'] for item in batch]),
        'event_times': torch.stack([item['event_times'] for item in batch]),
        'event_indicators': torch.stack([item['event_indicators'] for item in batch]),
    }


def create_dataloaders(df_train, df_val, df_test, batch_size=64, num_workers=0):
    """Create train, validation, and test dataloaders."""

    # Create datasets
    train_dataset = ALSDataset(df_train, fit_scalers=True)
    val_dataset = ALSDataset(
        df_val,
        scaler_static=train_dataset.scaler_static,
        scaler_temporal=train_dataset.scaler_temporal
    )
    test_dataset = ALSDataset(
        df_test,
        scaler_static=train_dataset.scaler_static,
        scaler_temporal=train_dataset.scaler_temporal
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_dataset
