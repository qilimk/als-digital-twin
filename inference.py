"""
Digital Twin Inference Script

Use the trained digital twin model to make predictions for a patient.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json

from models.data_loader import (
    load_data, ALSDataset, collate_fn,
    STATIC_FEATURES, TEMPORAL_FEATURES, ALSFRS_ITEMS
)
from models.digital_twin import ALSDigitalTwin


class DigitalTwinPredictor:
    """
    Wrapper for making predictions with the trained digital twin.
    """

    def __init__(self, model_dir='./trained_models'):
        self.model_dir = Path(model_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        # Load deep model
        self._load_deep_model()

        # Load classical model
        self._load_classical_model()

        # Event names
        self.event_names = ['Death', 'Gastrostomy', 'NIV', 'Wheelchair', 'Speech Loss']
        self.state_names = ['ALSFRS Total', 'Bulbar', 'Fine Motor', 'Gross Motor', 'Walking', 'Respiratory']

    def _load_deep_model(self):
        """Load the trained deep learning model."""
        checkpoint_path = self.model_dir / 'digital_twin_best.pt'
        if not checkpoint_path.exists():
            print("Warning: Deep model checkpoint not found")
            self.deep_model = None
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Recreate model
        self.deep_model = ALSDigitalTwin(
            static_dim=checkpoint['static_dim'],
            temporal_dim=checkpoint['temporal_dim'],
            **checkpoint['config']
        )
        self.deep_model.load_state_dict(checkpoint['model_state_dict'])
        self.deep_model = self.deep_model.to(self.device)
        self.deep_model.eval()

        print(f"Loaded deep model from epoch {checkpoint['epoch']}")

    def _load_classical_model(self):
        """Load the trained classical model."""
        classical_path = self.model_dir / 'classical_twin.pkl'
        if not classical_path.exists():
            print("Warning: Classical model not found")
            self.classical_model = None
            return

        with open(classical_path, 'rb') as f:
            self.classical_model = pickle.load(f)
        print("Loaded classical model")

    def predict_from_dataframe(self, patient_df):
        """
        Make predictions for a patient given their visit history.

        Args:
            patient_df: DataFrame with patient's visit history

        Returns:
            dict with predictions from both models
        """
        results = {}

        # Deep model predictions
        if self.deep_model is not None:
            results['deep'] = self._predict_deep(patient_df)

        # Classical model predictions
        if self.classical_model is not None:
            results['classical'] = self._predict_classical(patient_df)

        return results

    def _predict_deep(self, patient_df):
        """Get predictions from deep model."""
        # Create dataset for single patient
        dataset = ALSDataset(patient_df, fit_scalers=True)

        if len(dataset) == 0:
            return None

        # Get last landmark (most recent visit)
        sample = dataset[-1]

        # Add batch dimension
        batch = {
            'static': sample['static'].unsqueeze(0).to(self.device),
            'temporal': sample['temporal'].unsqueeze(0).to(self.device),
            'mask': sample['mask'].unsqueeze(0).to(self.device),
            'time_deltas': sample['time_deltas'].unsqueeze(0).to(self.device),
            'seq_lens': torch.tensor([sample['seq_len']]).to(self.device),
        }

        with torch.no_grad():
            twin_state = self.deep_model.get_digital_twin_state(
                batch['static'], batch['temporal'], batch['mask'],
                batch['time_deltas'], batch['seq_lens']
            )

        # Extract predictions
        state_pred = twin_state['state_prediction'][0].cpu().numpy()
        median_times = [m[0].cpu().item() * 365 for m in twin_state['median_event_times']]
        survival_probs = [s[0].cpu().numpy() for s in twin_state['survival_probs']]

        return {
            'state_prediction': dict(zip(self.state_names, state_pred)),
            'median_event_times_days': dict(zip(self.event_names, median_times)),
            'survival_curves': dict(zip(self.event_names, survival_probs)),
        }

    def _predict_classical(self, patient_df):
        """Get predictions from classical model."""
        predictions = self.classical_model.predict(patient_df.tail(1))

        # Get last row predictions
        state_preds = {k: v[-1] for k, v in predictions['state'].items()}
        time_preds = {k: v[-1] for k, v in predictions['median_times'].items() if len(v) > 0}
        risk_preds = {k: v[-1] for k, v in predictions['risk'].items() if len(v) > 0}

        return {
            'state_prediction': state_preds,
            'median_event_times_days': time_preds,
            'risk_1year': risk_preds,
        }

    def format_report(self, predictions, patient_id=None):
        """Format predictions as a readable report."""
        lines = []
        lines.append("=" * 60)
        lines.append("DIGITAL TWIN PREDICTION REPORT")
        if patient_id:
            lines.append(f"Patient: {patient_id}")
        lines.append("=" * 60)

        if 'deep' in predictions and predictions['deep'] is not None:
            lines.append("\n--- Deep Learning Model Predictions ---\n")

            lines.append("Next Visit ALSFRS-R Predictions:")
            for name, val in predictions['deep']['state_prediction'].items():
                lines.append(f"  {name}: {val:.1f}")

            lines.append("\nMedian Time to Events (days):")
            for name, val in predictions['deep']['median_event_times_days'].items():
                lines.append(f"  {name}: {val:.0f}")

        if 'classical' in predictions and predictions['classical'] is not None:
            lines.append("\n--- Classical Model Predictions ---\n")

            lines.append("Next Visit ALSFRS-R Predictions:")
            for name, val in predictions['classical']['state_prediction'].items():
                lines.append(f"  {name}: {val:.1f}")

            if predictions['classical'].get('median_event_times_days'):
                lines.append("\nMedian Time to Events (days):")
                for name, val in predictions['classical']['median_event_times_days'].items():
                    lines.append(f"  {name}: {val:.0f}")

            if predictions['classical'].get('risk_1year'):
                lines.append("\n1-Year Event Risk:")
                for name, val in predictions['classical']['risk_1year'].items():
                    lines.append(f"  {name}: {val*100:.1f}%")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


def demo_inference():
    """Demo: Run inference on a sample patient."""
    print("Digital Twin Inference Demo")
    print("=" * 60)

    # Load data
    df = load_data('./cleaned_data')

    # Get a sample patient with multiple visits
    patient_visits = df.groupby('SubjectUID').size()
    sample_patient = patient_visits[patient_visits >= 5].index[0]
    patient_df = df[df['SubjectUID'] == sample_patient].copy()

    print(f"\nSample patient: {sample_patient}")
    print(f"Number of visits: {len(patient_df)}")

    # Create predictor
    predictor = DigitalTwinPredictor()

    # Make predictions
    predictions = predictor.predict_from_dataframe(patient_df)

    # Print report
    report = predictor.format_report(predictions, sample_patient)
    print(report)

    return predictions


if __name__ == "__main__":
    demo_inference()
