"""
Digital Twin Deep Learning Model for ALS.

Architecture:
- Static encoder for baseline features
- GRU-D temporal encoder for longitudinal history
- Shared latent representation
- Multiple output heads:
  - Next-state prediction head
  - Event survival heads (one per event type)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUDCell(nn.Module):
    """
    GRU-D cell that handles missing data and irregular time intervals.
    Based on "Recurrent Neural Networks for Multivariate Time Series with Missing Values"
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # GRU gates
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Decay parameters for hidden state (scalar delta -> hidden_dim decay)
        self.W_gamma_h = nn.Linear(1, hidden_dim)

        # Decay parameters for input (scalar delta -> input_dim decay)
        self.W_gamma_x = nn.Linear(1, input_dim)

        # Learnable mean for imputation
        self.x_mean = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x, h, mask, delta):
        """
        Args:
            x: input at current time [batch, input_dim]
            h: hidden state [batch, hidden_dim]
            mask: observation mask [batch, input_dim]
            delta: time since last observation [batch, 1] (normalized)
        """
        batch_size = x.size(0)

        # Ensure delta has right shape
        if delta.dim() == 1:
            delta = delta.unsqueeze(1)
        if delta.size(1) != 1:
            delta = delta[:, :1]  # Take first column if needed

        # Compute decay rates
        gamma_h = torch.exp(-F.relu(self.W_gamma_h(delta)))  # [batch, hidden_dim]
        gamma_x = torch.exp(-F.relu(self.W_gamma_x(delta)))  # [batch, input_dim]

        # Decay hidden state
        h = gamma_h * h

        # Decay-based imputation for missing values
        x_last = self.x_mean.unsqueeze(0).expand(batch_size, -1)
        x_imputed = mask * x + (1 - mask) * (gamma_x * x + (1 - gamma_x) * x_last)

        # Standard GRU computation
        combined = torch.cat([x_imputed, h], dim=1)

        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))

        combined_r = torch.cat([x_imputed, r * h], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_r))

        h_new = (1 - z) * h + z * h_tilde

        return h_new


class GRUDEncoder(nn.Module):
    """GRU-D encoder for temporal sequences with missing data."""

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU-D cells for each layer
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(GRUDCell(in_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, time_deltas, seq_lens):
        """
        Args:
            x: input sequence [batch, seq_len, input_dim]
            mask: observation mask [batch, seq_len, input_dim]
            time_deltas: time since last observation [batch, seq_len]
            seq_lens: actual sequence lengths [batch]
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        # Initialize hidden states
        h = [torch.zeros(batch_size, self.hidden_dim, device=device)
             for _ in range(self.num_layers)]

        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            mask_t = mask[:, t, :]

            # Time delta as scalar per batch element [batch, 1]
            delta_t = time_deltas[:, t].unsqueeze(1)

            # Process through layers
            layer_input = x_t
            for i, cell in enumerate(self.cells):
                # For first layer, use actual mask. For subsequent layers, assume all observed
                if i == 0:
                    layer_mask = mask_t
                else:
                    layer_mask = torch.ones(batch_size, self.hidden_dim, device=device)
                h[i] = cell(layer_input, h[i], layer_mask, delta_t)
                layer_input = self.dropout(h[i]) if i < self.num_layers - 1 else h[i]

            outputs.append(h[-1])

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, hidden_dim]

        # Get final hidden state (at actual sequence end)
        final_hidden = []
        for i, length in enumerate(seq_lens):
            final_hidden.append(outputs[i, length - 1, :])
        final_hidden = torch.stack(final_hidden)

        return outputs, final_hidden


class StaticEncoder(nn.Module):
    """Encoder for static baseline features."""

    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.encoder(x)


class StateHead(nn.Module):
    """Head for next-state prediction (ALSFRS-R scores)."""

    def __init__(self, input_dim, num_targets, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_targets)
        )

    def forward(self, x):
        return self.net(x)


class SurvivalHead(nn.Module):
    """
    Discrete-time survival head for event prediction.
    Outputs hazard probabilities for discrete time intervals.
    """

    def __init__(self, input_dim, num_intervals=20, hidden_dim=64):
        super().__init__()
        self.num_intervals = num_intervals

        # Time intervals in years: 0-0.5, 0.5-1, 1-1.5, ..., 9.5-10
        self.interval_bounds = torch.linspace(0, 10, num_intervals + 1)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_intervals)
        )

    def forward(self, x):
        # Output hazard logits for each interval
        hazard_logits = self.net(x)
        return hazard_logits

    def get_survival_probs(self, hazard_logits):
        """Convert hazard logits to survival probabilities."""
        # Hazard probability for each interval
        hazard_probs = torch.sigmoid(hazard_logits)

        # Survival probability = cumulative product of (1 - hazard)
        survival_probs = torch.cumprod(1 - hazard_probs, dim=1)

        # Prepend 1 (survival at time 0 is 1)
        ones = torch.ones(survival_probs.size(0), 1, device=survival_probs.device)
        survival_probs = torch.cat([ones, survival_probs], dim=1)

        return survival_probs

    def get_median_time(self, hazard_logits):
        """Get median survival time prediction."""
        survival_probs = self.get_survival_probs(hazard_logits)

        # Find first interval where survival drops below 0.5
        below_50 = survival_probs < 0.5
        median_idx = below_50.float().argmax(dim=1)

        # Handle cases where survival never drops below 0.5
        never_below = ~below_50.any(dim=1)
        median_idx[never_below] = self.num_intervals

        # Convert to time
        interval_width = 10.0 / self.num_intervals
        median_time = median_idx.float() * interval_width

        return median_time


class ALSDigitalTwin(nn.Module):
    """
    ALS Digital Twin Model.

    Multi-task model that predicts:
    1. Next clinical state (ALSFRS-R scores)
    2. Time-to-event for multiple events (survival analysis)
    """

    def __init__(
        self,
        static_dim,
        temporal_dim,
        hidden_dim=128,
        num_layers=2,
        num_state_targets=6,
        num_events=5,
        num_survival_intervals=20,
        dropout=0.2
    ):
        super().__init__()

        self.static_dim = static_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim

        # Encoders
        self.static_encoder = StaticEncoder(static_dim, hidden_dim, dropout)
        self.temporal_encoder = GRUDEncoder(temporal_dim, hidden_dim, num_layers, dropout)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.state_head = StateHead(hidden_dim, num_state_targets)

        self.survival_heads = nn.ModuleList([
            SurvivalHead(hidden_dim, num_survival_intervals)
            for _ in range(num_events)
        ])

    def forward(self, static, temporal, mask, time_deltas, seq_lens):
        """
        Forward pass.

        Args:
            static: static features [batch, static_dim]
            temporal: temporal features [batch, seq_len, temporal_dim]
            mask: observation mask [batch, seq_len, temporal_dim]
            time_deltas: time gaps [batch, seq_len]
            seq_lens: sequence lengths [batch]

        Returns:
            state_pred: predicted next state [batch, num_state_targets]
            hazard_logits: list of hazard logits [batch, num_intervals] for each event
        """
        # Encode static features
        static_encoded = self.static_encoder(static)

        # Encode temporal features
        _, temporal_encoded = self.temporal_encoder(temporal, mask, time_deltas, seq_lens)

        # Fuse representations
        combined = torch.cat([static_encoded, temporal_encoded], dim=1)
        latent = self.fusion(combined)

        # Predictions
        state_pred = self.state_head(latent)

        hazard_logits = [head(latent) for head in self.survival_heads]

        return state_pred, hazard_logits

    def get_digital_twin_state(self, static, temporal, mask, time_deltas, seq_lens):
        """
        Get complete digital twin state including predictions and survival curves.
        """
        state_pred, hazard_logits = self.forward(
            static, temporal, mask, time_deltas, seq_lens
        )

        # Get survival probabilities and median times
        survival_probs = []
        median_times = []
        for i, head in enumerate(self.survival_heads):
            surv = head.get_survival_probs(hazard_logits[i])
            med = head.get_median_time(hazard_logits[i])
            survival_probs.append(surv)
            median_times.append(med)

        return {
            'state_prediction': state_pred,
            'hazard_logits': hazard_logits,
            'survival_probs': survival_probs,
            'median_event_times': median_times
        }


class DigitalTwinLoss(nn.Module):
    """
    Combined loss for Digital Twin training.

    Combines:
    1. MSE loss for state prediction
    2. Discrete-time survival loss for events
    """

    def __init__(
        self,
        state_weight=1.0,
        survival_weight=1.0,
        num_events=5,
        num_intervals=20
    ):
        super().__init__()
        self.state_weight = state_weight
        self.survival_weight = survival_weight
        self.num_events = num_events
        self.num_intervals = num_intervals

        # Time interval bounds (in years)
        self.register_buffer(
            'interval_bounds',
            torch.linspace(0, 10, num_intervals + 1)
        )

    def forward(self, state_pred, hazard_logits, state_target, state_mask,
                event_times, event_indicators):
        """
        Compute combined loss.

        Args:
            state_pred: [batch, num_targets]
            hazard_logits: list of [batch, num_intervals]
            state_target: [batch, num_targets]
            state_mask: [batch, num_targets] - which targets are valid
            event_times: [batch, num_events] - time to event in years
            event_indicators: [batch, num_events] - whether event occurred
        """
        losses = {}

        # State prediction loss (masked MSE)
        state_diff = (state_pred - state_target) ** 2
        state_loss = (state_diff * state_mask).sum() / (state_mask.sum() + 1e-8)
        losses['state'] = state_loss

        # Survival losses
        survival_losses = []
        for i in range(self.num_events):
            hazard = hazard_logits[i]
            times = event_times[:, i]
            indicators = event_indicators[:, i]

            loss = self._discrete_survival_loss(hazard, times, indicators)
            survival_losses.append(loss)
            losses[f'survival_{i}'] = loss

        survival_loss = sum(survival_losses) / self.num_events
        losses['survival'] = survival_loss

        # Total loss
        total = self.state_weight * state_loss + self.survival_weight * survival_loss
        losses['total'] = total

        return total, losses

    def _discrete_survival_loss(self, hazard_logits, times, indicators):
        """
        Discrete-time survival loss (negative log-likelihood).

        For observed events: log P(T in interval k)
        For censored: log P(T > observed time)
        """
        device = hazard_logits.device
        batch_size = hazard_logits.size(0)

        # Find which interval each event/censoring time falls into
        interval_idx = torch.bucketize(times, self.interval_bounds[1:].to(device))
        interval_idx = torch.clamp(interval_idx, 0, self.num_intervals - 1)

        # Hazard probabilities
        hazard_probs = torch.sigmoid(hazard_logits)

        # Survival to each interval
        log_survival = torch.zeros(batch_size, self.num_intervals + 1, device=device)
        for k in range(self.num_intervals):
            log_survival[:, k + 1] = log_survival[:, k] + torch.log(1 - hazard_probs[:, k] + 1e-8)

        # For events: log P(T in interval k) = log S(k-1) + log h(k)
        # For censored: log P(T > t) = log S(interval containing t)
        log_likelihood = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            k = interval_idx[i]
            if indicators[i] > 0.5:  # Event observed
                # Probability of surviving to interval k and then having event
                log_likelihood[i] = log_survival[i, k] + torch.log(hazard_probs[i, k] + 1e-8)
            else:  # Censored
                # Probability of surviving past observed time
                log_likelihood[i] = log_survival[i, k]

        # Negative log-likelihood
        loss = -log_likelihood.mean()

        return loss


def create_model(static_dim, temporal_dim, config=None):
    """Create model with default or custom configuration."""

    if config is None:
        config = {
            'hidden_dim': 128,
            'num_layers': 2,
            'num_state_targets': 6,
            'num_events': 5,
            'num_survival_intervals': 20,
            'dropout': 0.2
        }

    model = ALSDigitalTwin(
        static_dim=static_dim,
        temporal_dim=temporal_dim,
        **config
    )

    return model
