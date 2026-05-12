"""
Domain Stage Transition Predictor for ALS.

Given a patient's current clinical observation (history), predicts how long
until their score drops to the next stage for each of 5 domains:
  Bulbar, Upper (fine motor), Axial (gross motor), Lower (walking), Resp

Uses a shared Transformer encoder + 5 domain-specific discrete-time hazard heads.
"Next stage" = first future visit where domain score drops by ≥1 point.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DOMAINS = ['bulbar', 'fine_motor', 'gross_motor', 'walking', 'respiratory']
DOMAIN_COLS = [f'domain_{d}' for d in DOMAINS]
DOMAIN_MAX = {'bulbar': 12, 'fine_motor': 8, 'gross_motor': 8, 'walking': 8, 'respiratory': 12}
DOMAIN_LABELS = ['Bulbar', 'Upper (Fine Motor)', 'Axial (Gross Motor)', 'Lower (Walking)', 'Respiratory']


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DomainStagePredictor(nn.Module):
    """
    Predicts time-to-next-stage for each ALS domain.

    Architecture:
      1. Static embedding
      2. Temporal projection → Transformer encoder
      3. Cross-attention pooling → patient state vector
      4. 5 domain-specific hazard heads (each conditioned on current domain score)
    """

    def __init__(self, static_dim, temporal_dim, config):
        super().__init__()

        d_model = config['d_model']
        n_heads = config['n_heads']
        n_layers = config['n_layers']
        dropout = config.get('dropout', 0.1)
        self.num_intervals = config['num_intervals']

        # Static embedding
        self.static_embed = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Temporal projection + norm
        self.temporal_proj = nn.Linear(temporal_dim, d_model)
        self.temporal_norm = nn.LayerNorm(d_model)

        # Shared Transformer encoder (no causal mask — encoder-only style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Learnable query for pooling the sequence into a single patient-state vector
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Fusion: static + temporal context → shared representation
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Domain-specific survival heads
        # Input: shared rep (d_model) + current normalised domain score (1)
        head_in = d_model + 1
        head_hidden = d_model // 2
        self.domain_heads = nn.ModuleDict({
            d: nn.Sequential(
                nn.Linear(head_in, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, self.num_intervals),
            )
            for d in DOMAINS
        })

    def _encode(self, static, temporal, mask, time_deltas, seq_lens):
        B, T, _ = temporal.shape

        static_emb = self.static_embed(static)                          # [B, d]

        temporal_emb = self.temporal_norm(self.temporal_proj(temporal)) # [B, T, d]

        # Padding mask: True = position to IGNORE (PyTorch convention)
        # Sequences are left-padded by collate_fn, so padding is at the front.
        pad_mask = torch.ones(B, T, dtype=torch.bool, device=temporal.device)  # True = ignore
        for i, l in enumerate(seq_lens):
            pad_mask[i, T - l.item():] = False  # valid positions → False (don't ignore)

        encoded = self.encoder(temporal_emb, src_key_padding_mask=pad_mask)
        # encoded: [B, T, d]

        # Pool via cross-attention
        query = self.pool_query.expand(B, -1, -1)                       # [B, 1, d]
        key_padding_mask = pad_mask                                      # True = ignore
        context, _ = self.pool_attn(
            query, encoded, encoded, key_padding_mask=key_padding_mask
        )
        context = context.squeeze(1)                                    # [B, d]

        fused = self.fusion(torch.cat([static_emb, context], dim=-1))  # [B, d]
        return fused

    def forward(self, static, temporal, mask, time_deltas, seq_lens,
                current_domain_scores=None):
        """
        Returns:
            hazards: dict {domain_name: Tensor[B, num_intervals]}
                     sigmoid hazard probabilities per interval
        """
        fused = self._encode(static, temporal, mask, time_deltas, seq_lens)

        B = fused.shape[0]
        hazards = {}
        for i, domain in enumerate(DOMAINS):
            if current_domain_scores is not None:
                score = current_domain_scores[:, i:i+1]                # [B, 1]
            else:
                score = torch.zeros(B, 1, device=fused.device)

            logits = self.domain_heads[domain](torch.cat([fused, score], dim=-1))
            hazards[domain] = torch.sigmoid(logits)                    # [B, K]

        return hazards

    def predict_median_times(self, hazards, interval_width_days=30):
        """Convert hazard curves to predicted median time (days) per domain."""
        medians = {}
        for domain, h in hazards.items():
            surv = torch.cumprod(1 - h + 1e-7, dim=1)                  # [B, K]
            below = (surv < 0.5).float()
            idx = below.argmax(dim=1)                                   # [B]
            # If survival never drops below 0.5, predict beyond last interval
            never_cross = (below.sum(dim=1) == 0)
            idx[never_cross] = self.num_intervals - 1
            medians[domain] = (idx.float() + 0.5) * interval_width_days
        return medians


def create_stage_predictor(static_dim, temporal_dim, config=None):
    if config is None:
        config = {
            'd_model': 128, 'n_heads': 4, 'n_layers': 3,
            'num_intervals': 24, 'dropout': 0.1,
        }
    return DomainStagePredictor(static_dim, temporal_dim, config)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class StagePredictorLoss(nn.Module):
    """
    Discrete-time survival NLL averaged over the 5 domains.

    For a sample with event in bin t (delta=1):
        log P(T=t) = sum_{k<t} log(1-h_k) + log(h_t)
    For a censored sample at bin c (delta=0):
        log P(T>c) = sum_{k<=c} log(1-h_k)
    """

    def forward(self, hazards, event_times_bin, event_indicators):
        """
        Args:
            hazards:          dict {domain: [B, K]}
            event_times_bin:  [B, 5]  bin index (int)
            event_indicators: [B, 5]  1=event, 0=censored
        Returns:
            total_loss, dict of per-domain losses
        """
        total = 0.0
        domain_losses = {}

        for i, domain in enumerate(DOMAINS):
            h = hazards[domain]                         # [B, K]
            B, K = h.shape

            t = event_times_bin[:, i].long().clamp(0, K - 1)  # [B]
            delta = event_indicators[:, i]                     # [B]

            log_1mh = torch.log(1 - h + 1e-7)           # [B, K]
            cum_log_1mh = torch.cumsum(log_1mh, dim=1)  # [B, K]  cumulative log-survival

            # log S(t-1): sum of log(1-h) for intervals strictly before t
            prev_t = (t - 1).clamp(0, K - 1)
            log_surv_before = cum_log_1mh[torch.arange(B, device=h.device), prev_t]
            log_surv_before = torch.where(t > 0, log_surv_before,
                                          torch.zeros_like(log_surv_before))

            log_h_at_t = torch.log(h[torch.arange(B, device=h.device), t] + 1e-7)

            # log S(t): includes interval t (for censored)
            log_surv_through = cum_log_1mh[torch.arange(B, device=h.device), t]

            nll = -(delta * (log_surv_before + log_h_at_t)
                    + (1 - delta) * log_surv_through)

            # Exclude samples that are immediately censored at bin 0 with no event
            # (no future visits and score not at floor — uninformative)
            valid = (t > 0) | (delta > 0)
            loss = nll[valid].mean() if valid.sum() > 0 else h.sum() * 0.0

            total = total + loss
            domain_losses[domain] = loss

        return total / len(DOMAINS), domain_losses
