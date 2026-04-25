"""
Transformer-based Autoencoder Digital Twin for ALS.

Architecture:
- Transformer encoder for longitudinal patient history
- Latent bottleneck (digital twin state)
- Multiple decoder heads:
  - Reconstruction head (autoencoder)
  - State prediction head (next visit)
  - Survival prediction heads (time-to-event)

The autoencoder structure encourages learning a compact, meaningful
representation of the patient's disease trajectory.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with time-aware modifications."""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x, time_deltas=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            time_deltas: [batch, seq_len] - optional time gaps
        """
        seq_len = x.size(1)

        if time_deltas is not None:
            # Time-aware positional encoding
            # Scale positions by cumulative time
            cum_time = time_deltas.cumsum(dim=1)  # [batch, seq_len]
            # Normalize to reasonable range
            cum_time = cum_time / (cum_time.max(dim=1, keepdim=True)[0] + 1e-8) * seq_len
            cum_time = cum_time.long().clamp(0, self.pe.size(1) - 1)

            # Gather time-aware positions
            pe = self.pe.expand(x.size(0), -1, -1)  # [batch, max_len, d_model]
            pe = torch.gather(pe, 1, cum_time.unsqueeze(-1).expand(-1, -1, pe.size(-1)))
        else:
            pe = self.pe[:, :seq_len, :]

        x = x + pe
        return self.dropout(x)


class TemporalAttention(nn.Module):
    """
    Multi-head attention with causal masking and time-aware attention bias.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Time gap bias projection
        self.time_bias = nn.Linear(1, n_heads)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None, time_deltas=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len] - padding mask (1 = valid, 0 = pad)
            time_deltas: [batch, seq_len] - time gaps
        """
        batch_size, seq_len, _ = x.size()

        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, heads, seq, seq]

        # Add time-aware bias if available
        if time_deltas is not None:
            # Compute pairwise time differences
            cum_time = time_deltas.cumsum(dim=1)  # [batch, seq_len]
            time_diff = cum_time.unsqueeze(2) - cum_time.unsqueeze(1)  # [batch, seq, seq]
            time_diff = time_diff.abs().unsqueeze(-1)  # [batch, seq, seq, 1]

            # Project to per-head bias
            time_bias = self.time_bias(time_diff)  # [batch, seq, seq, n_heads]
            time_bias = time_bias.permute(0, 3, 1, 2)  # [batch, n_heads, seq, seq]

            # Apply decay based on time difference
            time_bias = -F.softplus(time_bias)  # Negative bias for larger gaps
            scores = scores + time_bias

        # Causal mask (only attend to past)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Padding mask
        if mask is not None:
            # Convert to attention mask [batch, 1, 1, seq_len]
            attn_mask = (1 - mask.unsqueeze(1).unsqueeze(2)).bool()
            scores = scores.masked_fill(attn_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, V)  # [batch, heads, seq, d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.W_o(context)
        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with temporal attention."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = TemporalAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, time_deltas=None):
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, mask, time_deltas)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None, time_deltas=None):
        all_attn_weights = []

        for layer in self.layers:
            x, attn_weights = layer(x, mask, time_deltas)
            all_attn_weights.append(attn_weights)

        return x, all_attn_weights


class LatentBottleneck(nn.Module):
    """
    Compress sequence into fixed-size latent representation.
    This is the "digital twin state" - a compact representation of the patient.
    """

    def __init__(self, d_model, latent_dim, n_heads=4):
        super().__init__()

        # Learnable query for extracting patient state
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

        # Cross-attention to summarize sequence
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Project to latent space
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.Tanh()  # Bounded latent space
        )

        # For VAE-style training (optional)
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, encoder_output, mask=None, return_variational=False):
        """
        Args:
            encoder_output: [batch, seq_len, d_model]
            mask: [batch, seq_len]
            return_variational: if True, return mu and logvar for VAE training
        """
        batch_size = encoder_output.size(0)

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)

        # Create key padding mask for cross-attention
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (1 - mask).bool()

        # Cross-attention: query attends to encoder outputs
        summary, _ = self.cross_attention(
            query, encoder_output, encoder_output,
            key_padding_mask=key_padding_mask
        )
        summary = summary.squeeze(1)  # [batch, d_model]

        if return_variational:
            mu = self.to_mu(summary)
            logvar = self.to_logvar(summary)
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else:
            z = self.to_latent(summary)
            return z


class ReconstructionDecoder(nn.Module):
    """Decoder for reconstructing input sequence from latent state."""

    def __init__(self, latent_dim, d_model, output_dim, max_len=50):
        super().__init__()

        # Project latent to initial hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, d_model)

        # Positional embeddings for decoding
        self.pos_embed = nn.Embedding(max_len, d_model)

        # Transformer decoder layers
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, z, seq_len, target=None):
        """
        Args:
            z: latent state [batch, latent_dim]
            seq_len: length of sequence to generate
            target: optional target sequence for teacher forcing [batch, seq_len, output_dim]
        """
        batch_size = z.size(0)
        device = z.device

        # Create memory from latent
        memory = self.latent_to_hidden(z).unsqueeze(1)  # [batch, 1, d_model]

        # Position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)  # [batch, seq_len, d_model]

        # Decode
        if target is not None:
            # Use target for teacher forcing (during training)
            tgt = pos_emb
        else:
            tgt = pos_emb

        # Causal mask
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        decoded = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.output_proj(decoded)

        return output


class StatePredictionHead(nn.Module):
    """Predicts next clinical state from latent representation."""

    def __init__(self, latent_dim, hidden_dim, num_targets):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_targets)
        )

    def forward(self, z):
        return self.net(z)


class SurvivalPredictionHead(nn.Module):
    """
    Predicts survival distribution using discrete-time hazards.
    """

    def __init__(self, latent_dim, hidden_dim, num_intervals=20):
        super().__init__()
        self.num_intervals = num_intervals

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_intervals)
        )

    def forward(self, z):
        return self.net(z)

    def get_survival_probs(self, hazard_logits):
        """Convert hazard logits to survival curve."""
        hazard_probs = torch.sigmoid(hazard_logits)
        survival = torch.cumprod(1 - hazard_probs, dim=1)
        ones = torch.ones(survival.size(0), 1, device=survival.device)
        return torch.cat([ones, survival], dim=1)

    def get_median_time(self, hazard_logits, max_time=10.0):
        """Get median survival time."""
        survival = self.get_survival_probs(hazard_logits)
        below_50 = survival < 0.5
        median_idx = below_50.float().argmax(dim=1)
        never_below = ~below_50.any(dim=1)
        median_idx[never_below] = self.num_intervals
        interval_width = max_time / self.num_intervals
        return median_idx.float() * interval_width


class TransformerAutoencoderTwin(nn.Module):
    """
    Transformer-based Autoencoder Digital Twin.

    The model learns a compressed latent representation of the patient's
    disease trajectory through reconstruction, then uses this representation
    for prediction tasks.
    """

    def __init__(
        self,
        static_dim,
        temporal_dim,
        d_model=128,
        n_heads=4,
        n_layers=3,
        latent_dim=64,
        num_state_targets=6,
        num_events=5,
        num_survival_intervals=20,
        dropout=0.1,
        use_variational=False
    ):
        super().__init__()

        self.d_model = d_model
        self.latent_dim = latent_dim
        self.use_variational = use_variational

        # Input projections
        self.static_embed = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.temporal_embed = nn.Sequential(
            nn.Linear(temporal_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Combine static and temporal
        self.combine = nn.Linear(d_model * 2, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * 4,
            n_layers=n_layers,
            dropout=dropout
        )

        # Latent bottleneck
        self.bottleneck = LatentBottleneck(d_model, latent_dim, n_heads)

        # Reconstruction decoder
        self.reconstruction_decoder = ReconstructionDecoder(
            latent_dim, d_model, temporal_dim
        )

        # Prediction heads
        self.state_head = StatePredictionHead(latent_dim, d_model, num_state_targets)

        self.survival_heads = nn.ModuleList([
            SurvivalPredictionHead(latent_dim, d_model // 2, num_survival_intervals)
            for _ in range(num_events)
        ])

    def encode(self, static, temporal, mask=None, time_deltas=None, seq_lens=None):
        """
        Encode patient history into latent representation.

        Args:
            static: [batch, static_dim]
            temporal: [batch, seq_len, temporal_dim]
            mask: [batch, seq_len, temporal_dim] - observation mask
            time_deltas: [batch, seq_len]
            seq_lens: [batch]

        Returns:
            z: latent representation [batch, latent_dim]
            encoder_output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = temporal.size()

        # Embed inputs
        static_emb = self.static_embed(static)  # [batch, d_model]
        temporal_emb = self.temporal_embed(temporal)  # [batch, seq_len, d_model]

        # Broadcast static to all timesteps and combine
        static_expanded = static_emb.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([static_expanded, temporal_emb], dim=-1)
        x = self.combine(combined)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoding(x, time_deltas)

        # Create attention mask from sequence lengths
        attn_mask = None
        if seq_lens is not None:
            attn_mask = torch.zeros(batch_size, seq_len, device=temporal.device)
            for i, length in enumerate(seq_lens):
                attn_mask[i, :length] = 1

        # Encode
        encoder_output, attn_weights = self.encoder(x, attn_mask, time_deltas)

        # Compress to latent
        if self.use_variational:
            z, mu, logvar = self.bottleneck(encoder_output, attn_mask, return_variational=True)
            return z, encoder_output, mu, logvar
        else:
            z = self.bottleneck(encoder_output, attn_mask)
            return z, encoder_output

    def decode_reconstruction(self, z, seq_len):
        """Reconstruct input sequence from latent."""
        return self.reconstruction_decoder(z, seq_len)

    def predict_state(self, z):
        """Predict next clinical state."""
        return self.state_head(z)

    def predict_survival(self, z):
        """Predict survival distributions for all events."""
        return [head(z) for head in self.survival_heads]

    def forward(self, static, temporal, mask, time_deltas, seq_lens):
        """
        Full forward pass.

        Returns:
            reconstruction: [batch, seq_len, temporal_dim]
            state_pred: [batch, num_state_targets]
            hazard_logits: list of [batch, num_intervals]
            z: latent representation [batch, latent_dim]
        """
        # Encode
        if self.use_variational:
            z, encoder_output, mu, logvar = self.encode(
                static, temporal, mask, time_deltas, seq_lens
            )
        else:
            z, encoder_output = self.encode(
                static, temporal, mask, time_deltas, seq_lens
            )
            mu, logvar = None, None

        # Get sequence length from input
        seq_len = temporal.size(1)

        # Reconstruct
        reconstruction = self.decode_reconstruction(z, seq_len)

        # Predict
        state_pred = self.predict_state(z)
        hazard_logits = self.predict_survival(z)

        outputs = {
            'reconstruction': reconstruction,
            'state_pred': state_pred,
            'hazard_logits': hazard_logits,
            'latent': z,
            'encoder_output': encoder_output
        }

        if self.use_variational:
            outputs['mu'] = mu
            outputs['logvar'] = logvar

        return outputs

    def get_digital_twin_state(self, static, temporal, mask, time_deltas, seq_lens):
        """
        Get complete digital twin state for a patient.
        """
        outputs = self.forward(static, temporal, mask, time_deltas, seq_lens)

        # Get survival curves and median times
        survival_probs = []
        median_times = []
        for i, head in enumerate(self.survival_heads):
            hazard = outputs['hazard_logits'][i]
            survival_probs.append(head.get_survival_probs(hazard))
            median_times.append(head.get_median_time(hazard))

        return {
            'latent_state': outputs['latent'],
            'state_prediction': outputs['state_pred'],
            'survival_probs': survival_probs,
            'median_event_times': median_times,
            'reconstruction': outputs['reconstruction']
        }


class TransformerTwinLoss(nn.Module):
    """
    Combined loss for Transformer Autoencoder Twin.

    Components:
    1. Reconstruction loss (MSE)
    2. State prediction loss (MSE)
    3. Survival prediction loss (negative log-likelihood)
    4. KL divergence (if variational)
    """

    def __init__(
        self,
        recon_weight=1.0,
        state_weight=1.0,
        survival_weight=1.0,
        kl_weight=0.1,
        num_events=5,
        num_intervals=20
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.state_weight = state_weight
        self.survival_weight = survival_weight
        self.kl_weight = kl_weight
        self.num_events = num_events
        self.num_intervals = num_intervals

        self.register_buffer(
            'interval_bounds',
            torch.linspace(0, 10, num_intervals + 1)
        )

    def forward(
        self,
        outputs,
        temporal_target,
        temporal_mask,
        state_target,
        state_mask,
        event_times,
        event_indicators
    ):
        """
        Compute combined loss.
        """
        losses = {}

        # Reconstruction loss
        recon = outputs['reconstruction']
        recon_diff = (recon - temporal_target) ** 2

        # Average over valid (observed) values
        if temporal_mask is not None:
            # Use first column of mask as sequence mask
            seq_mask = temporal_mask.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]
            recon_loss = (recon_diff * seq_mask).sum() / (seq_mask.sum() * recon.size(-1) + 1e-8)
        else:
            recon_loss = recon_diff.mean()
        losses['reconstruction'] = recon_loss

        # State prediction loss
        state_pred = outputs['state_pred']
        state_diff = (state_pred - state_target) ** 2
        state_loss = (state_diff * state_mask).sum() / (state_mask.sum() + 1e-8)
        losses['state'] = state_loss

        # Survival losses
        survival_losses = []
        for i in range(self.num_events):
            hazard = outputs['hazard_logits'][i]
            times = event_times[:, i]
            indicators = event_indicators[:, i]
            loss = self._discrete_survival_loss(hazard, times, indicators)
            survival_losses.append(loss)
            losses[f'survival_{i}'] = loss

        survival_loss = sum(survival_losses) / self.num_events
        losses['survival'] = survival_loss

        # KL divergence (if variational)
        if 'mu' in outputs and outputs['mu'] is not None:
            mu = outputs['mu']
            logvar = outputs['logvar']
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            losses['kl'] = kl_loss
        else:
            kl_loss = 0
            losses['kl'] = torch.tensor(0.0)

        # Total loss
        total = (
            self.recon_weight * recon_loss +
            self.state_weight * state_loss +
            self.survival_weight * survival_loss +
            self.kl_weight * kl_loss
        )
        losses['total'] = total

        return total, losses

    def _discrete_survival_loss(self, hazard_logits, times, indicators):
        """Discrete-time survival negative log-likelihood."""
        device = hazard_logits.device
        batch_size = hazard_logits.size(0)

        interval_idx = torch.bucketize(times, self.interval_bounds[1:].to(device))
        interval_idx = torch.clamp(interval_idx, 0, self.num_intervals - 1)

        hazard_probs = torch.sigmoid(hazard_logits)

        log_survival = torch.zeros(batch_size, self.num_intervals + 1, device=device)
        for k in range(self.num_intervals):
            log_survival[:, k + 1] = log_survival[:, k] + torch.log(1 - hazard_probs[:, k] + 1e-8)

        log_likelihood = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            k = interval_idx[i]
            if indicators[i] > 0.5:
                log_likelihood[i] = log_survival[i, k] + torch.log(hazard_probs[i, k] + 1e-8)
            else:
                log_likelihood[i] = log_survival[i, k]

        return -log_likelihood.mean()


def create_transformer_twin(static_dim, temporal_dim, config=None):
    """Create Transformer Autoencoder Twin with default or custom config."""

    if config is None:
        config = {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'latent_dim': 64,
            'num_state_targets': 6,
            'num_events': 5,
            'num_survival_intervals': 20,
            'dropout': 0.1,
            'use_variational': False
        }

    model = TransformerAutoencoderTwin(
        static_dim=static_dim,
        temporal_dim=temporal_dim,
        **config
    )

    return model
