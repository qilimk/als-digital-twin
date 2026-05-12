"""
ALS Digital Twin — Interactive Web App

Two prediction modes in one tool:
  1. Domain Stage Predictor — Transformer model that predicts when each of the
     5 ALSFRS-R domains will drop by ≥1 point.
  2. Wheelchair-Free Survival — Cox proportional-hazards model that predicts
     time-to-wheelchair-access from a single visit.

Run: streamlit run app.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from models.stage_predictor import create_stage_predictor, DOMAINS, DOMAIN_MAX
from models.data_loader import ALSDataset, prepare_features


# ============================================================================
# Constants
# ============================================================================

TRAINED_DIR    = Path('./trained_models_new')
CPH_MODEL_PATH = Path('./ALSdigitaltwin/cph_model.pkl')

INTERVAL_WIDTH = 30
NUM_INTERVALS  = 24
YEARS          = 5
HORIZON_DAYS   = YEARS * 365
EXTRAP_BINS    = HORIZON_DAYS // INTERVAL_WIDTH

DOMAIN_COLORS  = ['#2563EB', '#DC2626', '#16A34A', '#D97706', '#7C3AED']
DOMAIN_MAXES   = [DOMAIN_MAX[d] for d in DOMAINS]
DOMAIN_LABELS  = ['Bulbar', 'Upper limb', 'Axial', 'Lower limb', 'Respiratory']
DOMAIN_SHORT   = ['Bulbar', 'Upper limb', 'Axial', 'Lower limb', 'Resp']

ALSFRS_ITEM_LABELS = {
    'alsfrs1':  'Speech',
    'alsfrs2':  'Salivation',
    'alsfrs3':  'Swallowing',
    'alsfrs4':  'Handwriting',
    'alsfrs5':  'Cutting food / utensils',
    'alsfrs6':  'Dressing & hygiene',
    'alsfrs7':  'Turning in bed',
    'alsfrs8':  'Walking',
    'alsfrs9':  'Climbing stairs',
    'alsfrsr1': 'Dyspnea',
    'alsfrsr2': 'Orthopnea',
    'alsfrsr3': 'Respiratory insufficiency',
}

CPH_FEATURES = ['age_at_visit', 'FU_since_dx', 'sex',
                'alsfrs1', 'alsfrs2', 'alsfrs3', 'alsfrs4', 'alsfrs5',
                'alsfrs6', 'alsfrs7', 'alsfrs8', 'alsfrs9',
                'alsfrsr1', 'alsfrsr2', 'alsfrsr3']


# ============================================================================
# Model loading
# ============================================================================

@st.cache_resource
def load_stage_model():
    ckpt = torch.load(TRAINED_DIR / 'stage_predictor_best.pt',
                      map_location='cpu', weights_only=False)
    model = create_stage_predictor(ckpt['static_dim'], ckpt['temporal_dim'], ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    train_df = pd.read_csv(TRAINED_DIR / 'train_split.csv')
    for c in ['assessment_date', 'diagnosis_date', 'prev_date']:
        if c in train_df.columns:
            train_df[c] = pd.to_datetime(train_df[c])
    ds = ALSDataset(train_df, fit_scalers=True)
    return model, ds.scaler_static, ds.scaler_temporal, ds.static_features, ds.temporal_features


@st.cache_resource
def load_cph_model():
    if not CPH_MODEL_PATH.exists():
        return None
    try:
        with open(CPH_MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f'Failed to load CPH model: {e}')
        return None


# ============================================================================
# Stage-predictor inference (Mode 1)
# ============================================================================

def build_input_tensors(visits_df, scaler_static, scaler_temporal,
                        static_feat_names, temporal_feat_names):
    df = prepare_features(visits_df, static_feat_names, temporal_feat_names)

    static_cols  = [f for f in static_feat_names if f in df.columns]
    static_vals  = df[static_cols].iloc[0].values.astype(np.float32)
    static_scaled = scaler_static.transform(static_vals.reshape(1, -1)).flatten()
    static_t = torch.tensor(static_scaled, dtype=torch.float32).unsqueeze(0)

    temp_cols  = [f for f in temporal_feat_names if f in df.columns]
    temp_vals  = df[temp_cols].values.astype(np.float32)
    temp_scaled = scaler_temporal.transform(temp_vals)
    T, F = temp_scaled.shape
    temporal_t = torch.tensor(temp_scaled, dtype=torch.float32).unsqueeze(0)

    mask_cols = [f'{c}_missing' for c in temp_cols if f'{c}_missing' in df.columns]
    mask_vals = (1 - df[mask_cols].values.astype(np.float32)
                 if mask_cols else np.ones((T, F), dtype=np.float32))
    mask_t = torch.tensor(mask_vals, dtype=torch.float32).unsqueeze(0)

    td = (np.clip(df['days_since_prev_visit'].fillna(0).values / 30.0, 0, 12).astype(np.float32)
          if 'days_since_prev_visit' in df.columns else np.zeros(T, dtype=np.float32))
    td_t = torch.tensor(td, dtype=torch.float32).unsqueeze(0)
    seq_lens = torch.tensor([T], dtype=torch.long)

    last = visits_df.iloc[-1]
    dom_cols   = [f'domain_{d}' for d in DOMAINS]
    cur_scores = np.array([last.get(c, 0.0) / dm
                           for c, dm in zip(dom_cols, DOMAIN_MAXES)], dtype=np.float32)
    cur_t = torch.tensor(cur_scores, dtype=torch.float32).unsqueeze(0)

    return static_t, temporal_t, mask_t, td_t, seq_lens, cur_t


def run_stage_inference(model, static_t, temporal_t, mask_t, td_t, seq_lens, cur_t):
    with torch.no_grad():
        hazards_raw = model(static_t, temporal_t, mask_t, td_t, seq_lens, cur_t)

    hazards, surv_curves = {}, {}
    for d in DOMAINS:
        h       = hazards_raw[d][0].numpy()
        tail_h  = h[-3:].mean()
        h_full  = np.concatenate([h, np.full(EXTRAP_BINS - NUM_INTERVALS, tail_h)])
        surv    = np.cumprod(1 - h_full + 1e-8)
        hazards[d]     = h
        surv_curves[d] = surv
    return hazards, surv_curves


def predicted_time_at_k(surv, k):
    below = np.where(surv < k)[0]
    return (below[0] + 0.5) * INTERVAL_WIDTH if len(below) else None


def compute_domains(items):
    return {
        'domain_bulbar':      items['alsfrs1'] + items['alsfrs2'] + items['alsfrs3'],
        'domain_fine_motor':  items['alsfrs4'] + items['alsfrs5'],
        'domain_gross_motor': items['alsfrs6'] + items['alsfrs7'],
        'domain_walking':     items['alsfrs8'] + items['alsfrs9'],
        'domain_respiratory': items['alsfrsr1'] + items['alsfrsr2'] + items['alsfrsr3'],
    }


# ============================================================================
# CPH inference (Mode 2)
# ============================================================================

def run_cph_inference(cph, age_at_visit, fu_since_dx, sex_is_male, items):
    """Returns (times_months, survival_probs, median_str)."""
    input_data = {
        'age_at_visit': float(age_at_visit),
        'FU_since_dx':  float(fu_since_dx),
        'sex':          1.0 if sex_is_male else 0.0,
        **{k: float(v) for k, v in items.items()},
    }
    df = pd.DataFrame([input_data])[CPH_FEATURES]
    surv_func = cph.predict_survival_function(df)
    times = np.asarray(surv_func.index.values, dtype=float)
    probs = np.asarray(surv_func.iloc[:, 0].values, dtype=float)

    try:
        med = cph.predict_median(df)
        med_v = float(med.iloc[0]) if hasattr(med, 'iloc') else float(med)
        if np.isinf(med_v):
            median_str = 'Not reached in window'
        else:
            median_str = f'{med_v:.1f} months'
    except Exception:
        median_str = 'Unavailable'

    return times, probs, median_str


# ============================================================================
# Plot builders
# ============================================================================

def hex_to_rgba(hex_color, alpha=0.10):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def build_survival_figure(surv_curves, domains_vals, k):
    """5-panel interactive survival curve figure (stage predictor)."""
    t_days = (np.arange(EXTRAP_BINS) + 0.5) * INTERVAL_WIDTH
    t_yr   = t_days / 365
    model_cutoff_yr = NUM_INTERVALS * INTERVAL_WIDTH / 365

    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=DOMAIN_LABELS,
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    for row_i, (d, label, color, d_max) in enumerate(
        zip(DOMAINS, DOMAIN_LABELS, DOMAIN_COLORS, DOMAIN_MAXES), start=1
    ):
        surv    = surv_curves[d]
        dom_val = domains_vals[f'domain_{d}']
        t_pred  = predicted_time_at_k(surv, k)

        hover_text = [
            f"<b>{label}</b><br>"
            f"Time: {yr:.2f} yr ({day:.0f} days)<br>"
            f"P(in stage): {s*100:.1f}%<br>"
            f"Score: {dom_val:.0f}/{d_max}"
            for yr, day, s in zip(t_yr, t_days, surv)
        ]

        fig.add_vrect(x0=model_cutoff_yr, x1=YEARS, fillcolor='#94A3B8', opacity=0.08,
                      layer='below', line_width=0, row=row_i, col=1)
        fig.add_vline(x=model_cutoff_yr, line=dict(color='#94A3B8', width=1, dash='dot'),
                      row=row_i, col=1)

        fig.add_trace(go.Scatter(
            x=np.concatenate([t_yr, t_yr[::-1]]),
            y=np.concatenate([surv, np.zeros(EXTRAP_BINS)]),
            fill='toself', fillcolor=hex_to_rgba(color, 0.10),
            line=dict(width=0), hoverinfo='skip', showlegend=False,
        ), row=row_i, col=1)

        fig.add_trace(go.Scatter(
            x=t_yr, y=surv, mode='lines',
            line=dict(color=color, width=2.5),
            name=label, text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            showlegend=(row_i == 1),
        ), row=row_i, col=1)

        fig.add_hline(y=k, line=dict(color='#374151', width=1.5, dash='dash'),
                      row=row_i, col=1)

        if t_pred is not None:
            t_pred_yr    = t_pred / 365
            surv_at_pred = float(np.interp(t_pred_yr, t_yr, surv))
            fig.add_vline(x=t_pred_yr, line=dict(color=color, width=2, dash='dot'),
                          row=row_i, col=1)
            fig.add_trace(go.Scatter(
                x=[t_pred_yr], y=[surv_at_pred],
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=12, color=color,
                            line=dict(color='white', width=1.5)),
                text=[f'{t_pred_yr:.1f} yr'], textposition='top center',
                textfont=dict(size=10, color=color),
                hovertemplate=(
                    f'<b>Predicted drop ≥ 1 pt</b><br>'
                    f'Time: {t_pred_yr:.2f} yr ({t_pred:.0f} days)<br>'
                    f'P(in stage) = {surv_at_pred*100:.1f}%<extra></extra>'
                ),
                showlegend=False,
            ), row=row_i, col=1)
        else:
            fig.add_annotation(x=YEARS * 0.55, y=k + 0.07, text=f'> {YEARS} yr',
                               showarrow=False, font=dict(color=color, size=11),
                               row=row_i, col=1)

        fig.add_hrect(y0=0, y1=k, fillcolor='#DC2626', opacity=0.04,
                      layer='below', line_width=0, row=row_i, col=1)

        fig.add_annotation(
            x=YEARS * 0.92, y=0.88, text=f'Score {dom_val:.0f}/{d_max}',
            showarrow=False, font=dict(size=10, color=color),
            bgcolor='white', bordercolor=color, borderwidth=1,
            borderpad=3, opacity=0.90, row=row_i, col=1,
        )

    fig.update_layout(
        height=820,
        margin=dict(l=70, r=24, t=40, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#F8FAFC',
        font=dict(family='-apple-system, "Segoe UI", Inter, sans-serif', size=11),
        hovermode='x unified', showlegend=False,
    )
    fig.update_xaxes(range=[0, YEARS], tickvals=list(range(YEARS + 1)),
                     ticktext=[f'Y{y}' for y in range(YEARS + 1)],
                     gridcolor='rgba(148,163,184,0.25)', showline=True,
                     linecolor='#CBD5E1')
    fig.update_xaxes(title_text='Years from now', title_font_size=10, row=5, col=1)
    fig.update_yaxes(range=[-0.02, 1.06], tickformat='.0%',
                     title_text='P(in stage)', title_font_size=10,
                     gridcolor='rgba(148,163,184,0.25)', showline=True,
                     linecolor='#CBD5E1')
    for i, color in enumerate(DOMAIN_COLORS):
        fig.layout.annotations[i].font.color = color
        fig.layout.annotations[i].font.size  = 12
    return fig


def build_hazard_figure(hazards):
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=[l.split(' ')[0] for l in DOMAIN_LABELS],
        shared_xaxes=True, vertical_spacing=0.06,
    )
    model_bins_yr  = (np.arange(NUM_INTERVALS) + 0.5) * INTERVAL_WIDTH / 365
    extrap_bins_yr = (np.arange(NUM_INTERVALS, EXTRAP_BINS) + 0.5) * INTERVAL_WIDTH / 365
    bar_width      = INTERVAL_WIDTH / 365 * 0.85

    for row_i, (d, label, color) in enumerate(zip(DOMAINS, DOMAIN_LABELS, DOMAIN_COLORS), start=1):
        h      = hazards[d]
        tail_h = h[-3:].mean()
        fig.add_trace(go.Bar(
            x=model_bins_yr, y=h, width=bar_width,
            marker_color=color, marker_opacity=0.75,
            hovertemplate='%{x:.2f} yr<br>Hazard: %{y:.3f}<extra></extra>',
            showlegend=False,
        ), row=row_i, col=1)
        fig.add_trace(go.Bar(
            x=extrap_bins_yr,
            y=np.full(EXTRAP_BINS - NUM_INTERVALS, tail_h),
            width=bar_width, marker_color=color, marker_opacity=0.22,
            marker_line=dict(color=color, width=1),
            hovertemplate='%{x:.2f} yr<br>Hazard (extrap): %{y:.3f}<extra></extra>',
            showlegend=False,
        ), row=row_i, col=1)
        fig.add_vline(x=NUM_INTERVALS * INTERVAL_WIDTH / 365,
                      line=dict(color='#94A3B8', width=1, dash='dot'),
                      row=row_i, col=1)

    fig.update_layout(
        height=620, barmode='overlay',
        margin=dict(l=70, r=24, t=30, b=50),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#F8FAFC',
        font=dict(family='-apple-system, Inter, sans-serif', size=10),
        hovermode='x', bargap=0.05, showlegend=False,
    )
    fig.update_xaxes(range=[0, YEARS], tickvals=list(range(YEARS + 1)),
                     ticktext=[f'Y{y}' for y in range(YEARS + 1)],
                     gridcolor='rgba(148,163,184,0.2)')
    fig.update_xaxes(title_text='Years', title_font_size=9, row=5, col=1)
    fig.update_yaxes(title_text='Hazard h(t)', title_font_size=9,
                     gridcolor='rgba(148,163,184,0.2)')
    for i, color in enumerate(DOMAIN_COLORS):
        fig.layout.annotations[i].font.color = color
    return fig


def build_cph_figure(times_months, probs, median_v=None):
    """Single survival curve for the wheelchair-free CPH model."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([times_months, times_months[::-1]]),
        y=np.concatenate([probs, np.zeros_like(probs)]),
        fill='toself', fillcolor='rgba(37,99,235,0.10)',
        line=dict(width=0), hoverinfo='skip', showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=times_months, y=probs, mode='lines',
        line=dict(color='#2563EB', width=3),
        name='Wheelchair-free survival',
        hovertemplate='Month: %{x:.1f}<br>P(wheelchair-free): %{y:.1%}<extra></extra>',
    ))

    fig.add_hline(y=0.5, line=dict(color='#374151', width=1.2, dash='dash'),
                  annotation_text='50%', annotation_position='right')

    if median_v is not None and not np.isinf(median_v):
        fig.add_vline(x=float(median_v), line=dict(color='#DC2626', width=2, dash='dot'),
                      annotation_text=f'median ≈ {float(median_v):.1f} mo',
                      annotation_position='top')

    fig.update_layout(
        height=480, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#F8FAFC',
        margin=dict(l=70, r=30, t=40, b=60),
        font=dict(family='-apple-system, Inter, sans-serif', size=12),
        hovermode='x unified',
        title=dict(text='Probability of remaining wheelchair-free over time',
                   font=dict(size=14, color='#0F172A'), x=0.5, xanchor='center'),
        xaxis=dict(title='Months since current visit',
                   gridcolor='rgba(148,163,184,0.25)', showline=True, linecolor='#CBD5E1'),
        yaxis=dict(range=[0, 1.02], tickformat='.0%',
                   title='P(wheelchair-free)',
                   gridcolor='rgba(148,163,184,0.25)', showline=True, linecolor='#CBD5E1'),
    )
    return fig


# ============================================================================
# Page config + global styles
# ============================================================================

st.set_page_config(
    page_title='ALS Digital Twin',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
:root {
    --bg: #FAFBFC;
    --surface: #FFFFFF;
    --surface-2: #F4F6F9;
    --primary: #2563EB;
    --text: #111827;
    --text-dim: #6B7280;
    --border: #E5E7EB;
}
html, body, [class*="css"] {
    font-family: -apple-system, "Segoe UI", Inter, system-ui, sans-serif;
}
.main { background-color: var(--bg); }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }

/* ----- Top bar ----- */
.topbar {
    display: flex; align-items: baseline; justify-content: space-between;
    padding: 4px 2px 14px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 18px;
}
.topbar .title {
    font-size: 1.35rem; font-weight: 700; color: var(--text);
    letter-spacing: -0.01em; margin: 0;
}
.topbar .subtitle {
    font-size: 0.85rem; color: var(--text-dim);
    margin: 2px 0 0;
}
.topbar .tag {
    font-size: 0.7rem; font-weight: 600; color: var(--text-dim);
    background: var(--surface-2); padding: 3px 10px; border-radius: 4px;
    letter-spacing: 0.04em; text-transform: uppercase;
    border: 1px solid var(--border);
}

/* ----- Tabs / radio nav ----- */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div {
    flex-direction: row !important; gap: 0;
    border-bottom: 1px solid var(--border);
    width: 100%; margin-bottom: 4px;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: transparent; padding: 9px 18px;
    cursor: pointer; transition: all .15s ease; font-weight: 500;
    color: var(--text-dim); font-size: 0.92rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    color: var(--text);
}
div[data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] {
    color: var(--primary); border-bottom-color: var(--primary);
}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none;
}

/* ----- Metric tiles ----- */
.metric-card {
    background: var(--surface); border-radius: 6px; padding: 12px 14px;
    border: 1px solid var(--border);
    margin-bottom: 8px;
}
.metric-card .label {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--text-dim); margin-bottom: 4px;
}
.metric-card .value {
    font-size: 1.45rem; font-weight: 600; color: var(--text);
    line-height: 1.15;
}
.metric-card .sub {
    font-size: 0.75rem; color: var(--text-dim); margin-top: 2px;
}

/* ----- Section header (no card wrapper) ----- */
.section-title {
    font-size: 0.98rem; font-weight: 600; color: var(--text);
    margin: 18px 0 2px; letter-spacing: -0.005em;
}
.section-subtitle {
    font-size: 0.82rem; color: var(--text-dim); margin: 0 0 10px;
}
.section-divider {
    border: 0; border-top: 1px solid var(--border); margin: 8px 0 14px;
}

/* ----- Chart panel ----- */
.panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 14px 16px; margin-bottom: 14px;
}

/* ----- Sidebar polish ----- */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: var(--text);
}

/* ----- Tables ----- */
[data-testid="stDataFrame"] {
    border-radius: 6px; overflow: hidden; border: 1px solid var(--border);
}

/* ----- Expander ----- */
[data-testid="stExpander"] {
    border: 1px solid var(--border); border-radius: 6px;
    background: var(--surface);
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Top bar + page navigation
# ============================================================================

st.markdown("""
<div class="topbar">
  <div>
    <h1 class="title">ALS Digital Twin</h1>
    <p class="subtitle">Domain-level stage progression and wheelchair-free survival from a single visit.</p>
  </div>
  <span class="tag">Research use only</span>
</div>
""", unsafe_allow_html=True)

PAGE_STAGE = 'Domain Stage Predictor'
PAGE_CPH   = 'Wheelchair-Free Survival'

page = st.radio(
    'Select mode',
    [PAGE_STAGE, PAGE_CPH],
    horizontal=True,
    label_visibility='collapsed',
)


# ============================================================================
# Sidebar — inputs (shared)
# ============================================================================

with st.sidebar:
    st.markdown('### Patient Input')

    st.markdown('#### Demographics')
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input('Age (yr)', 20, 95, 60, step=1)
    with col_b:
        months_dx = st.number_input('Months since Dx', 0, 240, 12, step=1)
    sex       = st.selectbox('Sex', ['Male', 'Female'])

    if page == PAGE_STAGE:
        ethnicity = st.selectbox('Ethnicity', ['Not Hispanic or Latino', 'Hispanic or Latino'])
        race      = st.selectbox('Race', ['White', 'Black/African American', 'Asian', 'Other'])
    else:
        ethnicity, race = 'Not Hispanic or Latino', 'White'

    st.markdown('---')
    st.markdown('#### ALSFRS-R Items')
    st.caption('Each item: 0 = unable → 4 = normal')
    items = {}
    for group_name, group_items in [
        ('Bulbar',      ['alsfrs1', 'alsfrs2', 'alsfrs3']),
        ('Upper limb',  ['alsfrs4', 'alsfrs5', 'alsfrs6']),
        ('Axial',       ['alsfrs7']),
        ('Lower limb',  ['alsfrs8', 'alsfrs9']),
        ('Respiratory', ['alsfrsr1', 'alsfrsr2', 'alsfrsr3']),
    ]:
        with st.expander(group_name, expanded=(group_name == 'Bulbar')):
            for item in group_items:
                items[item] = st.slider(ALSFRS_ITEM_LABELS[item], 0, 4, 4, key=item)

    if page == PAGE_STAGE:
        st.markdown('---')
        st.markdown('#### Previous Visit (optional)')
        use_prev = st.checkbox('Include a previous visit')
        prev_days_ago, prev_alsfrs = 90, None
        if use_prev:
            prev_days_ago = st.slider('Days before current visit', 30, 720, 90)
            prev_alsfrs   = st.slider('Previous ALSFRS total', 0, 48, 40)

        st.markdown('---')
        st.markdown('#### Prediction Threshold')
        k = st.slider(
            'k — survival probability trigger', 0.10, 0.95, 0.70, 0.05,
            help='Predicted transition = first day P(still in stage) < k.\n'
                 'Lower k → later (more optimistic) prediction.',
        )
        st.caption(f'Flags when > {(1-k)*100:.0f}% chance of stage drop has occurred')
    else:
        use_prev, prev_days_ago, prev_alsfrs, k = False, 90, None, 0.5


# ============================================================================
# Mode 1 — Domain Stage Predictor
# ============================================================================

def render_stage_predictor():
    model, scaler_s, scaler_t, static_feats, temporal_feats = load_stage_model()

    domains      = compute_domains(items)
    alsfrs_total = sum(domains.values())
    pct_total    = alsfrs_total / 48 * 100

    def make_row(alsfrs_t, dom_vals, item_vals, months, days_prev):
        d = {
            'SubjectUID': 'DEMO',
            'assessment_date': pd.Timestamp.today(),
            'diagnosis_date':  pd.Timestamp.today() - pd.Timedelta(days=months * 30.44),
            'alsfrs_total': float(alsfrs_t),
            **{f'domain_{d}': float(v) for d, v in zip(DOMAINS, dom_vals.values())},
            **{f'pct_{d}': float(v / DOMAIN_MAXES[i] * 100)
               for i, (d, v) in enumerate(zip(DOMAINS, dom_vals.values()))},
            'pct_total': float(alsfrs_t / 48 * 100),
            **{k_: float(v) for k_, v in item_vals.items()},
            'age_at_diagnosis':       float(age),
            'months_since_diagnosis': float(months),
            'days_since_prev_visit':  float(days_prev),
            'is_female':   1.0 if sex == 'Female' else 0.0,
            'is_hispanic': 1.0 if ethnicity == 'Hispanic or Latino' else 0.0,
            'race_white':  1.0 if race == 'White' else 0.0,
            'race_black':  1.0 if race == 'Black/African American' else 0.0,
            'race_asian':  1.0 if race == 'Asian' else 0.0,
            'race_other':  1.0 if race == 'Other' else 0.0,
        }
        return d

    rows = []
    if use_prev and prev_alsfrs is not None:
        scale       = prev_alsfrs / max(alsfrs_total, 1)
        prev_items  = {k_: min(4, round(v * scale)) for k_, v in items.items()}
        prev_doms   = compute_domains(prev_items)
        prev_row    = make_row(sum(prev_doms.values()), prev_doms, prev_items,
                               max(0.0, months_dx - prev_days_ago / 30.44), 0.0)
        prev_row['assessment_date'] = pd.Timestamp.today() - pd.Timedelta(days=prev_days_ago)
        rows.append(prev_row)

    rows.append(make_row(alsfrs_total, domains, items, months_dx,
                         float(prev_days_ago) if use_prev else 0.0))
    visits_df = pd.DataFrame(rows).sort_values('assessment_date').reset_index(drop=True)

    static_t, temporal_t, mask_t, td_t, seq_lens, cur_t = build_input_tensors(
        visits_df, scaler_s, scaler_t, static_feats, temporal_feats
    )
    hazards, surv_curves = run_stage_inference(
        model, static_t, temporal_t, mask_t, td_t, seq_lens, cur_t
    )

    t_yr = (np.arange(EXTRAP_BINS) + 0.5) * INTERVAL_WIDTH / 365

    # ── metric cards ───────────────────────────────────────────────────────
    cols = st.columns([1, 1, 1, 1, 1, 1])
    cols[0].markdown(
        f'<div class="metric-card">'
        f'<div class="label">ALSFRS Total</div>'
        f'<div class="value">{alsfrs_total}/48</div>'
        f'<div class="sub">{pct_total:.0f}% of max</div></div>',
        unsafe_allow_html=True,
    )
    for i, (d, label, color) in enumerate(zip(DOMAINS, DOMAIN_SHORT, DOMAIN_COLORS)):
        dom_val = domains[f'domain_{d}']
        t_pred  = predicted_time_at_k(surv_curves[d], k)
        pred_s  = f'{t_pred/365:.1f} yr' if t_pred is not None else f'> {YEARS} yr'
        cols[i + 1].markdown(
            f'<div class="metric-card">'
            f'<div class="label" style="color:{color}">{label}</div>'
            f'<div class="value">{dom_val:.0f}/{DOMAIN_MAXES[i]}</div>'
            f'<div class="sub">Drop predicted: {pred_s}</div></div>',
            unsafe_allow_html=True,
        )

    # ── survival curves ────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-title">Per-domain survival curves</div>'
        f'<div class="section-subtitle">P(still in current stage) over time. '
        f'Threshold <b>k = {k:.2f}</b>; grey shading = extrapolated beyond 2 yr.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_survival_figure(surv_curves, domains, k),
        use_container_width=True,
    )

    # ── prediction table ───────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-title">Predicted stage transition times</div>'
        f'<div class="section-subtitle">Time when P(in stage) first drops below k = {k:.2f}.</div>',
        unsafe_allow_html=True,
    )

    RISK_FLAGS = [(180, '🔴 High (<6 mo)'), (365, '🟠 Moderate-High (<1 yr)'),
                  (730, '🟡 Moderate (<2 yr)'), (float('inf'), '🟢 Low-Moderate (>2 yr)')]

    table_rows = []
    for d, label, d_max in zip(DOMAINS, DOMAIN_LABELS, DOMAIN_MAXES):
        surv   = surv_curves[d]
        t_pred = predicted_time_at_k(surv, k)
        dom_v  = domains[f'domain_{d}']
        if t_pred is None:
            pred_s = f'> {YEARS} yr'
            risk   = f'✅ Low (>{surv[-1]*100:.0f}% in-stage @ {YEARS} yr)'
        else:
            pred_s = f'{t_pred:.0f} d  ({t_pred/365:.1f} yr)'
            risk   = next(flag for cutoff, flag in RISK_FLAGS if t_pred < cutoff)
        table_rows.append({
            'Domain': label, 'Score': f'{dom_v:.0f}/{d_max}  ({dom_v/d_max*100:.0f}%)',
            'Predicted Drop Time': pred_s, 'Risk': risk,
            'P @ 6 mo': f'{np.interp(0.5, t_yr, surv)*100:.0f}%',
            'P @ 1 yr': f'{np.interp(1.0, t_yr, surv)*100:.0f}%',
            'P @ 2 yr': f'{np.interp(2.0, t_yr, surv)*100:.0f}%',
            'P @ 5 yr': f'{float(surv[-1])*100:.0f}%',
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ── hazard chart ───────────────────────────────────────────────────────
    with st.expander('Instantaneous hazard rates (per 30-day bin)', expanded=False):
        st.plotly_chart(build_hazard_figure(hazards), use_container_width=True)

    # ── model details ──────────────────────────────────────────────────────
    with st.expander('Model & Data Details'):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
**Model:** Domain Stage Predictor
**Architecture:** Transformer encoder (d=128, 4 heads, 3 layers) + 5 hazard heads
**Training:** 1,385 visits · 186 patients
**Model horizon:** 720 days (24 × 30-day bins)
**5-year ext.:** constant hazard extrapolation
            """)
        with c2:
            st.markdown(f"""
**Static features:** {len(static_feats)} (age, sex, ethnicity, race)
**Temporal features:** {len(temporal_feats)} (ALSFRS scores, domain %, time)
**Stage transition:** score drop ≥ 1 point
**Current k = {k:.2f}** → predicts when P(in stage) < {k:.2f}
            """)


# ============================================================================
# Mode 2 — Wheelchair-Free Survival (CPH)
# ============================================================================

def render_cph_survival():
    cph = load_cph_model()
    if cph is None:
        st.error(
            'CPH model not found at `./ALSdigitaltwin/cph_model.pkl`. '
            'Make sure the file exists, then refresh the page.'
        )
        return

    times, probs, median_str = run_cph_inference(
        cph, age, months_dx, sex == 'Male', items
    )
    median_v = None
    if 'months' in median_str:
        try:
            median_v = float(median_str.split()[0])
        except Exception:
            median_v = None

    HORIZONS = [(6, 'P @ 6 mo'), (12, 'P @ 1 yr'), (24, 'P @ 2 yr'),
                (36, 'P @ 3 yr'), (60, 'P @ 5 yr')]

    def p_at(months):
        return float(np.interp(months, times, probs)) if len(times) else float('nan')

    # ── metric cards ───────────────────────────────────────────────────────
    n_cards = 1 + len(HORIZONS)
    cols = st.columns([1.3] + [1] * len(HORIZONS))
    cols[0].markdown(
        f'<div class="metric-card">'
        f'<div class="label">Median time to wheelchair</div>'
        f'<div class="value" style="color:#2563EB">{median_str}</div>'
        f'<div class="sub">50% probability crossing</div></div>',
        unsafe_allow_html=True,
    )
    for i, (m, label) in enumerate(HORIZONS):
        p = p_at(m) * 100
        # colour-code: high prob (low survival) = warmer tones
        if p >= 80:    color = '#10B981'   # likely wheelchair-free
        elif p >= 60:  color = '#3B82F6'
        elif p >= 40:  color = '#F59E0B'
        elif p >= 20:  color = '#F97316'
        else:          color = '#EF4444'
        cols[i + 1].markdown(
            f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value" style="color:{color}">{p:.0f}%</div>'
            f'<div class="sub">wheelchair-free</div></div>',
            unsafe_allow_html=True,
        )

    # ── survival curve ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">Wheelchair-free survival curve</div>'
        '<div class="section-subtitle">Cox proportional-hazards model. '
        'The curve drops below 50% at the predicted median time.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(build_cph_figure(times, probs, median_v), use_container_width=True)

    # ── milestone table ────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">Wheelchair-free probability at horizons</div>'
        '<div class="section-subtitle">Discrete probabilities sampled along the survival curve.</div>',
        unsafe_allow_html=True,
    )
    rows = []
    for m, label in HORIZONS:
        p = p_at(m) * 100
        if p >= 80:    risk = '🟢 Low risk'
        elif p >= 60:  risk = '🔵 Mild'
        elif p >= 40:  risk = '🟡 Moderate'
        elif p >= 20:  risk = '🟠 High'
        else:          risk = '🔴 Very high'
        rows.append({'Horizon': label.split('@ ')[1], 'P(wheelchair-free)': f'{p:.0f}%',
                     'P(wheelchair by then)': f'{100-p:.0f}%', 'Risk band': risk})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── model details ──────────────────────────────────────────────────────
    with st.expander('Model & Data Details'):
        st.markdown(f"""
**Model:** Cox Proportional-Hazards (`lifelines.CoxPHFitter`)
**Endpoint:** Time to wheelchair access (months)
**Features ({len(CPH_FEATURES)}):** age at visit, follow-up months since Dx,
sex, and 12 ALSFRS-R items (alsfrs1–9, alsfrsr1–3)
**Training data:** Temple University ALS dataset
**Output:** Predicted survival function `S(t)` and median time-to-event
        """)
        st.info('Research use only — not for clinical decisions.')


# ============================================================================
# Dispatch
# ============================================================================

if page == PAGE_STAGE:
    render_stage_predictor()
else:
    render_cph_survival()


# ============================================================================
# Footer
# ============================================================================

st.markdown(
    '<div style="text-align:center; color:#94A3B8; font-size:0.78rem; '
    'margin-top:30px; padding-top:18px; border-top:1px solid #E2E8F0;">'
    'AI.pharm Lab · Auburn University · &copy; 2026 — Research use only'
    '</div>',
    unsafe_allow_html=True,
)
