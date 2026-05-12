# ALS Digital Twin — Streamlit App

Interactive web app with two prediction modes:

1. **Domain Stage Predictor** — Transformer model that predicts when each of
   the 5 ALSFRS-R domains will drop by ≥1 point (24 × 30-day bins, 2-year
   horizon with constant-hazard extrapolation to 5 years).
2. **Wheelchair-Free Survival** — Cox proportional-hazards model that predicts
   time-to-wheelchair-access from a single visit.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Repository layout

```
.
├── app.py                                    # Streamlit entry point
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── stage_predictor.py                    # Transformer + 5 hazard heads
│   └── data_loader.py                        # ALSDataset, feature prep
├── trained_models_new/
│   ├── stage_predictor_best.pt               # Stage predictor checkpoint
│   ├── stage_predictor_config.json
│   └── train_split.csv                       # Used to refit scalers on load
└── ALSdigitaltwin/
    └── cph_model.pkl                         # lifelines CoxPHFitter (pickled)
```

## How files are wired

- `app.py` loads the stage model from
  `./trained_models_new/stage_predictor_best.pt` and rebuilds the static /
  temporal scalers from `./trained_models_new/train_split.csv` (Streamlit
  `@st.cache_resource` keeps both warm across reruns).
- The CPH model is loaded from `./ALSdigitaltwin/cph_model.pkl` via `pickle`,
  so the `lifelines` version at inference must be compatible with the version
  it was trained with.
- Patient inputs come from the sidebar (demographics + 12 ALSFRS-R item
  sliders). The stage mode also accepts an optional previous visit.

## Notes

- Research use only — not for clinical decisions.
- The stage checkpoint is ~9 MB and committed directly (no Git LFS required).
