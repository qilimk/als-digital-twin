"""
Classical Baseline Models for ALS Digital Twin.

Includes:
1. Linear Mixed Effects for next-state prediction
2. Random Survival Forest for event prediction
3. XGBoost for both tasks
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import survival analysis libraries
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False
    print("Warning: scikit-survival not installed. Survival models will be simplified.")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not installed. Using sklearn alternatives.")


# Feature columns
BASELINE_FEATURES = [
    'age_at_diagnosis', 'is_female', 'is_hispanic',
    'race_white', 'race_black', 'race_asian', 'race_other',
    'el_escorial', 'umn_burden', 'lmn_burden', 'emg_burden'
]

CURRENT_STATE_FEATURES = [
    'alsfrs_total', 'domain_bulbar', 'domain_fine_motor',
    'domain_gross_motor', 'domain_walking', 'domain_respiratory',
    'months_since_diagnosis', 'visit_num'
]

SLOPE_FEATURES = [
    'alsfrs_total_slope', 'domain_bulbar_slope', 'domain_fine_motor_slope',
    'domain_gross_motor_slope', 'domain_walking_slope', 'domain_respiratory_slope'
]

STATE_TARGETS = [
    'alsfrs_total', 'domain_bulbar', 'domain_fine_motor',
    'domain_gross_motor', 'domain_walking', 'domain_respiratory'
]

EVENT_TARGETS = [
    ('days_to_death', 'future_death'),
    ('days_to_gastrostomy', 'future_gastrostomy'),
    ('days_to_niv', 'future_niv'),
    ('days_to_wheelchair', 'future_wheelchair'),
    ('days_to_speech_loss', 'future_speech_loss'),
]


def prepare_classical_features(df):
    """Prepare features for classical models."""
    df = df.copy()

    # All features
    all_features = BASELINE_FEATURES + CURRENT_STATE_FEATURES + SLOPE_FEATURES

    # Fill missing values and handle infinities
    for col in all_features:
        if col in df.columns:
            # Replace infinities with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Fill NaN with median or 0
            if df[col].dtype in ['float64', 'int64', 'float32']:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
            else:
                df[col] = df[col].fillna(0)
            # Clip extreme values
            if df[col].dtype in ['float64', 'int64', 'float32']:
                q01 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                if not pd.isna(q01) and not pd.isna(q99):
                    df[col] = df[col].clip(q01, q99)

    # Filter to available features
    available_features = [f for f in all_features if f in df.columns]

    return df, available_features


class NextStatePredictor:
    """
    Predicts next-visit ALSFRS-R scores using gradient boosting.
    Serves as a strong baseline for state prediction.
    """

    def __init__(self, use_xgb=True):
        self.use_xgb = use_xgb and HAS_XGB
        self.models = {}
        self.feature_cols = None

    def fit(self, df_train, targets=None):
        """Fit models for each target."""
        targets = targets or STATE_TARGETS

        df_train, self.feature_cols = prepare_classical_features(df_train)

        # Create next-visit targets
        df_train = df_train.sort_values(['SubjectUID', 'assessment_date'])

        for target in targets:
            # Get next visit value as target
            target_col = f'next_{target}'
            df_train[target_col] = df_train.groupby('SubjectUID')[target].shift(-1)

            # Filter to valid rows (have next visit)
            valid = df_train[target_col].notna() & df_train[self.feature_cols].notna().all(axis=1)
            X = df_train.loc[valid, self.feature_cols].values
            y = df_train.loc[valid, target_col].values

            if len(X) < 10:
                print(f"Warning: Not enough data for {target}")
                continue

            # Fit model
            if self.use_xgb:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )

            model.fit(X, y)
            self.models[target] = model
            print(f"  Fitted model for {target}")

    def predict(self, df):
        """Predict next-visit values."""
        df, _ = prepare_classical_features(df)
        X = df[self.feature_cols].values

        predictions = {}
        for target, model in self.models.items():
            predictions[target] = model.predict(X)

        return predictions

    def evaluate(self, df_test):
        """Evaluate on test set."""
        df_test, _ = prepare_classical_features(df_test)
        df_test = df_test.sort_values(['SubjectUID', 'assessment_date'])

        results = {}
        for target in self.models.keys():
            # Get actual next values
            target_col = f'next_{target}'
            df_test[target_col] = df_test.groupby('SubjectUID')[target].shift(-1)

            valid = df_test[target_col].notna() & df_test[self.feature_cols].notna().all(axis=1)
            X = df_test.loc[valid, self.feature_cols].values
            y_true = df_test.loc[valid, target_col].values

            if len(X) == 0:
                continue

            y_pred = self.models[target].predict(X)

            results[target] = {
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }

        return results


class SurvivalPredictor:
    """
    Predicts time-to-event using Random Survival Forest or simplified model.
    """

    def __init__(self):
        self.models = {}
        self.feature_cols = None
        self.use_rsf = HAS_SKSURV

    def fit(self, df_train, events=None):
        """Fit survival models for each event type."""
        events = events or EVENT_TARGETS

        df_train, self.feature_cols = prepare_classical_features(df_train)

        for time_col, event_col in events:
            # Prepare survival data
            valid = (
                df_train[time_col].notna() &
                df_train[event_col].notna() &
                df_train[self.feature_cols].notna().all(axis=1) &
                (df_train[time_col] > 0)  # Positive times only
            )

            X = df_train.loc[valid, self.feature_cols].values
            times = df_train.loc[valid, time_col].values
            events = df_train.loc[valid, event_col].values.astype(bool)

            if len(X) < 10:
                print(f"Warning: Not enough data for {time_col}")
                continue

            event_name = time_col.replace('days_to_', '')

            if self.use_rsf:
                # Use Random Survival Forest
                y = Surv.from_arrays(events, times)
                model = RandomSurvivalForest(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y)
            else:
                # Simplified: predict log(time) with regression
                # Only use observed events for training
                if events.sum() > 10:
                    X_events = X[events]
                    y_events = np.log(times[events] + 1)
                    model = GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=5,
                        random_state=42
                    )
                    model.fit(X_events, y_events)
                else:
                    model = None

            self.models[event_name] = {
                'model': model,
                'time_col': time_col,
                'event_col': event_col.replace('future_', 'event_')
            }
            print(f"  Fitted survival model for {event_name}")

    def predict_risk(self, df, time_horizon=365):
        """
        Predict risk scores at a given time horizon.
        """
        df, _ = prepare_classical_features(df)
        X = df[self.feature_cols].values

        predictions = {}
        for event_name, model_info in self.models.items():
            model = model_info['model']
            if model is None:
                continue

            if self.use_rsf:
                # Get survival function and evaluate at time horizon
                surv_funcs = model.predict_survival_function(X)
                risks = []
                for sf in surv_funcs:
                    # Find survival probability at time horizon
                    if time_horizon in sf.x:
                        idx = np.where(sf.x == time_horizon)[0][0]
                    else:
                        idx = np.searchsorted(sf.x, time_horizon)
                        idx = min(idx, len(sf.y) - 1)
                    risks.append(1 - sf.y[idx])  # Risk = 1 - survival
                predictions[event_name] = np.array(risks)
            else:
                # Simplified: predict log time, convert to risk
                log_time_pred = model.predict(X)
                time_pred = np.exp(log_time_pred) - 1
                # Higher predicted time = lower risk
                predictions[event_name] = 1 / (1 + time_pred / time_horizon)

        return predictions

    def predict_median_time(self, df):
        """
        Predict median survival time for each event.
        """
        df, _ = prepare_classical_features(df)
        X = df[self.feature_cols].values

        predictions = {}
        for event_name, model_info in self.models.items():
            model = model_info['model']
            if model is None:
                continue

            if self.use_rsf:
                surv_funcs = model.predict_survival_function(X)
                median_times = []
                for sf in surv_funcs:
                    # Find time where survival = 0.5
                    below_50 = sf.y <= 0.5
                    if below_50.any():
                        idx = np.argmax(below_50)
                        median_times.append(sf.x[idx])
                    else:
                        median_times.append(sf.x[-1])  # Use max observed time
                predictions[event_name] = np.array(median_times)
            else:
                log_time_pred = model.predict(X)
                predictions[event_name] = np.exp(log_time_pred) - 1

        return predictions

    def evaluate(self, df_test):
        """Evaluate survival models."""
        df_test, _ = prepare_classical_features(df_test)

        results = {}
        for event_name, model_info in self.models.items():
            if model_info['model'] is None:
                continue

            time_col = model_info['time_col']
            event_col = model_info['event_col'].replace('event_', 'future_')

            valid = (
                df_test[time_col].notna() &
                df_test[event_col].notna() &
                df_test[self.feature_cols].notna().all(axis=1) &
                (df_test[time_col] > 0)
            )

            if valid.sum() < 10:
                continue

            X = df_test.loc[valid, self.feature_cols].values
            times = df_test.loc[valid, time_col].values
            events = df_test.loc[valid, event_col].values.astype(bool)

            if self.use_rsf:
                # C-index
                y = Surv.from_arrays(events, times)
                c_index = self.models[event_name]['model'].score(X, y)
                results[event_name] = {'c_index': c_index}
            else:
                # For simplified model, compute correlation
                pred_times = self.predict_median_time(df_test.loc[valid])
                if event_name in pred_times:
                    # Only compare for events that occurred
                    event_mask = events
                    if event_mask.sum() > 5:
                        correlation = np.corrcoef(
                            times[event_mask],
                            pred_times[event_name][event_mask]
                        )[0, 1]
                        results[event_name] = {'correlation': correlation}

        return results


class ClassicalDigitalTwin:
    """
    Combined classical baseline for Digital Twin.
    Wraps state and survival predictors.
    """

    def __init__(self):
        self.state_predictor = NextStatePredictor()
        self.survival_predictor = SurvivalPredictor()

    def fit(self, df_train):
        """Fit all models."""
        print("\nFitting state prediction models...")
        self.state_predictor.fit(df_train)

        print("\nFitting survival models...")
        self.survival_predictor.fit(df_train)

    def predict(self, df, survival_horizon=365):
        """Get all predictions."""
        state_preds = self.state_predictor.predict(df)
        risk_preds = self.survival_predictor.predict_risk(df, survival_horizon)
        time_preds = self.survival_predictor.predict_median_time(df)

        return {
            'state': state_preds,
            'risk': risk_preds,
            'median_times': time_preds
        }

    def evaluate(self, df_test):
        """Evaluate all models."""
        state_results = self.state_predictor.evaluate(df_test)
        survival_results = self.survival_predictor.evaluate(df_test)

        return {
            'state': state_results,
            'survival': survival_results
        }
