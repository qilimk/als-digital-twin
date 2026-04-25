"""
ALS Digital Twin Data Cleaning Pipeline

This script cleans and prepares the ALS Natural History dataset for training
a digital twin model. It creates:
1. Patient-level static features (demographics, diagnosis)
2. Longitudinal ALSFRS-R data with domain scores
3. Event outcomes (NIV, gastrostomy, wheelchair, death, speech loss)
4. Visit-level landmark instances for training

Based on the research design outlined in the project documentation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./cleaned_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ALSFRS-R Domain Definitions (based on standard groupings)
# Domain 1: Bulbar (speech, salivation, swallowing)
# Domain 2: Fine Motor (handwriting, cutting food/handling utensils)
# Domain 3: Gross Motor (dressing/hygiene, turning in bed)
# Domain 4: Walking (walking, climbing stairs)
# Domain 5: Respiratory (dyspnea, orthopnea, respiratory insufficiency)

ALSFRS_DOMAINS = {
    'bulbar': ['alsfrs1', 'alsfrs2', 'alsfrs3'],  # speech, salivation, swallowing
    'fine_motor': ['alsfrs4', 'alsfrs5', 'alsfrs5a', 'alsfrs5b'],  # handwriting, cutting food (with gastrostomy branch)
    'gross_motor': ['alsfrs6', 'alsfrs7'],  # dressing/hygiene, turning in bed
    'walking': ['alsfrs8', 'alsfrs9'],  # walking, climbing stairs
    'respiratory': ['alsfrsr1', 'alsfrsr2', 'alsfrsr3']  # dyspnea, orthopnea, respiratory insufficiency
}

# Domain max scores
DOMAIN_MAX = {
    'bulbar': 12,  # 3 items x 4 max
    'fine_motor': 8,  # 2 items x 4 max (using either cutting food or gastrostomy branch)
    'gross_motor': 8,  # 2 items x 4 max
    'walking': 8,  # 2 items x 4 max
    'respiratory': 12  # 3 items x 4 max
}

ALSFRS_TOTAL_MAX = 48

# Assistive device codes (from the data dictionary)
DEVICE_CODES = {
    'cane': [3],
    'walker': [4],
    'manual_wheelchair': [5],
    'power_wheelchair': [6, 7],  # power scooter, power wheelchair
    'afo': [11],  # ankle foot orthosis
    'hospital_bed': [17],
    'bipap': [19],  # NIV/BiPAP
    'cpap': [18],
    'cough_assist': [23],
    'communication_device': [34, 35, 36],  # eye gaze, speech generating, other
}

# ============================================================================
# Utility Functions
# ============================================================================

def parse_date(date_str):
    """Parse various date formats in the dataset."""
    if pd.isna(date_str) or date_str in ['NULL', '', 'NaN']:
        return pd.NaT

    date_str = str(date_str).strip()

    # Try multiple formats
    formats = [
        '%m/%d/%Y',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%m/%d/%y',
    ]

    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue

    # Last resort: let pandas figure it out
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.NaT


def clean_numeric(value):
    """Convert to numeric, handling NaN and string 'NaN'."""
    if pd.isna(value) or value == 'NaN' or value == '':
        return np.nan
    try:
        return float(value)
    except:
        return np.nan


def compute_domain_score(row, items, use_gastrostomy_branch=False):
    """
    Compute domain score from ALSFRS-R items.
    For fine motor domain, handle the cutting food/gastrostomy branch.
    """
    if 'alsfrs5' in items:  # Fine motor domain
        # Use alsfrs5a (cutting without gastrostomy) or alsfrs5b (cutting with gastrostomy)
        # Based on which one is available
        base_items = ['alsfrs4']

        # Check if using gastrostomy branch
        if pd.notna(row.get('alsfrs5b')) and row.get('alsfrs5b') != 'NaN':
            branch_item = 'alsfrs5b'
        elif pd.notna(row.get('alsfrs5a')) and row.get('alsfrs5a') != 'NaN':
            branch_item = 'alsfrs5a'
        else:
            branch_item = 'alsfrs5'  # Original

        items_to_sum = base_items + [branch_item]
    else:
        items_to_sum = items

    scores = []
    for item in items_to_sum:
        val = row.get(item)
        if pd.notna(val) and val != 'NaN':
            try:
                scores.append(float(val))
            except:
                pass

    if len(scores) == 0:
        return np.nan

    return sum(scores)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_demographics():
    """Load and clean demographics data."""
    print("Loading Demographics...")
    df = pd.read_csv(DATA_DIR / "Demographics.csv")

    # Select relevant columns
    cols = ['SubjectUID', 'internal_subject_id', 'Visit Date', 'dob', 'age', 'sex',
            'ethnic', 'racewt', 'raceblk', 'raceasn', 'racenh', 'raceamin']
    df = df[cols].copy()

    # Clean data
    df['visit_date'] = df['Visit Date'].apply(parse_date)
    df['dob'] = df['dob'].apply(parse_date)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['sex'] = pd.to_numeric(df['sex'], errors='coerce')
    df['ethnic'] = pd.to_numeric(df['ethnic'], errors='coerce')

    # Race columns
    for col in ['racewt', 'raceblk', 'raceasn', 'racenh', 'raceamin']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Create combined race variable
    df['race_white'] = (df['racewt'] == 1).astype(int)
    df['race_black'] = (df['raceblk'] == 1).astype(int)
    df['race_asian'] = (df['raceasn'] == 1).astype(int)
    df['race_other'] = ((df['racenh'] == 1) | (df['raceamin'] == 1)).astype(int)

    # Sex: 1=Male, 2=Female -> binary
    df['is_female'] = (df['sex'] == 2).astype(int)

    # Ethnicity: 1=Hispanic/Latino, 2=Not Hispanic/Latino
    df['is_hispanic'] = (df['ethnic'] == 1).astype(int)

    # Keep one row per subject (baseline)
    df = df.drop_duplicates(subset=['SubjectUID'], keep='first')

    result = df[['SubjectUID', 'internal_subject_id', 'dob', 'age', 'is_female',
                 'is_hispanic', 'race_white', 'race_black', 'race_asian', 'race_other']].copy()

    print(f"  Loaded {len(result)} patients")
    return result


def load_diagnosis():
    """Load and clean ALS diagnosis data with UMN/LMN findings."""
    print("Loading ALS Diagnosis...")
    df = pd.read_csv(DATA_DIR / "ALS Diagnosis (ALS Natural History).csv")

    # Diagnosis date
    df['diagnosis_date'] = df['alsdxdt'].apply(parse_date)

    # El Escorial classification
    df['el_escorial'] = pd.to_numeric(df['elescrlr'], errors='coerce')

    # UMN/LMN/EMG findings by region
    # Columns: {region}c{sign} where region in [blb, lue, rue, trnk, lle, rle]
    # and sign in [umn, lmn, elmn]
    # Values: 1=present, 2=absent, 90=not assessed

    regions = ['blb', 'lue', 'rue', 'trnk', 'lle', 'rle']
    signs = ['umn', 'lmn', 'elmn']

    for region in regions:
        for sign in signs:
            col = f'{region}c{sign}'
            if col in df.columns:
                df[f'{region}_{sign}'] = df[col].apply(
                    lambda x: 1 if clean_numeric(x) == 1 else (0 if clean_numeric(x) == 2 else np.nan)
                )

    # Aggregate UMN/LMN burden
    umn_cols = [f'{r}_umn' for r in regions if f'{r}_umn' in df.columns]
    lmn_cols = [f'{r}_lmn' for r in regions if f'{r}_lmn' in df.columns]
    emg_cols = [f'{r}_elmn' for r in regions if f'{r}_elmn' in df.columns]

    df['umn_burden'] = df[umn_cols].sum(axis=1, skipna=True)
    df['lmn_burden'] = df[lmn_cols].sum(axis=1, skipna=True)
    df['emg_burden'] = df[emg_cols].sum(axis=1, skipna=True)

    # Bulbar vs limb onset indicator
    df['bulbar_umn'] = df.get('blb_umn', np.nan)
    df['bulbar_lmn'] = df.get('blb_lmn', np.nan)

    # Keep one row per subject
    df = df.drop_duplicates(subset=['SubjectUID'], keep='first')

    output_cols = ['SubjectUID', 'diagnosis_date', 'el_escorial',
                   'umn_burden', 'lmn_burden', 'emg_burden',
                   'bulbar_umn', 'bulbar_lmn']

    # Add individual region indicators
    for region in regions:
        for sign in signs:
            col = f'{region}_{sign}'
            if col in df.columns:
                output_cols.append(col)

    result = df[[c for c in output_cols if c in df.columns]].copy()
    print(f"  Loaded {len(result)} diagnosis records")
    return result


def load_alsfrs():
    """Load and clean ALSFRS-R longitudinal data."""
    print("Loading ALSFRS-R...")
    df = pd.read_csv(DATA_DIR / "ALSFRS-R.csv")

    # Parse visit date
    df['visit_date'] = df['Visit Date'].apply(parse_date)
    df['alsfrs_date'] = df['alsfrsdt'].apply(parse_date)

    # Use alsfrs_date if available, otherwise visit_date
    df['assessment_date'] = df['alsfrs_date'].combine_first(df['visit_date'])

    # Clean individual item scores (0-4 scale)
    item_cols = ['alsfrs1', 'alsfrs2', 'alsfrs3', 'alsfrs4', 'alsfrs5',
                 'alsfrs5a', 'alsfrs5b', 'alsfrs6', 'alsfrs7', 'alsfrs8',
                 'alsfrs9', 'alsfrsr1', 'alsfrsr2', 'alsfrsr3']

    for col in item_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
            # Clip to valid range
            df[col] = df[col].clip(0, 4)

    # Total score
    df['alsfrs_total'] = df['alsfrst'].apply(clean_numeric)

    # Compute domain scores
    df['domain_bulbar'] = df.apply(
        lambda r: compute_domain_score(r, ALSFRS_DOMAINS['bulbar']), axis=1
    )
    df['domain_fine_motor'] = df.apply(
        lambda r: compute_domain_score(r, ALSFRS_DOMAINS['fine_motor']), axis=1
    )
    df['domain_gross_motor'] = df.apply(
        lambda r: compute_domain_score(r, ALSFRS_DOMAINS['gross_motor']), axis=1
    )
    df['domain_walking'] = df.apply(
        lambda r: compute_domain_score(r, ALSFRS_DOMAINS['walking']), axis=1
    )
    df['domain_respiratory'] = df.apply(
        lambda r: compute_domain_score(r, ALSFRS_DOMAINS['respiratory']), axis=1
    )

    # Compute domain percentages
    df['pct_bulbar'] = (df['domain_bulbar'] / DOMAIN_MAX['bulbar']) * 100
    df['pct_fine_motor'] = (df['domain_fine_motor'] / DOMAIN_MAX['fine_motor']) * 100
    df['pct_gross_motor'] = (df['domain_gross_motor'] / DOMAIN_MAX['gross_motor']) * 100
    df['pct_walking'] = (df['domain_walking'] / DOMAIN_MAX['walking']) * 100
    df['pct_respiratory'] = (df['domain_respiratory'] / DOMAIN_MAX['respiratory']) * 100
    df['pct_total'] = (df['alsfrs_total'] / ALSFRS_TOTAL_MAX) * 100

    # Speech item for speech loss event
    df['speech_score'] = df['alsfrs1'].apply(clean_numeric)

    # Remove rows with no valid assessment date
    df = df[df['assessment_date'].notna()].copy()

    # Sort by patient and date
    df = df.sort_values(['SubjectUID', 'assessment_date'])

    # Select output columns
    output_cols = ['SubjectUID', 'internal_subject_id', 'Visit Name', 'assessment_date',
                   'alsfrs_total', 'domain_bulbar', 'domain_fine_motor',
                   'domain_gross_motor', 'domain_walking', 'domain_respiratory',
                   'pct_bulbar', 'pct_fine_motor', 'pct_gross_motor',
                   'pct_walking', 'pct_respiratory', 'pct_total', 'speech_score']

    # Add individual items
    for col in item_cols:
        if col in df.columns:
            output_cols.append(col)

    result = df[[c for c in output_cols if c in df.columns]].copy()
    print(f"  Loaded {len(result)} ALSFRS-R assessments for {result['SubjectUID'].nunique()} patients")
    return result


def load_assistive_devices():
    """Load and clean assistive devices log."""
    print("Loading Assistive Devices...")
    df = pd.read_csv(DATA_DIR / "Assistive Devices Log.csv")

    df['device_code'] = pd.to_numeric(df['dev'], errors='coerce')
    df['device_start_date'] = df['devstdt'].apply(parse_date)
    df['device_end_date'] = df['devenddt'].apply(parse_date)
    df['device_rec_date'] = df['devrecdt'].apply(parse_date)

    # Use recommendation date as proxy if start date missing
    df['device_start_date'] = df['device_start_date'].combine_first(df['device_rec_date'])

    # Create device type indicators
    result_rows = []

    for _, row in df.iterrows():
        code = row['device_code']
        if pd.isna(code):
            continue

        code = int(code)
        device_type = None

        for dtype, codes in DEVICE_CODES.items():
            if code in codes:
                device_type = dtype
                break

        if device_type:
            result_rows.append({
                'SubjectUID': row['SubjectUID'],
                'device_type': device_type,
                'device_start_date': row['device_start_date'],
                'device_end_date': row['device_end_date']
            })

    result = pd.DataFrame(result_rows)
    print(f"  Loaded {len(result)} device records")
    return result


def load_niv():
    """Load non-invasive ventilation data."""
    print("Loading NIV data...")
    df = pd.read_csv(DATA_DIR / "Non-Invasive Ventilation Log.csv")

    df['niv_type'] = pd.to_numeric(df['niv'], errors='coerce')
    df['niv_start_date'] = df['nivstdt'].apply(parse_date)
    df['niv_end_date'] = df['nivenddt'].apply(parse_date)
    df['niv_rec_date'] = df['nivrecdt'].apply(parse_date)

    # NIV types: 1=CPAP, 2=BiPAP, 3=ASV, 4=Tracheostomy, 99=Other
    # For our purposes, NIV includes BiPAP (2), continuous NIV would be trach (4)

    df['is_bipap'] = (df['niv_type'] == 2).astype(int)
    df['is_trach'] = (df['niv_type'] == 4).astype(int)

    # Use start date or recommendation date
    df['niv_event_date'] = df['niv_start_date'].combine_first(df['niv_rec_date'])

    # Usage (hours/day)
    df['niv_usage_low'] = pd.to_numeric(df['nivurlow'], errors='coerce')
    df['niv_usage_high'] = pd.to_numeric(df['nivurhi'], errors='coerce')

    result = df[['SubjectUID', 'niv_type', 'is_bipap', 'is_trach',
                 'niv_event_date', 'niv_start_date', 'niv_end_date',
                 'niv_usage_low', 'niv_usage_high']].copy()

    result = result[result['niv_event_date'].notna()].copy()
    print(f"  Loaded {len(result)} NIV records")
    return result


def load_feeding_tube():
    """Load feeding tube/gastrostomy data."""
    print("Loading Feeding Tube data...")
    df = pd.read_csv(DATA_DIR / "Feeding Tube Placement.csv")

    # Key date columns
    df['ftube_rec_date'] = df['ftprecdt'].apply(parse_date)  # Recommended
    df['ftube_accept_date'] = df['ftpaccdt'].apply(parse_date)  # Accepted
    df['ftube_admit_date'] = df['ftpadmdt'].apply(parse_date)  # Admission for procedure
    df['ftube_discharge_date'] = df['ftpdchdt'].apply(parse_date)  # Discharge

    # Use admission date as the event date (when tube was placed)
    df['gastrostomy_date'] = df['ftube_admit_date'].combine_first(df['ftube_accept_date'])
    df['gastrostomy_date'] = df['gastrostomy_date'].combine_first(df['ftube_rec_date'])

    # Tube type: 2=PEG
    df['ftube_type'] = pd.to_numeric(df['ftptyp'], errors='coerce')

    result = df[['SubjectUID', 'gastrostomy_date', 'ftube_type']].copy()
    result = result[result['gastrostomy_date'].notna()].copy()

    # Keep first gastrostomy per patient
    result = result.sort_values('gastrostomy_date').drop_duplicates(subset=['SubjectUID'], keep='first')

    print(f"  Loaded {len(result)} gastrostomy records")
    return result


def load_mortality():
    """Load mortality data."""
    print("Loading Mortality data...")
    df = pd.read_csv(DATA_DIR / "Mortality.csv")

    df['death_date'] = df['dieddt'].apply(parse_date)
    df['death_cause'] = df['diedcaus']

    result = df[['SubjectUID', 'death_date', 'death_cause']].copy()
    result = result[result['death_date'].notna()].copy()

    # Keep one death record per patient
    result = result.drop_duplicates(subset=['SubjectUID'], keep='first')

    print(f"  Loaded {len(result)} mortality records")
    return result


# ============================================================================
# Dataset Construction
# ============================================================================

def create_patient_static_features(demographics, diagnosis):
    """Create patient-level static features."""
    print("\nCreating patient static features...")

    # Merge demographics and diagnosis
    patients = demographics.merge(diagnosis, on='SubjectUID', how='left')

    # Compute age at diagnosis if possible
    patients['age_at_diagnosis'] = np.nan
    mask = patients['diagnosis_date'].notna() & patients['dob'].notna()
    patients.loc[mask, 'age_at_diagnosis'] = (
        (patients.loc[mask, 'diagnosis_date'] - patients.loc[mask, 'dob']).dt.days / 365.25
    )

    # Use recorded age if age_at_diagnosis not available
    patients['age_at_diagnosis'] = patients['age_at_diagnosis'].fillna(patients['age'])

    print(f"  Created static features for {len(patients)} patients")
    return patients


def create_event_outcomes(patients, alsfrs, devices, niv, gastrostomy, mortality):
    """Create event outcome indicators."""
    print("\nCreating event outcomes...")

    # Initialize outcome columns
    events = patients[['SubjectUID', 'diagnosis_date']].copy()

    # 1. Death
    events = events.merge(
        mortality[['SubjectUID', 'death_date']],
        on='SubjectUID', how='left'
    )
    events['event_death'] = events['death_date'].notna().astype(int)

    # 2. Gastrostomy
    events = events.merge(
        gastrostomy[['SubjectUID', 'gastrostomy_date']],
        on='SubjectUID', how='left'
    )
    events['event_gastrostomy'] = events['gastrostomy_date'].notna().astype(int)

    # 3. NIV (first BiPAP)
    niv_bipap = niv[niv['is_bipap'] == 1].copy()
    niv_first = niv_bipap.sort_values('niv_event_date').drop_duplicates(
        subset=['SubjectUID'], keep='first'
    )[['SubjectUID', 'niv_event_date']].rename(columns={'niv_event_date': 'niv_date'})

    events = events.merge(niv_first, on='SubjectUID', how='left')
    events['event_niv'] = events['niv_date'].notna().astype(int)

    # 4. Continuous NIV (tracheostomy)
    niv_trach = niv[niv['is_trach'] == 1].copy()
    trach_first = niv_trach.sort_values('niv_event_date').drop_duplicates(
        subset=['SubjectUID'], keep='first'
    )[['SubjectUID', 'niv_event_date']].rename(columns={'niv_event_date': 'trach_date'})

    events = events.merge(trach_first, on='SubjectUID', how='left')
    events['event_continuous_niv'] = events['trach_date'].notna().astype(int)

    # 5. Wheelchair use (manual or power)
    wheelchair_devices = devices[devices['device_type'].isin(['manual_wheelchair', 'power_wheelchair'])].copy()
    wheelchair_first = wheelchair_devices.sort_values('device_start_date').drop_duplicates(
        subset=['SubjectUID'], keep='first'
    )[['SubjectUID', 'device_start_date']].rename(columns={'device_start_date': 'wheelchair_date'})

    events = events.merge(wheelchair_first, on='SubjectUID', how='left')
    events['event_wheelchair'] = events['wheelchair_date'].notna().astype(int)

    # 6. Speech loss (ALSFRS-R speech score <= 1)
    speech_loss = alsfrs[alsfrs['speech_score'] <= 1].copy()
    speech_first = speech_loss.sort_values('assessment_date').drop_duplicates(
        subset=['SubjectUID'], keep='first'
    )[['SubjectUID', 'assessment_date']].rename(columns={'assessment_date': 'speech_loss_date'})

    events = events.merge(speech_first, on='SubjectUID', how='left')
    events['event_speech_loss'] = events['speech_loss_date'].notna().astype(int)

    # Compute time from diagnosis to events
    for event_col, date_col in [
        ('event_death', 'death_date'),
        ('event_gastrostomy', 'gastrostomy_date'),
        ('event_niv', 'niv_date'),
        ('event_continuous_niv', 'trach_date'),
        ('event_wheelchair', 'wheelchair_date'),
        ('event_speech_loss', 'speech_loss_date')
    ]:
        time_col = f'time_to_{event_col.replace("event_", "")}'
        mask = events[date_col].notna() & events['diagnosis_date'].notna()
        events[time_col] = np.nan
        events.loc[mask, time_col] = (
            events.loc[mask, date_col] - events.loc[mask, 'diagnosis_date']
        ).dt.days

    print(f"  Event counts:")
    for col in ['event_death', 'event_gastrostomy', 'event_niv',
                'event_continuous_niv', 'event_wheelchair', 'event_speech_loss']:
        print(f"    {col}: {events[col].sum()}")

    return events


def create_longitudinal_dataset(patients, alsfrs, events):
    """
    Create visit-level landmark dataset for training.
    Each row is a visit that can serve as a landmark for prediction.
    """
    print("\nCreating longitudinal landmark dataset...")

    # Merge patient info with ALSFRS visits
    visits = alsfrs.merge(
        patients[['SubjectUID', 'diagnosis_date', 'age_at_diagnosis', 'is_female',
                  'is_hispanic', 'race_white', 'race_black', 'race_asian', 'race_other',
                  'el_escorial', 'umn_burden', 'lmn_burden', 'emg_burden']],
        on='SubjectUID', how='left'
    )

    # Compute time from diagnosis
    mask = visits['assessment_date'].notna() & visits['diagnosis_date'].notna()
    visits['months_since_diagnosis'] = np.nan
    visits.loc[mask, 'months_since_diagnosis'] = (
        (visits.loc[mask, 'assessment_date'] - visits.loc[mask, 'diagnosis_date']).dt.days / 30.44
    )

    # Sort and compute visit number
    visits = visits.sort_values(['SubjectUID', 'assessment_date'])
    visits['visit_num'] = visits.groupby('SubjectUID').cumcount() + 1

    # Compute time since previous visit
    visits['prev_date'] = visits.groupby('SubjectUID')['assessment_date'].shift(1)
    visits['days_since_prev_visit'] = (
        visits['assessment_date'] - visits['prev_date']
    ).dt.days

    # Compute ALSFRS-R change from previous visit
    for col in ['alsfrs_total', 'domain_bulbar', 'domain_fine_motor',
                'domain_gross_motor', 'domain_walking', 'domain_respiratory']:
        visits[f'{col}_prev'] = visits.groupby('SubjectUID')[col].shift(1)
        visits[f'{col}_change'] = visits[col] - visits[f'{col}_prev']
        visits[f'{col}_slope'] = visits[f'{col}_change'] / visits['days_since_prev_visit'] * 30  # per month

    # Merge event dates for computing time-to-event from each landmark
    visits = visits.merge(
        events[['SubjectUID', 'death_date', 'gastrostomy_date', 'niv_date',
                'trach_date', 'wheelchair_date', 'speech_loss_date']],
        on='SubjectUID', how='left'
    )

    # For each event, compute time from current visit to event
    event_dates = {
        'death': 'death_date',
        'gastrostomy': 'gastrostomy_date',
        'niv': 'niv_date',
        'continuous_niv': 'trach_date',
        'wheelchair': 'wheelchair_date',
        'speech_loss': 'speech_loss_date'
    }

    for event_name, date_col in event_dates.items():
        # Time from this visit to event
        visits[f'days_to_{event_name}'] = np.nan
        mask = visits[date_col].notna() & visits['assessment_date'].notna()
        visits.loc[mask, f'days_to_{event_name}'] = (
            visits.loc[mask, date_col] - visits.loc[mask, 'assessment_date']
        ).dt.days

        # Event indicator (only count future events)
        visits[f'future_{event_name}'] = (visits[f'days_to_{event_name}'] > 0).astype(int)

        # Set time to NaN for events that already happened
        visits.loc[visits[f'days_to_{event_name}'] <= 0, f'days_to_{event_name}'] = np.nan

    # Remove visits before diagnosis (invalid landmarks)
    visits = visits[visits['months_since_diagnosis'] >= 0].copy()

    print(f"  Created {len(visits)} landmark visits for {visits['SubjectUID'].nunique()} patients")

    return visits


def create_threshold_crossing_events(visits):
    """
    Create threshold crossing events for domain percentages.
    Thresholds: 80%, 70%, 60%, 50%
    """
    print("\nComputing threshold crossing events...")

    thresholds = [80, 70, 60, 50]
    domains = ['bulbar', 'fine_motor', 'gross_motor', 'walking', 'respiratory', 'total']

    # For each domain and threshold, find first crossing date
    crossing_records = []

    for subject_id in visits['SubjectUID'].unique():
        patient_visits = visits[visits['SubjectUID'] == subject_id].sort_values('assessment_date')

        for domain in domains:
            pct_col = f'pct_{domain}'
            if pct_col not in patient_visits.columns:
                continue

            for threshold in thresholds:
                # Find first visit where percentage drops below threshold
                below = patient_visits[patient_visits[pct_col] < threshold]

                if len(below) > 0:
                    first_crossing = below.iloc[0]
                    crossing_records.append({
                        'SubjectUID': subject_id,
                        'domain': domain,
                        'threshold': threshold,
                        'crossing_date': first_crossing['assessment_date'],
                        'value_at_crossing': first_crossing[pct_col]
                    })

    crossings = pd.DataFrame(crossing_records)
    print(f"  Found {len(crossings)} threshold crossing events")

    # Pivot to wide format
    if len(crossings) > 0:
        crossings_wide = crossings.pivot_table(
            index='SubjectUID',
            columns=['domain', 'threshold'],
            values='crossing_date',
            aggfunc='first'
        )
        crossings_wide.columns = [f'cross_{d}_{t}' for d, t in crossings_wide.columns]
        crossings_wide = crossings_wide.reset_index()
    else:
        crossings_wide = pd.DataFrame({'SubjectUID': visits['SubjectUID'].unique()})

    return crossings, crossings_wide


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("=" * 60)
    print("ALS Digital Twin Data Cleaning Pipeline")
    print("=" * 60)

    # Load raw data
    demographics = load_demographics()
    diagnosis = load_diagnosis()
    alsfrs = load_alsfrs()
    devices = load_assistive_devices()
    niv = load_niv()
    gastrostomy = load_feeding_tube()
    mortality = load_mortality()

    # Create patient static features
    patients = create_patient_static_features(demographics, diagnosis)

    # Create event outcomes
    events = create_event_outcomes(
        patients, alsfrs, devices, niv, gastrostomy, mortality
    )

    # Create longitudinal landmark dataset
    visits = create_longitudinal_dataset(patients, alsfrs, events)

    # Create threshold crossing events
    crossings_long, crossings_wide = create_threshold_crossing_events(visits)

    # Merge threshold crossings into patient events
    events_full = events.merge(crossings_wide, on='SubjectUID', how='left')

    # ========================================================================
    # Save cleaned datasets
    # ========================================================================
    print("\n" + "=" * 60)
    print("Saving cleaned datasets...")

    # 1. Patient static features
    patients.to_csv(OUTPUT_DIR / "patients_static.csv", index=False)
    print(f"  Saved patients_static.csv: {len(patients)} patients")

    # 2. Event outcomes
    events_full.to_csv(OUTPUT_DIR / "patient_events.csv", index=False)
    print(f"  Saved patient_events.csv: {len(events_full)} patients")

    # 3. Longitudinal landmark dataset
    visits.to_csv(OUTPUT_DIR / "visits_longitudinal.csv", index=False)
    print(f"  Saved visits_longitudinal.csv: {len(visits)} visits")

    # 4. Threshold crossings (long format)
    crossings_long.to_csv(OUTPUT_DIR / "threshold_crossings.csv", index=False)
    print(f"  Saved threshold_crossings.csv: {len(crossings_long)} crossings")

    # 5. Create merged training dataset
    # Merge static features with visits for a complete landmark dataset
    training_data = visits.merge(
        events_full.drop(columns=['diagnosis_date'], errors='ignore'),
        on='SubjectUID', how='left'
    )
    training_data.to_csv(OUTPUT_DIR / "training_landmarks.csv", index=False)
    print(f"  Saved training_landmarks.csv: {len(training_data)} landmark instances")

    # ========================================================================
    # Summary statistics
    # ========================================================================
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print(f"\nPatients: {patients['SubjectUID'].nunique()}")
    print(f"Total ALSFRS-R assessments: {len(alsfrs)}")
    print(f"Training landmark instances: {len(training_data)}")

    print(f"\nVisits per patient:")
    visit_counts = visits.groupby('SubjectUID').size()
    print(f"  Mean: {visit_counts.mean():.1f}")
    print(f"  Median: {visit_counts.median():.0f}")
    print(f"  Min: {visit_counts.min()}")
    print(f"  Max: {visit_counts.max()}")

    print(f"\nEvent rates:")
    for col in ['event_death', 'event_gastrostomy', 'event_niv',
                'event_continuous_niv', 'event_wheelchair', 'event_speech_loss']:
        rate = events_full[col].mean() * 100
        count = events_full[col].sum()
        print(f"  {col}: {count} ({rate:.1f}%)")

    print(f"\nMissing data in training set:")
    key_cols = ['alsfrs_total', 'months_since_diagnosis', 'age_at_diagnosis']
    for col in key_cols:
        missing = training_data[col].isna().mean() * 100
        print(f"  {col}: {missing:.1f}%")

    print("\n" + "=" * 60)
    print("Data cleaning complete!")
    print(f"Output saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
