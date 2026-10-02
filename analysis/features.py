"""Row-local feature candidates: no labels, IDs, or fitted target statistics."""
import numpy as np
import pandas as pd

RATINGS = [
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Cleanliness',
]
CATEGORIES = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
PAIRS = [
    ('Inflight wifi service', 'Ease of Online booking'),
    ('Ease of Online booking', 'Online boarding'),
    ('Seat comfort', 'Inflight entertainment'),
    ('Food and drink', 'Cleanliness'),
    ('Seat comfort', 'Cleanliness'),
    ('On-board service', 'Baggage handling'),
]

def feature_groups(df):
    """Return independent candidate families to test against the raw baseline."""
    r = df[RATINGS]
    positive = r.where(r > 0)  # Zero semantics are a hypothesis, not assumed truth.
    zero = pd.DataFrame(index=df.index)
    for i, c in enumerate(RATINGS):
        zero[f'zero_{i}'] = (df[c] == 0).astype('int8')
    zero['zero_count'] = (r == 0).sum(axis=1)
    aggregates = pd.DataFrame({
        'rating_mean': r.mean(axis=1), 'rating_std': r.std(axis=1),
        'rating_min': r.min(axis=1), 'rating_max': r.max(axis=1),
        'rating_range': r.max(axis=1) - r.min(axis=1),
        'rating_low_count': ((r >= 1) & (r <= 2)).sum(axis=1),
        'rating_high_count': (r >= 4).sum(axis=1),
        'rating_positive_mean': positive.mean(axis=1),
    }, index=df.index)
    for name, cols in {
        'digital': ['Inflight wifi service', 'Ease of Online booking', 'Online boarding'],
        'comfort': ['Seat comfort', 'Inflight entertainment', 'Food and drink', 'Cleanliness'],
        'staff': ['On-board service', 'Baggage handling', 'Checkin service'],
    }.items():
        aggregates[f'{name}_mean'] = df[cols].mean(axis=1)
        aggregates[f'{name}_range'] = df[cols].max(axis=1) - df[cols].min(axis=1)
    disagreement = pd.DataFrame(index=df.index)
    for i, (a, b) in enumerate(PAIRS):
        disagreement[f'pair_{i}_equal'] = (df[a] == df[b]).astype('int8')
        disagreement[f'pair_{i}_diff'] = df[a] - df[b]
        # Numeric difference + categorical combination capture different views.
        disagreement[f'pair_{i}_category'] = df[a].astype(str) + ':' + df[b].astype(str)
    context = pd.DataFrame(index=df.index)
    context['customer_travel_class'] = df['Customer Type'] + '|' + df['Type of Travel'] + '|' + df['Class']
    for c in ['Inflight wifi service', 'Online boarding', 'Inflight entertainment']:
        context[f'{c}_by_travel'] = df[c].astype(str) + '|' + df['Type of Travel']
        context[f'{c}_by_class'] = df[c].astype(str) + '|' + df['Class']
    dep, arr = df['Departure Delay in Minutes'], df['Arrival Delay in Minutes']
    delay = pd.DataFrame({
        'arrival_missing': arr.isna().astype('int8'),
        'arrival_minus_departure': arr - dep,
        'delay_recovered': (dep - arr).clip(lower=0),
        'delay_added': (arr - dep).clip(lower=0),
        'departure_zero': (dep == 0).astype('int8'),
        'arrival_zero': (arr == 0).astype('int8'),
        'departure_log': np.log1p(dep), 'arrival_log': np.log1p(arr),
    }, index=df.index)
    return {'zero': zero, 'aggregates': aggregates, 'disagreement': disagreement,
            'context': context, 'delay': delay}

def model_frame(df, families=()):
    raw = df.drop(columns=['id', 'satisfaction'], errors='ignore').copy()
    groups = feature_groups(df) if families else {}
    result = pd.concat([raw] + [groups[f] for f in families], axis=1)
    cats = result.select_dtypes(include=['object', 'str', 'category']).columns.tolist()
    result[cats] = result[cats].fillna('__MISSING__').astype(str)
    nums = result.columns.difference(cats)
    result[nums] = result[nums].astype('float32').fillna(-999)
    return result, cats
