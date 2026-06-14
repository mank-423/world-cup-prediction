import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

print("="*50)
print("TRAINING MODEL LOCALLY")
print("="*50)

# Load your data files
# Make sure these CSV files are in the same directory
matches_df = pd.read_csv('data/WorldCupMatches.csv')
elo_df = pd.read_csv('data/elo_ratings.csv')

print(f"✅ Loaded matches: {matches_df.shape}")
print(f"✅ Loaded Elo ratings: {elo_df.shape}")

# Clean match data
def clean_match_data(df):
    data = df.copy()
    
    for col in ['home_score', 'away_score', 'Year']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data['home_win'] = np.nan
    valid = data['home_score'].notna() & data['away_score'].notna()
    data.loc[valid & (data['home_score'] > data['away_score']), 'home_win'] = 1
    data.loc[valid & (data['away_score'] > data['home_score']), 'home_win'] = 0
    data = data[data['home_win'].notna()].copy()
    
    if 'Round' in data.columns:
        round_weights = {'Group': 1, 'Round of 16': 2, 'Quarter-final': 3, 'Semi-final': 4, 'Final': 5}
        data['round_code'] = data['Round'].map(round_weights).fillna(1)
    else:
        data['round_code'] = 1
    
    return data

matches_clean = clean_match_data(matches_df)
print(f"✅ Cleaned matches: {len(matches_clean)}")

# Prepare Elo ratings
def prepare_elo_data(elo_df):
    min_rating = elo_df['current_rating'].min()
    max_rating = elo_df['current_rating'].max()
    
    team_elo = {}
    for _, row in elo_df.iterrows():
        team_name = row['team']
        rating = row['current_rating']
        normalized_strength = (rating - min_rating) / (max_rating - min_rating)
        
        wins = row.get('wins', 0)
        total_matches = row.get('total_matches', 1)
        win_rate = wins / max(1, total_matches)
        
        team_elo[team_name] = {
            'elo_rating': float(rating),
            'strength': float(normalized_strength),
            'win_rate': float(win_rate),
            'recent_form': float(win_rate),
            'rank': int(row.get('rank', 99))
        }
    
    return team_elo

team_strengths = prepare_elo_data(elo_df)
print(f"✅ Loaded Elo for {len(team_strengths)} teams")

# Merge data
def merge_with_elo(matches_df, team_elo):
    data = matches_df.copy()
    
    data['home_elo'] = data['home_team'].map(lambda x: team_elo.get(x, {}).get('elo_rating', 1500))
    data['away_elo'] = data['away_team'].map(lambda x: team_elo.get(x, {}).get('elo_rating', 1500))
    data['home_strength'] = data['home_team'].map(lambda x: team_elo.get(x, {}).get('strength', 0.5))
    data['away_strength'] = data['away_team'].map(lambda x: team_elo.get(x, {}).get('strength', 0.5))
    data['home_win_rate'] = data['home_team'].map(lambda x: team_elo.get(x, {}).get('win_rate', 0.5))
    data['away_win_rate'] = data['away_team'].map(lambda x: team_elo.get(x, {}).get('win_rate', 0.5))
    
    data['elo_diff'] = data['home_elo'] - data['away_elo']
    data['strength_diff'] = data['home_strength'] - data['away_strength']
    data['win_rate_diff'] = data['home_win_rate'] - data['away_win_rate']
    
    return data.dropna(subset=['home_win'])

merged_data = merge_with_elo(matches_clean, team_strengths)
print(f"✅ Merged data: {len(merged_data)} matches")

# Train model
feature_cols = ['elo_diff', 'strength_diff', 'win_rate_diff', 'round_code']
X = merged_data[feature_cols].fillna(0)
y = merged_data['home_win']

# Split
if 'Year' in merged_data.columns:
    train_mask = merged_data['Year'] <= 2014
    test_mask = merged_data['Year'] > 2014
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
else:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=20, random_state=42)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"\n📈 Model Performance:")
print(f"   Training accuracy: {train_acc:.3f}")
print(f"   Test accuracy: {test_acc:.3f}")

# Save model and data
joblib.dump(model, 'model/worldcup_model.pkl')

# Save team strengths as JSON
team_strengths_serializable = {}
for team, data in team_strengths.items():
    team_strengths_serializable[team] = {
        'elo_rating': data['elo_rating'],
        'strength': data['strength'],
        'win_rate': data['win_rate'],
        'recent_form': data['recent_form'],
        'rank': data['rank']
    }

with open('model/team_strengths.json', 'w') as f:
    json.dump(team_strengths_serializable, f)

with open('model/feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)

print("\n✅ Model and data saved to 'model/' folder!")