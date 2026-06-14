# app.py - Simplified World Cup Match Predictor API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
from typing import Optional, Dict, Any
import os

app = FastAPI(title="World Cup Match Predictor API")

# Load model and data
print("Loading model and data...")

MODEL_PATH = "model/worldcup_model.pkl"
TEAM_STRENGTHS_PATH = "model/team_strengths.json"
FEATURE_COLS_PATH = "model/feature_cols.json"

# Load model
try:
    with open(MODEL_PATH, 'rb') as f:
        model = joblib.load(f)
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Model error: {e}")
    model = None

# Load team strengths
try:
    with open(TEAM_STRENGTHS_PATH, 'r') as f:
        team_strengths = json.load(f)
    print(f"✅ Loaded {len(team_strengths)} teams")
except Exception as e:
    print(f"❌ Team error: {e}")
    team_strengths = {}

# Load feature columns
try:
    with open(FEATURE_COLS_PATH, 'r') as f:
        feature_cols = json.load(f)
    print(f"✅ Features: {feature_cols}")
except Exception as e:
    print(f"❌ Features error: {e}")
    feature_cols = ['elo_diff', 'strength_diff', 'win_rate_diff', 'round_code']

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    round_name: str = "Group"
    is_host: bool = False

class MatchResponse(BaseModel):
    home_team: str
    away_team: str
    winner: str
    home_win_probability: float
    away_win_probability: float
    confidence: float
    home_elo: int
    away_elo: int

@app.get("/")
async def root():
    return {"message": "World Cup Predictor API", "teams_loaded": len(team_strengths)}

@app.get("/health")
async def health():
    return {"status": "ok", "teams": len(team_strengths), "model": model is not None}

@app.get("/teams")
async def list_teams():
    teams = sorted(team_strengths.keys())
    return {"teams": teams[:50], "total": len(teams)}  # Show first 50

@app.post("/predict", response_model=MatchResponse)
async def predict(request: MatchRequest):
    # Check teams exist
    if request.home_team not in team_strengths:
        raise HTTPException(404, f"Home team '{request.home_team}' not found")
    if request.away_team not in team_strengths:
        raise HTTPException(404, f"Away team '{request.away_team}' not found")
    
    home = team_strengths[request.home_team]
    away = team_strengths[request.away_team]
    
    # Round weights
    round_weights = {'Group': 1, 'Round of 32': 2, 'Round of 16': 2,
                     'Quarter-final': 3, 'Semi-final': 4, 'Final': 5}
    round_code = round_weights.get(request.round_name, 1)
    
    # Features
    features = pd.DataFrame([{
        'elo_diff': home['elo_rating'] - away['elo_rating'],
        'strength_diff': home['strength'] - away['strength'],
        'win_rate_diff': home['win_rate'] - away['win_rate'],
        'round_code': round_code
    }])
    
    # Predict
    prob = model.predict_proba(features)[0, 1]
    
    # Host advantage
    if request.is_host:
        prob = min(0.85, prob * 1.08)
    
    winner = request.home_team if prob > 0.5 else request.away_team
    
    return MatchResponse(
        home_team=request.home_team,
        away_team=request.away_team,
        winner=winner,
        home_win_probability=round(prob, 4),
        away_win_probability=round(1 - prob, 4),
        confidence=round(max(prob, 1 - prob), 4),
        home_elo=int(home['elo_rating']),
        away_elo=int(away['elo_rating'])
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)