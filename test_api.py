# test_api.py - Test your FastAPI endpoint

import requests
import json

# API URL (local)
API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_URL}/health")
    print(f"Health: {response.json()}")

def test_teams():
    """Test teams endpoint"""
    response = requests.get(f"{API_URL}/teams")
    data = response.json()
    print(f"Teams loaded: {data['total']}")
    print(f"Top 5 teams: {data['teams'][:5]}")

def test_predict():
    """Test single prediction endpoint"""
    payload = {
        "home_team": "Brazil",
        "away_team": "France",
        "round_name": "Final",
        "is_host": False,
        "year": 2026
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    result = response.json()
    
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Match: {result['home_team']} vs {result['away_team']}")
    print(f"Round: {result['round_name']}")
    print(f"Winner: {result['winner']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Home win probability: {result['home_win_probability']:.1%}")
    print(f"Away win probability: {result['away_win_probability']:.1%}")
    print(f"Elo ratings: {result['home_elo']} vs {result['away_elo']}")

def test_batch():
    """Test batch prediction endpoint"""
    payload = {
        "matches": [
            {"home_team": "Brazil", "away_team": "Argentina", "round_name": "Final"},
            {"home_team": "Germany", "away_team": "Spain", "round_name": "Semi-final"},
            {"home_team": "USA", "away_team": "Mexico", "round_name": "Round of 16", "is_host": True}
        ]
    }
    
    response = requests.post(f"{API_URL}/predict-batch", json=payload)
    results = response.json()
    
    print("\n" + "="*50)
    print("BATCH PREDICTIONS")
    print("="*50)
    for pred in results['predictions']:
        print(f"\n{pred['home_team']} vs {pred['away_team']}: {pred['winner']} wins ({pred['confidence']:.1%})")

if __name__ == "__main__":
    print("Testing World Cup Predictor API...")
    print("Make sure the API is running (python app.py or uvicorn app:app --reload)")
    print()
    
    test_health()
    print()
    test_teams()
    print()
    test_predict()
    print()
    test_batch()