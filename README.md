# World Cup Match Predictor API
<img width="856" height="412" alt="image" src="https://github.com/user-attachments/assets/5af1a51a-924e-49bf-8db6-542755b4ea23" />


A machine learning API that predicts FIFA World Cup match outcomes using Elo ratings and historical data.

## Live Demo

API endpoint: `http://13.221.128.89:8000`

## Features

- Predict match winners between any two teams
- Returns win probabilities and confidence scores
- Supports different tournament stages (Group, Round of 16, Quarter-final, Semi-final, Final)
- Host advantage adjustment
- Docker containerized
- AWS EC2 deployment

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/teams` | List all available teams |
| POST | `/predict` | Predict match winner |
| GET | `/docs` | Interactive Swagger documentation |

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Brazil",
    "away_team": "France",
    "round_name": "Final",
    "is_host": false
  }'
```

## Example Response
```bash
{
  "home_team": "Brazil",
  "away_team": "France",
  "winner": "Brazil",
  "home_win_probability": 0.5846,
  "away_win_probability": 0.4154,
  "confidence": 0.5846,
  "home_elo": 1991,
  "away_elo": 2063
}
```

## Tech Stack:
1. Python 3.11
2. FastAPI - Web framework
3. scikit-learn - Machine learning
4. pandas/numpy - Data processing
5. Docker - Containerization
6. AWS EC2 - Cloud deployment

## Model Training
The model is trained on:
- Historical World Cup matches (1930-2022)
- Elo ratings from World Football Elo Ratings

## Features used
Elo difference (team strength gap)
Strength difference (normalized Elo)
Win rate difference (historical performance)
Round code (tournament stage)

## Training accuracy
Test accuracy: 58%
Most important feature: Win rate difference (31.6%)

