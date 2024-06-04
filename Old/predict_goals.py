import pandas as pd
import joblib

# Load the trained models and label encoders
model_home = joblib.load('models/model_home.joblib')
model_away = joblib.load('models/model_away.joblib')
label_encoders = joblib.load('models/label_encoders.joblib')

# Function to preprocess input data for prediction
def preprocess_input(home_team, away_team, tournament, home_ranking, away_ranking, neutral):
    # Encode input features using the loaded label encoders
    home_team_encoded = label_encoders['home_team'].transform([home_team])[0]
    away_team_encoded = label_encoders['away_team'].transform([away_team])[0]
    tournament_encoded = label_encoders['tournament'].transform([tournament])[0]
    
    if neutral == 1:
        team1 = min(home_team_encoded, away_team_encoded)
        team2 = max(home_team_encoded, away_team_encoded)
        ranking_diff = abs(home_ranking - away_ranking)
    else:
        team1 = home_team_encoded
        team2 = away_team_encoded
        ranking_diff = home_ranking - away_ranking
    
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'team1': [team1],
        'team2': [team2],
        'tournament': [tournament_encoded],
        'ranking_diff': [ranking_diff],
        'neutral': [int(neutral)]
    })
    
    return input_data

# Function to predict goals scored by each team
def predict_goals(home_team, away_team, tournament, home_ranking, away_ranking, neutral):
    # Preprocess input data
    input_data = preprocess_input(home_team, away_team, tournament, home_ranking, away_ranking, neutral)
    
    # Predict goals for home and away teams
    home_goals = model_home.predict(input_data)[0]
    away_goals = model_away.predict(input_data)[0]
    
    # If the match is neutral, assign goals back to the respective teams correctly
    if neutral == 1:
        if home_team > away_team:
            home_goals, away_goals = away_goals, home_goals
    
    return home_goals, away_goals

# Example usage
#home_team = "Slovakia"
home_team = "France"
#away_team = "Norway"
away_team = "England"
tournament = "UEFA Euro"
home_ranking = 2
away_ranking = 4
neutral = 1

home_goals, away_goals = predict_goals(home_team, away_team, tournament, home_ranking, away_ranking, neutral)

home_goals_rounded = round(home_goals)
away_goals_rounded = round(away_goals)

difference = abs(home_goals - away_goals)

print(f"Predicted Home Goals: {home_goals} ({home_goals_rounded}), Predicted Away Goals: {away_goals} ({away_goals_rounded})")

#print the winner
if difference < 0.3:
    print("Predicted Draw")
elif home_goals_rounded > away_goals_rounded:
    print(f"Predicted Winner: {home_team}")
elif away_goals_rounded > home_goals_rounded:
    print(f"Predicted Winner: {away_team}")
