import pandas as pd
import joblib

# Load the trained models and label encoders
model_home = joblib.load('models/model_home.joblib')
model_away = joblib.load('models/model_away.joblib')
label_encoders = joblib.load('models/label_encoders1.joblib')

# Function to preprocess input data for prediction
def preprocess_input(home_team, away_team, tournament, home_ranking, away_ranking, neutral):
    # Encode input features using the loaded label encoders
    home_team_encoded = label_encoders['home_team'].transform([home_team])[0]
    away_team_encoded = label_encoders['away_team'].transform([away_team])[0]
    tournament_encoded = label_encoders['tournament'].transform([tournament])[0]
    
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'home_team': [home_team_encoded],
        'away_team': [away_team_encoded],
        'tournament': [tournament_encoded],
        'home_ranking': [int(home_ranking)],
        'away_ranking': [int(away_ranking)],
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
    
    return home_goals, away_goals

# Example usage
home_team = "Spain"
away_team = "Portugal"
tournament = "UEFA Euro"
home_ranking = 8
away_ranking = 6
neutral = 0

home_goals, away_goals = predict_goals(home_team, away_team, tournament, home_ranking, away_ranking, neutral)

print(f"Predicted Home Goals: {home_goals}, Predicted Away Goals: {away_goals}")

#print the winner
if home_goals > away_goals:
    print(f"Predicted Winner: {home_team}")
elif away_goals > home_goals:
    print(f"Predicted Winner: {away_team}")
else:    
    print("Predicted Draw")
