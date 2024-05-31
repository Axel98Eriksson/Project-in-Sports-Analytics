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

file_path = 'Results/results_friendly.csv'
data = pd.read_csv(file_path)

for index, row in data.iterrows():
    #print(f"Predicting match {index + 1}...")
    home_team = row['home_team']
    away_team = row['away_team']
    tournament = 'UEFA Euro'
    home_ranking = row['home_ranking']
    away_ranking = row['away_ranking']
    neutral = row['neutral']

    home_goals, away_goals = predict_goals(home_team, away_team, tournament, home_ranking, away_ranking, neutral)

    home_goals_rounded = round(home_goals)
    away_goals_rounded = round(away_goals)

    #add the predicted goals to a list
    data.at[index, 'home_goals'] = home_goals_rounded
    data.at[index, 'away_goals'] = away_goals_rounded
    #add acutal goals to a list
    data.at[index, 'actual_home_goals'] = row['home_score']
    data.at[index, 'actual_away_goals'] = row['away_score']

    #add the difference between the predicted and actual goals to a list
    data.at[index, 'home_goals_diff'] = home_goals_rounded - row['home_score']
    data.at[index, 'away_goals_diff'] = away_goals_rounded - row['away_score']

    #add the absolute difference between the predicted and actual goals to a list
    data.at[index, 'home_goals_diff_abs'] = abs(home_goals_rounded - row['home_score'])
    data.at[index, 'away_goals_diff_abs'] = abs(away_goals_rounded - row['away_score'])

    #print the predicted goals and the actual goals
    #print(f"Predicted Home Goals: {home_goals_rounded} ({row['home_score']}), Predicted Away Goals: {away_goals_rounded} ({row['away_score']})")


#calculate the total number of correct predictions
home_goals_correct = data[data['home_goals'] == data['actual_home_goals']]
away_goals_correct = data[data['away_goals'] == data['actual_away_goals']]
total_correct = len(home_goals_correct) + len(away_goals_correct)
print(f"Total Correct Home Goals Predictions: {len(home_goals_correct)}")
print(f"Total Correct Away Goals Predictions: {len(away_goals_correct)}")
print(f"Total Correct Predictions: {total_correct}")

#calculate the total number of correct winner predictions
correct_winner = data[
    ((data['home_goals'] > data['away_goals']) & (data['actual_home_goals'] > data['actual_away_goals'])) |
    ((data['home_goals'] < data['away_goals']) & (data['actual_home_goals'] < data['actual_away_goals'])) |
    ((data['home_goals'] == data['away_goals']) & (data['actual_home_goals'] == data['actual_away_goals']))
]

print(f"Total Correct Winner Predictions: {len(correct_winner)}")

#calculate the accuracy of the model
accuracy = len(correct_winner) / len(data)
print(f"Model Accuracy: {accuracy:.2f}")
