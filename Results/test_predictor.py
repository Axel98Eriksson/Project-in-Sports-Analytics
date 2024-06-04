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
    #tournament should always be UEFA Euro
    tournament_encoded = label_encoders['tournament'].transform(['UEFA Euro'])[0]
    
    # Create separate DataFrames for home and away predictions
    home_input_data = pd.DataFrame({
        'team1': [home_team_encoded],
        'team2': [away_team_encoded],
        'tournament': [tournament_encoded],
        'ranking_diff': [home_ranking - away_ranking],
        'neutral': [1]
    })
    
    away_input_data = pd.DataFrame({
        'team1': [away_team_encoded],
        'team2': [home_team_encoded],
        'tournament': [tournament_encoded],
        'ranking_diff': [away_ranking - home_ranking],
        'neutral': [1]
    })
    
    return home_input_data, away_input_data

# Function to predict goals scored by each team
def predict_goals(home_team, away_team, tournament, home_ranking, away_ranking, neutral):
    # Preprocess input data
    home_input_data, away_input_data = preprocess_input(home_team, away_team, tournament, home_ranking, away_ranking, neutral)
    
    # Predict goals for home and away teams using their respective models
    home_goals = model_home.predict(home_input_data)[0]
    away_goals = model_away.predict(away_input_data)[0]
    
    return home_goals, away_goals

file_path = 'Results/results_with_rankings_wo_conference.csv'
data = pd.read_csv(file_path)

for index, row in data.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    tournament = 'UEFA Euro'
    home_ranking = row['home_ranking']
    away_ranking = row['away_ranking']
    neutral = row['neutral']

    home_goals, away_goals = predict_goals(home_team, away_team, tournament, home_ranking, away_ranking, neutral)

    home_goals_rounded = round(home_goals)
    away_goals_rounded = round(away_goals)

    data.at[index, 'home_goals'] = home_goals_rounded
    data.at[index, 'away_goals'] = away_goals_rounded
    data.at[index, 'actual_home_goals'] = row['home_score']
    data.at[index, 'actual_away_goals'] = row['away_score']

    
    if abs(home_goals - away_goals) < 0.4:          #if difference < 0.4 
        data.at[index, 'predicted_winner'] = 'Draw'
    elif home_goals_rounded > away_goals_rounded:
        data.at[index, 'predicted_winner'] = home_team
    elif home_goals_rounded < away_goals_rounded:
        data.at[index, 'predicted_winner'] = away_team
    else:
        data.at[index, 'predicted_winner'] = 'Draw'


    if row['home_score'] > row['away_score']:
        data.at[index, 'winner'] = home_team
    elif row['home_score'] < row['away_score']:
        data.at[index, 'winner'] = away_team
    else:
        data.at[index, 'winner'] = 'Draw'

data['correct_home_goals'] = data['home_goals'] == data['actual_home_goals']
data['correct_away_goals'] = data['away_goals'] == data['actual_away_goals']
data['correct_winner'] = data['predicted_winner'] == data['winner']

#print predicted winner and actual winner, predicted score and actual score
print(data[[ 'home_goals', 'away_goals', 'actual_home_goals', 'actual_away_goals', 'predicted_winner', 'winner', 'correct_winner']])


home_goals_accuracy = data['correct_home_goals'].mean()
away_goals_accuracy = data['correct_away_goals'].mean()
winner_accuracy = data['correct_winner'].mean()

print(f"Home Goals Accuracy: {home_goals_accuracy:.2f}")
print(f"Away Goals Accuracy: {away_goals_accuracy:.2f}")
print(f"Winner Accuracy: {winner_accuracy:.2f}")
