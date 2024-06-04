import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the trained models
home_goals_model = joblib.load('Oskars_predictor/home_goals_model.pkl')
away_goals_model = joblib.load('Oskars_predictor/away_goals_model.pkl')

def simulate_match(home_team, away_team, country, home_ranking, away_ranking, home_avg_goals, away_avg_goals, home_wins, away_wins, home_losses, away_losses, home_draws, away_draws, n_simulations=10000):
   # input_data = np.array([[home_team, away_team,country, home_ranking, away_ranking, home_avg_goals, away_avg_goals, home_wins, away_wins, home_losses, away_losses, home_draws, away_draws]])

    input_data = pd.DataFrame({
        'home_team': [home_team],
        'away_team': [away_team],
        'country': [country],
        'home_ranking': [home_ranking],
        'away_ranking': [away_ranking],
        'home_avg_goals': [home_avg_goals],
        'away_avg_goals': [away_avg_goals],
        'home_wins': [home_wins],
        'away_wins': [away_wins],
        'home_losses': [home_losses],
        'away_losses': [away_losses],
        'home_draws': [home_draws],
        'away_draws': [away_draws]
    })

    home_goals_pred = home_goals_model.predict(input_data)[0]
    away_goals_pred = away_goals_model.predict(input_data)[0]

    home_wins = 0
    away_wins = 0
    draws = 0

    for _ in range(n_simulations):
        home_goals = np.random.poisson(home_goals_pred)
        away_goals = np.random.poisson(away_goals_pred)

        if home_goals > away_goals:
            home_wins += 1
        elif home_goals < away_goals:
            away_wins += 1
        else:
            draws += 1

    total_matches = home_wins + away_wins + draws
    home_win_prob = home_wins / total_matches
    away_win_prob = away_wins / total_matches
    draw_prob = draws / total_matches

    return home_win_prob, draw_prob, away_win_prob

# Load the test dataset
test_data = pd.read_csv('Oskars_predictor/oskars_special_mix_after_2023.csv')

# Prepare features and actual outcomes
X_test = test_data[['home_team', 'away_team', 'country', 'home_ranking', 'away_ranking', 'home_avg_goals', 'away_avg_goals', 'home_wins', 'away_wins', 'home_losses', 'away_losses', 'home_draws', 'away_draws']]
y_test_home = test_data['home_score']
y_test_away = test_data['away_score']

# Initialize lists to store predictions and actual outcomes
predicted_results = []
actual_results = []

# Iterate through the test set and make predictions
for index, row in test_data.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    country = row['country']
    home_ranking = row['home_ranking']
    away_ranking = row['away_ranking']
    home_avg_goals = row['home_avg_goals']
    away_avg_goals = row['away_avg_goals']
    home_wins = row['home_wins']
    away_wins = row['away_wins']
    home_losses = row['home_losses']
    away_losses = row['away_losses']
    home_draws = row['home_draws']
    away_draws = row['away_draws']

    home_win_prob, draw_prob, away_win_prob = simulate_match(home_team, away_team, country, home_ranking, away_ranking, home_avg_goals, away_avg_goals, home_wins, away_wins, home_losses, away_losses, home_draws, away_draws)

    print(f'{home_team} vs {away_team}: Home Win: {home_win_prob:.2f}, Draw: {draw_prob:.2f}, Away Win: {away_win_prob:.2f}')

    # Determine the predicted result
    if home_win_prob > away_win_prob and home_win_prob > draw_prob:
        predicted_results.append('Home Win')
    elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
        predicted_results.append('Away Win')
    else:
        predicted_results.append('Draw')

    # Determine the actual result
    if row['home_score'] > row['away_score']:
        actual_results.append('Home Win')
    elif row['home_score'] < row['away_score']:
        actual_results.append('Away Win')
    else:
        actual_results.append('Draw')

# Compare predictions to actual outcomes
accuracy = accuracy_score(actual_results, predicted_results)
print(f'Prediction accuracy: {accuracy * 100:.2f}%')

# display a confusion matrix
cm = confusion_matrix(actual_results, predicted_results, labels=['Home Win', 'Draw', 'Away Win'])
cmd = ConfusionMatrixDisplay(cm, display_labels=['Home Win', 'Draw', 'Away Win'])
cmd.plot()