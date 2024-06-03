import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from model_trainer_wGoals import load_and_preprocess_data
import random
import joblib
import os

def create_features(data):
    features = data[['home_ranking', 'away_ranking', 'home_avg_goals_scored', 'away_avg_goals_scored', 
                     'home_avg_goals_conceded', 'away_avg_goals_conceded', 'neutral']]
    y_home = data['home_score']
    y_away = data['away_score']
    return features, y_home, y_away

def match_outcome(home_win_prob, draw_prob, away_win_prob):
    outcomes = ['Home Win', 'Draw', 'Away Win']
    probabilities = [home_win_prob, draw_prob, away_win_prob]
    
    # Ensure probabilities sum to 1
    assert sum(probabilities) < 1.05 and sum(probabilities) > 0.95, "Probabilities must sum to 1"
    
    return random.choices(outcomes, weights=probabilities)[0]

# Load and preprocess data
data_file = 'Results/results_with_averages.csv'
data, label_encoders = load_and_preprocess_data(data_file)

X, y_home, y_away = create_features(data)
X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(X, y_home, y_away, test_size=0.2, random_state=42)

poisson_home = PoissonRegressor()
poisson_home.fit(X_train, y_home_train)

poisson_away = PoissonRegressor()
poisson_away.fit(X_train, y_away_train)

# Create the models directory if it doesn't exist
# models_dir = 'models'
# os.makedirs(models_dir, exist_ok=True)
# Save the models
# joblib.dump(poisson_home, os.path.join(models_dir, 'poisson_home.pkl'))
# joblib.dump(poisson_away, os.path.join(models_dir, 'poisson_away.pkl'))

def create_prediction_features(home_ranking, away_ranking, home_avg_goals_scored, away_avg_goals_scored, home_avg_goals_conceded, away_avg_goals_conceded, neutral):
    data = {
        'home_ranking': [home_ranking],
        'away_ranking': [away_ranking],
        'home_avg_goals_scored': [home_avg_goals_scored],
        'away_avg_goals_scored': [away_avg_goals_scored],
        'home_avg_goals_conceded': [home_avg_goals_conceded],
        'away_avg_goals_conceded': [away_avg_goals_conceded],
        'neutral': [neutral]
    }
    return pd.DataFrame(data)

def poisson_probability(lmbda, k):
    """Calculate the Poisson probability of k given lambda."""
    return (lmbda**k * np.exp(-lmbda)) / np.math.factorial(k)

# Select a random match from the data
random_match = data.sample(n=1).iloc[0]

home_team_id = random_match['home_team']
away_team_id = random_match['away_team']
home_ranking = random_match['home_ranking']
away_ranking = random_match['away_ranking']
home_avg_goals_scored = random_match['home_avg_goals_scored']
away_avg_goals_scored = random_match['away_avg_goals_scored']
home_avg_goals_conceded = random_match['home_avg_goals_conceded']
away_avg_goals_conceded = random_match['away_avg_goals_conceded']
neutral = random_match['neutral']

# Decode team names using the label encoders
home_team = label_encoders['home_team'].inverse_transform([home_team_id])[0]
away_team = label_encoders['away_team'].inverse_transform([away_team_id])[0]

features = create_prediction_features(home_ranking, away_ranking, home_avg_goals_scored, away_avg_goals_scored, home_avg_goals_conceded, away_avg_goals_conceded, neutral)
home_lambda = poisson_home.predict(features)[0]
away_lambda = poisson_away.predict(features)[0]

max_goals = 5
goal_distribution_matrix = np.zeros((max_goals + 1, max_goals + 1))

for i in range(max_goals + 1):
    for j in range(max_goals + 1):
        home_goal_prob = poisson_probability(home_lambda, i)
        away_goal_prob = poisson_probability(away_lambda, j)
        goal_distribution_matrix[i, j] = home_goal_prob * away_goal_prob

goal_distribution_matrix /= goal_distribution_matrix.sum()

# Calculate probabilities
home_win_prob = np.sum(np.tril(goal_distribution_matrix, -1))
draw_prob = np.sum(np.diag(goal_distribution_matrix))
away_win_prob = np.sum(np.triu(goal_distribution_matrix, 1))

#print("Goal Distribution Matrix:")
#print(goal_distribution_matrix)

print(f"\n{home_team} vs. {away_team}")
print("Probabilities:")
print(f"Home Win Probability: {home_win_prob * 100:.1f}%")
print(f"Draw Probability: {draw_prob * 100:.1f}%")
print(f"Away Win Probability: {away_win_prob * 100:.1f}%")

result = match_outcome(home_win_prob, draw_prob, away_win_prob)

# Display the winning team's name
if result == 'Home Win':
    print(f"The winning team is: {home_team}")
elif result == 'Away Win':
    print(f"The winning team is: {away_team}")
else:
    print("The match is a draw.")
