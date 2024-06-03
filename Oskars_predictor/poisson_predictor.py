import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = 'Results/results_with_rankings_wo_conference.csv'
data = pd.read_csv(file_path)

# Convert date to datetime and extract additional features
data['match_date'] = pd.to_datetime(data['date'])
data['day_of_week'] = data['match_date'].dt.dayofweek
data['month'] = data['match_date'].dt.month

# Define the features and target variables
features = data[['home_team', 'away_team', 'home_ranking', 'away_ranking', 'neutral', 'day_of_week', 'month']]
target_home = data['home_score']
target_away = data['away_score']

# One-hot encode categorical features
categorical_features = ['home_team', 'away_team', 'neutral']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

# Split the data into training and testing sets
X_train, X_test, y_train_home, y_test_home, y_train_away, y_test_away = train_test_split(
    features, target_home, target_away, test_size=0.2, random_state=42
)

# Create and fit the PoissonRegressor model pipelines for home and away scores
model_pipeline_home = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', PoissonRegressor())
])

model_pipeline_away = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', PoissonRegressor())
])

model_pipeline_home.fit(X_train, y_train_home)
model_pipeline_away.fit(X_train, y_train_away)

# Function to predict match outcome probabilities
def predict_match_outcome(home_team, away_team, home_ranking, away_ranking, neutral=0, day_of_week=2, month=6):
    match_data = pd.DataFrame({
        'home_team': [home_team],
        'away_team': [away_team],
        'home_ranking': [home_ranking],
        'away_ranking': [away_ranking],
        'neutral': [neutral],
        'day_of_week': [day_of_week],
        'month': [month]
    })

    # Predict expected goals for home and away teams
    home_goals = model_pipeline_home.predict(match_data)[0]
    away_goals = model_pipeline_away.predict(match_data)[0]

    # Simulate match outcomes
    home_win_prob = 0
    away_win_prob = 0
    draw_prob = 0
    num_simulations = 10000

    for _ in range(num_simulations):
        home_score = np.random.poisson(home_goals)
        away_score = np.random.poisson(away_goals)

        if home_score > away_score:
            home_win_prob += 1
        elif home_score < away_score:
            away_win_prob += 1
        else:
            draw_prob += 1

    home_win_prob /= num_simulations
    away_win_prob /= num_simulations
    draw_prob /= num_simulations

    return home_win_prob, draw_prob, away_win_prob

# Predict the outcome probabilities for a specific match
home_team = 'France'
away_team = 'Sweden'
home_ranking = 2
away_ranking = 27

home_win_prob, draw_prob, away_win_prob = predict_match_outcome(home_team, away_team, home_ranking, away_ranking)
print(f"Home Win Probability: {home_win_prob:.2%}")
print(f"Draw Probability: {draw_prob:.2%}")
print(f"Away Win Probability: {away_win_prob:.2%}")
