import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
import joblib
import os

def load_and_preprocess_data(data_file):
    data = pd.read_csv(data_file)
    
    label_encoders = {}
    for column in ['home_team', 'away_team']:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    
    return data, label_encoders

def create_features(data):
    features = data[['home_ranking', 'away_ranking', 'home_avg_goals_scored', 'away_avg_goals_scored', 
                     'home_avg_goals_conceded', 'away_avg_goals_conceded', 'neutral']]
    y_home = data['home_score']
    y_away = data['away_score']
    return features, y_home, y_away

# Load and preprocess data
data_file = 'Results/results_with_averages.csv'
data, label_encoders = load_and_preprocess_data(data_file)

X, y_home, y_away = create_features(data)
X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(X, y_home, y_away, test_size=0.2, random_state=42)

poisson_home = PoissonRegressor(alpha=1)
poisson_home.fit(X_train, y_home_train)

poisson_away = PoissonRegressor(alpha=1)
poisson_away.fit(X_train, y_away_train)

# Create the models directory if it doesn't exist
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Save the models
joblib.dump(poisson_home, os.path.join(models_dir, 'poisson_home.pkl'))
joblib.dump(poisson_away, os.path.join(models_dir, 'poisson_away.pkl'))
joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))
