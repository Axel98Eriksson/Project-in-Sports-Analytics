import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
import joblib

# Load your dataset
data = pd.read_csv('Results\oskars_special_mix.csv')

# Convert date to datetime
data['date'] = pd.to_datetime(data['date'])

#save the rows after 2023-01-01
data_after_2023 = data[data['date'] >= '2023-01-01']
data_after_2023.to_csv('Oskars_predictor\oskars_special_mix_after_2023.csv', index=False)

#filter out rows after 2023-01-01
data = data[data['date'] < '2023-01-01']
data['date'] = pd.to_datetime(data['date'])


#filet out the columns that are not needed
data = data[['home_team', 'away_team', 'country', 'home_ranking', 'away_ranking', 'home_avg_goals', 'away_avg_goals', 'home_wins', 'away_wins', 'home_losses', 'away_losses', 'home_draws', 'away_draws', 'home_score', 'away_score']]


# Features and target
X = data[['home_team', 'away_team', 'country', 'home_ranking', 'away_ranking', 'home_avg_goals', 'away_avg_goals', 'home_wins', 'away_wins', 'home_losses', 'away_losses', 'home_draws', 'away_draws']]
y_home = data['home_score']
y_away = data['away_score']

# Preprocess categorical and numerical features
categorical_features = ['home_team', 'away_team']
numerical_features = ['home_ranking', 'away_ranking', 'home_avg_goals', 'away_avg_goals', 'home_wins', 'away_wins', 'home_losses', 'away_losses', 'home_draws', 'away_draws']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])

# Create a pipeline for the Poisson Regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', PoissonRegressor())
])

# Train the model for home goals
pipeline.fit(X, y_home)

# Save the trained model to a file
joblib.dump(pipeline, 'Oskars_predictor/home_goals_model.pkl')

# Train the model for away goals
pipeline.fit(X, y_away)

# Save the trained model to a file
joblib.dump(pipeline, 'Oskars_predictor/away_goals_model.pkl')

#save the preprocessor to a file
#joblib.dump(preprocessor, 'Oskars_predictor/preprocessor.pkl')
