import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Load the dataset
file_path = 'Results/results_with_rankings_wo_conference.csv'
data = pd.read_csv(file_path)

# Extract additional features
data['match_date'] = pd.to_datetime(data['date'])
data['day_of_week'] = data['match_date'].dt.dayofweek
data['month'] = data['match_date'].dt.month

# Features and target variables
features = data[['home_ranking', 'away_ranking', 'neutral', 'day_of_week', 'month']]
features = pd.get_dummies(features, columns=['neutral'], drop_first=True)
target_home = data['home_score']
target_away = data['away_score']

# Split the data into training and testing sets
X_train, X_test, y_train_home, y_test_home, y_train_away, y_test_away = train_test_split(
    features, target_home, target_away, test_size=0.2, random_state=42
)


# Define hyperparameters for RandomForestRegressor
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Define hyperparameters for XGBRegressor
xgb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}


# Define hyperparameters for PoissonRegressor
poisson_params = {
    'alpha': [41.0, 42.5, 43.0, 43.5, 44.0, 44.5, 44.8, 45.0]
}

# Grid search for RandomForestRegressor
rf_grid_home = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid_away = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Grid search for XGBRegressor
xgb_grid_home = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid_away = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Grid search for PoissonRegressor
poisson_grid_home = GridSearchCV(PoissonRegressor(), poisson_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
poisson_grid_away = GridSearchCV(PoissonRegressor(), poisson_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the models
rf_grid_home.fit(X_train, y_train_home)
rf_grid_away.fit(X_train, y_train_away)

xgb_grid_home.fit(X_train, y_train_home)
xgb_grid_away.fit(X_train, y_train_away)

poisson_grid_home.fit(X_train, y_train_home)
poisson_grid_away.fit(X_train, y_train_away)


# Get the best parameters and scores
best_rf_params_home = rf_grid_home.best_params_
best_rf_score_home = -rf_grid_home.best_score_
best_rf_params_away = rf_grid_away.best_params_
best_rf_score_away = -rf_grid_away.best_score_

best_xgb_params_home = xgb_grid_home.best_params_
best_xgb_score_home = -xgb_grid_home.best_score_
best_xgb_params_away = xgb_grid_away.best_params_
best_xgb_score_away = -xgb_grid_away.best_score_

best_poisson_params_home = poisson_grid_home.best_params_
best_poisson_score_home = -poisson_grid_home.best_score_
best_poisson_params_away = poisson_grid_away.best_params_
best_poisson_score_away = -poisson_grid_away.best_score_

# Print the results
print("Best hyperparameters and scores for home_score predictions:")
print(f"RandomForestRegressor: {best_rf_params_home} with MSE: {best_rf_score_home}")
print(f"XGBRegressor: {best_xgb_params_home} with MSE: {best_xgb_score_home}")
print(f"PoissonRegressor: {best_poisson_params_home} with MSE: {best_poisson_score_home}")

print("\nBest hyperparameters and scores for away_score predictions:")
print(f"RandomForestRegressor: {best_rf_params_away} with MSE: {best_rf_score_away}")
print(f"XGBRegressor: {best_xgb_params_away} with MSE: {best_xgb_score_away}")
print(f"PoissonRegressor: {best_poisson_params_away} with MSE: {best_poisson_score_away}")
