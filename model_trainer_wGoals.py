import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Data Loading and Preprocessing Component
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Handle categorical features with Label Encoding
    label_encoders = {}
    for column in ['home_team', 'away_team', 'tournament', 'city', 'country']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    # Convert boolean 'neutral' to integer
    data['neutral'] = data['neutral'].astype(int)
    
    return data, label_encoders

# Feature Engineering Component
def create_features(data):
    features = ['home_team', 'away_team', 'tournament', 'home_ranking', 'away_ranking', 'neutral']
    X = data[features].copy()
    
    # Ensure symmetry for neutral games
    X['team1'] = X.apply(lambda row: min(row['home_team'], row['away_team']) if row['neutral'] == 1 else row['home_team'], axis=1)
    X['team2'] = X.apply(lambda row: max(row['home_team'], row['away_team']) if row['neutral'] == 1 else row['away_team'], axis=1)
    X['ranking_diff'] = X.apply(lambda row: abs(row['home_ranking'] - row['away_ranking']) if row['neutral'] == 1 else row['home_ranking'] - row['away_ranking'], axis=1)
    
    y_home = data['home_score']
    y_away = data['away_score']
    
    return X[['team1', 'team2', 'tournament', 'ranking_diff', 'neutral']], y_home, y_away

# Model Training Component
def train_models(X, y_home, y_away):
    # Split the data into training and test sets
    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(X, y_home, y_away, test_size=0.2, random_state=42)
    

    models = {
        '1': RandomForestClassifier(random_state=42),
        '2': RandomForestRegressor(random_state=42),
        '3': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        '4': GaussianNB(),
        '5': KNeighborsClassifier(),
        '6': DecisionTreeClassifier(random_state=42)
    }

    model_home = models['6']
    model_away = models['6']


    
    print("Training RandomForest model for home goals...")
    model_home.fit(X_train, y_home_train)
    print("Training RandomForest model for away goals...")
    model_away.fit(X_train, y_away_train)
    
    return model_home, model_away, X_test, y_home_test, y_away_test

# Model Evaluation Component
def evaluate_models(model_home, model_away, X_test, y_home_test, y_away_test):
    y_home_pred = model_home.predict(X_test)
    y_away_pred = model_away.predict(X_test)
    
    home_mse = mean_squared_error(y_home_test, y_home_pred)
    away_mse = mean_squared_error(y_away_test, y_away_pred)
    
    print(f"Home Model Mean Squared Error: {home_mse}")
    print(f"Away Model Mean Squared Error: {away_mse}")

# Main Component to Run the Pipeline and Compare Models
def main(file_path):
    # Load and preprocess data
    data, label_encoders = load_and_preprocess_data(file_path)
    
    # Create features
    X, y_home, y_away = create_features(data)
    
    # Train models
    model_home, model_away, X_test, y_home_test, y_away_test = train_models(X, y_home, y_away)
    
    # Evaluate models
    evaluate_models(model_home, model_away, X_test, y_home_test, y_away_test)
    
    return model_home, model_away, label_encoders

# Example usage
file_path = 'Results/results_with_rankings_wo_conference.csv'  # Ensure this path is correct
model_home, model_away, label_encoders = main(file_path)

# Save trained models and label encoders for future use with joblib
joblib.dump(model_home, 'models/model_home.joblib')
joblib.dump(model_away, 'models/model_away.joblib')
joblib.dump(label_encoders, 'models/label_encoders.joblib')
