import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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
def train_models(X, y_home, y_away, model):
    # Split the data into training and test sets
    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(X, y_home, y_away, test_size=0.2, random_state=42)
    
    models = {
        '1': RandomForestRegressor(random_state=42),
        '2': RandomForestClassifier(random_state=42),
        '3': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        '4': GaussianNB(),
        '5': KNeighborsClassifier(),
        '6': DecisionTreeClassifier(random_state=42)
    }
    
    model_home = models[model]
    model_away = models[model]
    
    print(f"Training {model_home.__class__.__name__} model for home goals...")
    model_home.fit(X_train, y_home_train)
    print(f"Training {model_away.__class__.__name__} model for away goals...")
    model_away.fit(X_train, y_away_train)
    
    return model_home, model_away, X_test, y_home_test, y_away_test

# Model Evaluation Component
def evaluate_models(model_home, model_away, X_test, y_home_test, y_away_test):
    y_home_pred = model_home.predict(X_test)
    y_away_pred = model_away.predict(X_test)
    
    home_mse = mean_squared_error(y_home_test, y_home_pred)
    away_mse = mean_squared_error(y_away_test, y_away_pred)
    
    if isinstance(model_home, (RandomForestClassifier, xgb.XGBClassifier, GaussianNB, KNeighborsClassifier, DecisionTreeClassifier)):
        home_accuracy = accuracy_score(y_home_test, y_home_pred)
        away_accuracy = accuracy_score(y_away_test, y_away_pred)
    else:
        home_accuracy = None
        away_accuracy = None
    
    print(f"Home Model Mean Squared Error: {home_mse}")
    print(f"Away Model Mean Squared Error: {away_mse}")
    
    if home_accuracy is not None and away_accuracy is not None:
        print(f"Home Model Accuracy: {home_accuracy}")
        print(f"Away Model Accuracy: {away_accuracy}")
    
    return home_mse, away_mse, home_accuracy, away_accuracy

# Main Component to Run the Pipeline and Compare Models
def main(file_path):
    # Load and preprocess data
    data, label_encoders = load_and_preprocess_data(file_path)
    
    # Create features
    X, y_home, y_away = create_features(data)
    
    results = []
    
    for model in ['1', '2', '3', '4', '5', '6']:
        print(f"Training models with model: {model}")
        
        # Train models
        model_home, model_away, X_test, y_home_test, y_away_test = train_models(X, y_home, y_away, model)
        
        # Evaluate models
        home_mse, away_mse, home_accuracy, away_accuracy = evaluate_models(model_home, model_away, X_test, y_home_test, y_away_test)
        
        results.append({
            'file_path': file_path,
            'model': model,
            'home_mse': home_mse,
            'away_mse': away_mse,
            'home_accuracy': home_accuracy,
            'away_accuracy': away_accuracy
        })
        
        # Save results to a new text file
        with open('Results/model_and_data_test.txt', 'a') as f:
            f.write(f"File: {file_path}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Home Model Mean Squared Error: {home_mse}\n")
            f.write(f"Away Model Mean Squared Error: {away_mse}\n")
            if home_accuracy is not None and away_accuracy is not None:
                f.write(f"Home Model Accuracy: {home_accuracy}\n")
                f.write(f"Away Model Accuracy: {away_accuracy}\n")
            f.write("\n")
    
    return results

# Function to rank the models based on evaluation metrics
def rank_models(results):
    # Rank based on MSE for regressors
    mse_results = [res for res in results if res['home_accuracy'] is None]
    mse_results.sort(key=lambda x: (x['home_mse'] + x['away_mse']))
    
    # Rank based on accuracy for classifiers
    accuracy_results = [res for res in results if res['home_accuracy'] is not None]
    accuracy_results.sort(key=lambda x: -(x['home_accuracy'] + x['away_accuracy']))
    
    ranked_results = mse_results + accuracy_results
    
    for rank, res in enumerate(ranked_results, 1):
        print(f"Rank: {rank}")
        print(f"File: {res['file_path']}")
        print(f"Model: {res['model']}")
        print(f"Home MSE: {res['home_mse']}")
        print(f"Away MSE: {res['away_mse']}")
        if res['home_accuracy'] is not None:
            print(f"Home Accuracy: {res['home_accuracy']}")
            print(f"Away Accuracy: {res['away_accuracy']}")
        print()

# Datasets to train models on
file_path1 = 'Results/results_with_rankings_wo_conference.csv'  # Ensure this path is correct
file_path2 = 'Results/results_all.csv'
file_path3 = 'Results/results_some.csv'

all_results = []

for file_path in [file_path1, file_path2, file_path3]:
    print(f"Training models for file: {file_path}")
    results = main(file_path)
    all_results.extend(results)

# Rank the models
rank_models(all_results)

# Optional: Save trained models and label encoders for future use with joblib
# joblib.dump(model_home, 'models/model_home.joblib')
# joblib.dump(model_away, 'models/model_away.joblib')
# joblib.dump(label_encoders, 'models/label_encoders.joblib')
