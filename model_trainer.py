import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 

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
    # Create a feature for goal difference
    data['goal_difference'] = data['home_score'] - data['away_score']
    
    # Define the target variable (1 if home team wins, 0 otherwise)
    data['target'] = (data['home_score'] > data['away_score']).astype(int)
    
    features = ['home_team', 'away_team', 'tournament', 'home_ranking', 'away_ranking', 'neutral']
    X = data[features]
    y = data['target']
    
    return X, y

# Model Training Component
def train_models(X, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }
    
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models, X_test, y_test

# Model Evaluation Component
def evaluate_models(trained_models, X_test, y_test):
    results = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'report': report
        }
    
    return results

# Main Component to Run the Pipeline and Compare Models
def main(file_path):
    # Load and preprocess data
    data, label_encoders = load_and_preprocess_data(file_path)
    
    # Create features
    X, y = create_features(data)
    
    # Check target variable distribution
    print(y.value_counts(normalize=True))
    
    # Train models
    trained_models, X_test, y_test = train_models(X, y)
    
    # Evaluate models
    results = evaluate_models(trained_models, X_test, y_test)
    
    # Print results
    for name, metrics in results.items():
        print(f"Model: {name}")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Classification Report:\n{metrics['report']}\n")
    
    return trained_models, label_encoders


# Example usage
file_path = 'Results\\results_with_rankings.csv'
trained_models, label_encoders = main(file_path)

# Save trained models and label encoders for future use with joblib

joblib.dump(trained_models, 'models\\trained_models.joblib')
joblib.dump(label_encoders, 'models\\label_encoders.joblib')


