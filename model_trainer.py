import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

# Model Training and Cross-Validation Component
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'GaussianNB': GaussianNB(),
        'KNeighbors': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=10)
        mean_cv_score = cv_scores.mean()
        
        # Train the model on the full training set
        model.fit(X_train, y_train)
        
        # Evaluate on the test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        results[name] = {
            'cv_mean_accuracy': mean_cv_score,
            'test_accuracy': accuracy,
            'classification_report': report,
            'model': model
        }
    
    return results

# Main Component to Run the Pipeline and Compare Models
def main(file_path):
    # Load and preprocess data
    data, label_encoders = load_and_preprocess_data(file_path)
    
    # Create features
    X, y = create_features(data)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models with cross-validation
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    # Print results
    for name, metrics in results.items():
        print(f"Model: {name}")
        print(f"CV Mean Accuracy: {metrics['cv_mean_accuracy']}")
        print(f"Test Accuracy: {metrics['test_accuracy']}")
        print(f"Classification Report:\n{metrics['classification_report']}\n")
    
    return results, label_encoders

# Example usage
file_path = 'Results\\results_with_rankings.csv'
results, label_encoders = main(file_path)

# Save trained models and label encoders for future use with joblib
#joblib.dump({name: metrics['model'] for name, metrics in results.items()}, 'models\\trained_models.joblib')
#joblib.dump(label_encoders, 'models\\label_encoders.joblib')
