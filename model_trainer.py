import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
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
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'cv_mean_accuracy': mean_cv_score,
            'test_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'model': model
        }
    
    return results

# Visualization Functions
def plot_confusion_matrix(conf_matrix, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_cv_scores(cv_scores):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(cv_scores.keys()), y=[scores['cv_mean_accuracy'] for scores in cv_scores.values()])
    plt.title('Cross-Validation Mean Accuracy Scores')
    plt.xlabel('Model')
    plt.ylabel('CV Mean Accuracy')
    plt.show()

def plot_model_performance(results):
    metrics = ['test_accuracy', 'precision', 'recall', 'f1_score']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        sns.barplot(x=list(results.keys()), y=[results[model][metric] for model in results], ax=axes[idx])
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Scores')
        axes[idx].set_xlabel('Model')
        axes[idx].set_ylabel(metric.replace("_", " ").title())
    
    plt.tight_layout()
    plt.show()

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
    
    # Print and visualize results
    for name, metrics in results.items():
        print(f"Model: {name}")
        print(f"CV Mean Accuracy: {metrics['cv_mean_accuracy']}")
        print(f"Test Accuracy: {metrics['test_accuracy']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1 Score: {metrics['f1_score']}")
        print(f"Classification Report:\n{metrics['classification_report']}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")
        
        # Plot confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'], name)
    
    # Plot cross-validation scores
    plot_cv_scores(results)
    
    # Plot model performance metrics
    plot_model_performance(results)
    
    return results, label_encoders

# Example usage
file_path = 'Results/results_with_rankings.csv'
results, label_encoders = main(file_path)

# Save trained models and label encoders for future use with joblib
#joblib.dump({name: metrics['model'] for name, metrics in results.items()}, 'models/trained_models.joblib')
#joblib.dump(label_encoders, 'models/label_encoders.joblib')
