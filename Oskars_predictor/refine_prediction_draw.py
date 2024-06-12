import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load your dataset
data = pd.read_csv('Results\oskars_special_mix.csv')

# Convert date to datetime
data['date'] = pd.to_datetime(data['date'])

# #filter out rows after 2023-01-01
# data = data[data['date'] < '2023-01-01']
# data['date'] = pd.to_datetime(data['date'])

# Create the target variable for draw: 1 if draw, else 0
data['is_draw'] = (data['home_score'] == data['away_score']).astype(int)

# Select features relevant to predicting draws
# You can choose different features based on your analysis
features = [
    'home_ranking', 'away_ranking', 
    'home_avg_goals_scored', 'away_avg_goals_scored',
    'home_avg_goals_conceded', 'away_avg_goals_conceded',
    'home_wins', 'away_wins', 'home_losses', 'away_losses', 
    'home_draws', 'away_draws'
]

X = data[features]
y = data['is_draw']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
draw_model = RandomForestClassifier(n_estimators=100, random_state=42)
draw_model.fit(X_train, y_train)

# Evaluate the model
y_pred_train = draw_model.predict(X_train)
y_pred_test = draw_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred_test))

print("Confusion Matrix on Test Set:")
print(confusion_matrix(y_test, y_pred_test))

# Save the trained model
joblib.dump(draw_model, 'Oskars_predictor/draw_model_full.pkl')
print("Draw model saved as 'draw_model.pkl'")
