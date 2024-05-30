import joblib
import pandas as pd

# Load trained models and label encoders
trained_models = joblib.load('models/trained_models.joblib')
label_encoders = joblib.load('models/label_encoders.joblib')

# Define a function to predict match outcome
def predict_match_outcome(home_team, away_team, tournament, home_ranking, away_ranking, neutral):
    # Prepare input data for prediction
    home_team_encoded = label_encoders['home_team'].transform([home_team])[0]
    away_team_encoded = label_encoders['away_team'].transform([away_team])[0]
    tournament_encoded = label_encoders['tournament'].transform([tournament])[0]
    home_ranking = int(home_ranking)
    away_ranking = int(away_ranking)
    neutral_encoded = int(neutral)

    
    # Create input array for prediction
    input_data = [[home_team_encoded, away_team_encoded, neutral_encoded, tournament_encoded, home_ranking, away_ranking]]
    
    # Predict using each model
    predictions = {}
    for name, model in trained_models.items():
        prediction = model.predict(input_data)[0]
        predictions[name] = prediction
    
    return predictions

# Example usage
home_team = 'Georgia'
home_ranking = 50

away_team = 'Portugal'
away_ranking = 5

neutral = True
tournament = 'UEFA Euro'

predictions = predict_match_outcome(home_team, away_team,tournament,home_ranking, away_ranking, neutral)
print("Predictions for " + home_team +" vs " + away_team)

print(predictions)
for model, prediction in predictions.items():
    outcome = home_team + " Wins"if prediction == 1 else away_team + " Wins"

    odds = pd.DataFrame(predictions, index=['accuracy']).T

    print(f"{model}: {outcome} ({odds.loc[model, 'accuracy']:.2f})")
