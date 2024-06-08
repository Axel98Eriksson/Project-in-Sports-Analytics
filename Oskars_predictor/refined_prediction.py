import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier

# Load the trained models
home_goals_model_po = joblib.load('Oskars_predictor/home_goals_model.pkl')
away_goals_model_po = joblib.load('Oskars_predictor/away_goals_model.pkl')
home_goals_model_xgb = joblib.load('Oskars_predictor/home_goals_model_xgb.pkl')
away_goals_model_xgb = joblib.load('Oskars_predictor/away_goals_model_xgb.pkl')
home_goals_model_rf = joblib.load('Oskars_predictor/home_goals_model_rf.pkl')
away_goals_model_rf = joblib.load('Oskars_predictor/away_goals_model_rf.pkl')
draw_model = joblib.load('Oskars_predictor/draw_model.pkl')  

# Function to simulate the match with draw probability boost
def simulate_match_with_draw_boost(home_team, away_team, country, home_ranking, away_ranking, home_avg_goals_scored, away_avg_goals_scored, home_avg_goals_conceded, away_avg_goals_conceded, home_wins, away_wins, home_losses, away_losses, home_draws, away_draws, n_simulations=10000, draw_boost=1.3):
    input_data = pd.DataFrame({
        'home_team': [home_team],
        'away_team': [away_team],
        'country': [country],
        'home_ranking': [home_ranking],
        'away_ranking': [away_ranking],
        'home_avg_goals_scored': [home_avg_goals_scored],
        'away_avg_goals_scored': [away_avg_goals_scored],
        'home_avg_goals_conceded': [home_avg_goals_conceded],
        'away_avg_goals_conceded': [away_avg_goals_conceded],
        'home_wins': [home_wins],
        'away_wins': [away_wins],
        'home_losses': [home_losses],
        'away_losses': [away_losses],
        'home_draws': [home_draws],
        'away_draws': [away_draws]
    })

    # Predict goals using the ensemble of models
    home_goals_pred_po = home_goals_model_po.predict(input_data)[0]
    away_goals_pred_po = away_goals_model_po.predict(input_data)[0]

    home_goals_pred_xgb = home_goals_model_xgb.predict(input_data)[0]
    away_goals_pred_xgb = away_goals_model_xgb.predict(input_data)[0]

    home_goals_pred_rf = home_goals_model_rf.predict(input_data)[0]
    away_goals_pred_rf = away_goals_model_rf.predict(input_data)[0]

    home_goals_pred = max((0.4 * home_goals_pred_po + 0.3 * home_goals_pred_xgb + 0.3 * home_goals_pred_rf), 0)
    away_goals_pred = max((0.4 * away_goals_pred_po + 0.3 * away_goals_pred_xgb + 0.3 * away_goals_pred_rf), 0)

    home_wins = 0
    away_wins = 0
    draws = 0

    for _ in range(n_simulations):
        home_goals = np.random.poisson(home_goals_pred)
        away_goals = np.random.poisson(away_goals_pred)

        if home_goals > away_goals:
            home_wins += 1
        elif home_goals < away_goals:
            away_wins += 1
        else:
            draws += 1

    total_matches = home_wins + away_wins + draws
    home_win_prob = home_wins / total_matches
    away_win_prob = away_wins / total_matches
    draw_prob = draws / total_matches

    # Boost the draw probability
    draw_prob *= draw_boost
    total_prob = home_win_prob + draw_prob + away_win_prob

    # Normalize probabilities
    home_win_prob /= total_prob
    draw_prob /= total_prob
    away_win_prob /= total_prob

    return home_win_prob, draw_prob, away_win_prob

# Load the test dataset
test_data = pd.read_csv('Oskars_predictor/oskars_special_mix_after_2023.csv')

# Prepare features and actual outcomes
X_test = test_data[['home_team', 'away_team', 'country', 'home_ranking', 'away_ranking', 'home_avg_goals_scored', 'away_avg_goals_scored', 'home_avg_goals_conceded', 'away_avg_goals_conceded', 'home_wins', 'away_wins', 'home_losses', 'away_losses', 'home_draws', 'away_draws']]
y_test_home = test_data['home_score']
y_test_away = test_data['away_score']

# Initialize lists to store predictions and actual outcomes
predicted_results = []
actual_results = []

# Initialize the Isotonic Regression model for calibration
ir = IsotonicRegression(out_of_bounds='clip')

# Iterate through the test set and make predictions
for index, row in test_data.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    country = row['country']
    home_ranking = row['home_ranking']
    away_ranking = row['away_ranking']
    home_avg_goals_scored = row['home_avg_goals_scored']
    away_avg_goals_scored = row['away_avg_goals_scored']
    home_avg_goals_conceded = row['home_avg_goals_conceded']
    away_avg_goals_conceded = row['away_avg_goals_conceded']
    home_wins = row['home_wins']
    away_wins = row['away_wins']
    home_losses = row['home_losses']
    away_losses = row['away_losses']
    home_draws = row['home_draws']
    away_draws = row['away_draws']

    # Simulate the match and get win/draw probabilities
    home_win_prob, draw_prob, away_win_prob = simulate_match_with_draw_boost(
        home_team, away_team, country, home_ranking, away_ranking, home_avg_goals_scored, away_avg_goals_scored,
        home_avg_goals_conceded, away_avg_goals_conceded, home_wins, away_wins, home_losses, away_losses, home_draws, away_draws
    )

    # Predict draw probability using the separate draw model
    draw_features = ['home_ranking', 'away_ranking', 'home_avg_goals_scored', 'away_avg_goals_scored', 'home_avg_goals_conceded', 'away_avg_goals_conceded', 'home_wins', 'away_wins', 'home_losses', 'away_losses', 'home_draws', 'away_draws']
    draw_input = pd.DataFrame({feature: [row[feature]] for feature in draw_features})
    draw_prob_model = draw_model.predict_proba(draw_input)[0][1]

    # Combine draw probability with the draw model's prediction
    combined_draw_prob = max(draw_prob, draw_prob_model)

    # Apply isotonic regression calibration (optional)
    home_win_prob, combined_draw_prob, away_win_prob = ir.fit_transform(
        [home_win_prob, combined_draw_prob, away_win_prob],
        [1, 0, 1]  # Dummy target for fitting the isotonic model
    )

    # Determine the predicted result
    if home_win_prob > away_win_prob and home_win_prob > combined_draw_prob:
        predicted_results.append('Home Win')
    elif away_win_prob > home_win_prob and away_win_prob > combined_draw_prob:
        predicted_results.append('Away Win')
    else:
        predicted_results.append('Draw')

    # Determine the actual result
    if row['home_score'] > row['away_score']:
        actual_results.append('Home Win')
    elif row['home_score'] < row['away_score']:
        actual_results.append('Away Win')
    else:
        actual_results.append('Draw')

# Compare predictions to actual outcomes
accuracy = accuracy_score(actual_results, predicted_results)
print(f'Prediction accuracy: {accuracy * 100:.2f}%')

confusion_matrix = confusion_matrix(actual_results, predicted_results)
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['Home Win', 'Draw', 'Away Win'])

#add header label to the confusion matrix
cm_display = cm_display.plot(include_values=True, cmap='viridis', xticks_rotation='horizontal')
cm_display.ax_.set_title('Combined with trained draw model - Confusion Matrix')
#add the accuracy score
cm_display.ax_.text(0.5, 0.95, f'Prediction accuracy: {accuracy * 100:.2f}%' , horizontalalignment='center', verticalalignment='center', transform=cm_display.ax_.transAxes, color='white', fontsize=12, weight='bold')


cm_display.plot()
plt.show()

cm_display.plot()
plt.show()
