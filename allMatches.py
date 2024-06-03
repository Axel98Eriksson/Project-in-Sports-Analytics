import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Define the groups and teams
teams = {
    "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
    "B": ["Spain", "Croatia", "Italy", "Albania"],
    "C": ["Poland", "Netherlands", "Slovenia", "Denmark"],
    "D": ["Serbia", "England", "Romania", "Ukraine"],
    "E": ["Belgium", "Slovakia", "Austria", "France"],
    "F": ["Turkey", "Georgia", "Portugal", "Czechia"]
}

# Group stage matches
group_stage_matches = [
    ("2024-06-14", "Germany", "Scotland", "UEFA Euro"),
    ("2024-06-15", "Hungary", "Switzerland", "UEFA Euro"),
    ("2024-06-15", "Spain", "Croatia", "UEFA Euro"),
    ("2024-06-15", "Italy", "Albania", "UEFA Euro"),
    ("2024-06-16", "Poland", "Netherlands", "UEFA Euro"),
    ("2024-06-16", "Slovenia", "Denmark", "UEFA Euro"),
    ("2024-06-16", "Serbia", "England", "UEFA Euro"),
    ("2024-06-17", "Romania", "Ukraine", "UEFA Euro"),
    ("2024-06-17", "Belgium", "Slovakia", "UEFA Euro"),
    ("2024-06-17", "Austria", "France", "UEFA Euro"),
    ("2024-06-18", "Turkey", "Georgia", "UEFA Euro"),
    ("2024-06-18", "Portugal", "Czechia", "UEFA Euro"),
    ("2024-06-19", "Croatia", "Albania", "UEFA Euro"),
    ("2024-06-19", "Germany", "Hungary", "UEFA Euro"),
    ("2024-06-19", "Scotland", "Switzerland", "UEFA Euro"),
    ("2024-06-20", "Slovenia", "Serbia", "UEFA Euro"),
    ("2024-06-20", "Denmark", "England", "UEFA Euro"),
    ("2024-06-20", "Spain", "Italy", "UEFA Euro"),
    ("2024-06-21", "Slovakia", "Ukraine", "UEFA Euro"),
    ("2024-06-21", "Poland", "Austria", "UEFA Euro"),
    ("2024-06-21", "Netherlands", "France", "UEFA Euro"),
    ("2024-06-22", "Georgia", "Czechia", "UEFA Euro"),
    ("2024-06-22", "Turkey", "Portugal", "UEFA Euro"),
    ("2024-06-22", "Belgium", "Romania", "UEFA Euro"),
    ("2024-06-23", "Switzerland", "Germany", "UEFA Euro"),
    ("2024-06-23", "Scotland", "Hungary", "UEFA Euro"),
    ("2024-06-24", "Albania", "Spain", "UEFA Euro"),
    ("2024-06-24", "Croatia", "Italy", "UEFA Euro"),
    ("2024-06-25", "France", "Poland", "UEFA Euro"),
    ("2024-06-25", "Netherlands", "Austria", "UEFA Euro"),
    ("2024-06-25", "England", "Slovenia", "UEFA Euro"),
    ("2024-06-25", "Denmark", "Serbia", "UEFA Euro"),
    ("2024-06-26", "Ukraine", "Belgium", "UEFA Euro"),
    ("2024-06-26", "Slovakia", "Romania", "UEFA Euro"),
    ("2024-06-26", "Czechia", "Turkey", "UEFA Euro"),
    ("2024-06-26", "Georgia", "Portugal", "UEFA Euro"),
]

# Create DataFrame for group stage matches
group_stage_df = pd.DataFrame(group_stage_matches, columns=["Date", "Home Team", "Away Team", "Tournament"])

# Prompt user to select a model type
model_type = input("Enter the model type (RandomForest, LogisticRegression, SVM): ")

# Load the corresponding model based on user input
model_file_map = {
    "RandomForest": "path_to_your_random_forest_model.pkl",
    "LogisticRegression": "path_to_your_logistic_regression_model.pkl",
    "SVM": "path_to_your_svm_model.pkl"
}

model_file = model_file_map.get(model_type, None)
if not model_file:
    raise ValueError("Invalid model type entered.")

model = joblib.load(model_file)

# Encode teams
label_encoder = LabelEncoder()
all_teams = sum(teams.values(), [])
label_encoder.fit(all_teams)

# Function to predict match result using the selected model
def predict_match_result(home_team, away_team):
    # Create feature vector for the match (adjust this part based on your feature extraction method)
    home_team_encoded = label_encoder.transform([home_team])[0]
    away_team_encoded = label_encoder.transform([away_team])[0]
    features = np.array([home_team_encoded, away_team_encoded]).reshape(1, -1)
    
    # Predict the outcome (e.g., 0: Home Win, 1: Draw, 2: Away Win)
    prediction = model.predict(features)
    if prediction == 0:
        return (home_team, 3), (away_team, 0)
    elif prediction == 1:
        return (home_team, 1), (away_team, 1)
    else:
        return (home_team, 0), (away_team, 3)

# Simulate the group stage matches
results = []
for index, match in group_stage_df.iterrows():
    result = predict_match_result(match["Home Team"], match["Away Team"])
    results.append(result)

# Process group stage results
group_points = {group: {team: 0 for team in teams[group]} for group in teams}
for result in results:
    for team, points in result:
        for group, group_teams in teams.items():
            if team in group_teams:
                group_points[group][team] += points

# Calculate group standings
group_standings = {}
for group in teams:
    sorted_teams = sorted(group_points[group].items(), key=lambda x: x[1], reverse=True)
    group_standings[group] = sorted_teams

# Select top 2 teams from each group
top_teams = {group: [team for team, _ in group_standings[group][:2]] for group in group_standings}

# Determine the best 4 third-placed teams
third_place_teams = [(group, group_standings[group][2]) for group in group_standings]
third_place_teams_sorted = sorted(third_place_teams, key=lambda x: x[1][1], reverse=True)
best_third_place_teams = [team for group, (team, _) in third_place_teams_sorted[:4]]

# Map positions in knockout stage
knockout_positions = {
    "1A": top_teams["A"][0], "2C": top_teams["C"][1], "1B": top_teams["B"][0], "2A": top_teams["A"][1],
    "1C": top_teams["C"][0], "3D/E/F": best_third_place_teams[0], "1F": top_teams["F"][0], "2B": top_teams["B"][1],
    "1D": top_teams["D"][0], "2E": top_teams["E"][1], "1E": top_teams["E"][0], "3A/B/C/D": best_third_place_teams[1],
    "2D": top_teams["D"][1], "2F": top_teams["F"][1], "1G": best_third_place_teams[2], "3B/D/E/F": best_third_place_teams[3]
}

# Define knockout stage matches
knockout_stage_matches = [
    ("2024-06-29", "Round of 16", knockout_positions["1A"], knockout_positions["2C"]),
    ("2024-06-29", "Round of 16", knockout_positions["1B"], knockout_positions["2A"]),
    ("2024-06-30", "Round of 16", knockout_positions["1C"], knockout_positions["3D/E/F"]),
    ("2024-06-30", "Round of 16", knockout_positions["1F"], knockout_positions["2B"]),
    ("2024-07-01", "Round of 16", knockout_positions["1D"], knockout_positions["2E"]),
    ("2024-07-01", "Round of 16", knockout_positions["1E"], knockout_positions["3A/B/C/D"]),
    ("2024-07-02", "Round of 16", knockout_positions["2D"], knockout_positions["2F"]),
    ("2024-07-02", "Round of 16", knockout_positions["1G"], knockout_positions["3B/D/E/F"])
]

# Function to simulate knockout match using selected model
def simulate_knockout_match(home_team, away_team):
    home_team, away_team = label_encoder.inverse_transform([home_team]), label_encoder.inverse_transform([away_team])
    result = predict_match_result(home_team[0], away_team[0])
    return result[0][0] if result[0][1] > result[1][1] else result[1][0]

# Simulate Round of 16
round_of_16_winners = []
for match in knockout_stage_matches:
    winner = simulate_knockout_match(match[2], match[3])
    round_of_16_winners.append(winner)

# Define quarter-finals
quarter_final_matches = [
    ("2024-07-05", "Quarter-finals", round_of_16_winners[0], round_of_16_winners[1]),
    ("2024-07-05", "Quarter-finals", round_of_16_winners[2], round_of_16_winners[3]),
    ("2024-07-06", "Quarter-finals", round_of_16_winners[4], round_of_16_winners[5]),
    ("2024-07-06", "Quarter-finals", round_of_16_winners[6], round_of_16_winners[7])
]

# Simulate Quarter-finals
quarter_final_winners = []
for match in quarter_final_matches:
    winner = simulate_knockout_match(match[2], match[3])
    quarter_final_winners.append(winner)

# Define semi-finals
semi_final_matches = [
    ("2024-07-09", "Semi-finals", quarter_final_winners[0], quarter_final_winners[1]),
    ("2024-07-10", "Semi-finals", quarter_final_winners[2], quarter_final_winners[3])
]

# Simulate Semi-finals
semi_final_winners = []
for match in semi_final_matches:
    winner = simulate_knockout_match(match[2], match[3])
    semi_final_winners.append(winner)

# Define Final
final_match = ("2024-07-14", "Final", semi_final_winners[0], semi_final_winners[1])

# Combine all matches into a DataFrame
all_matches = group_stage_df.values.tolist() + knockout_stage_matches + quarter_final_matches + semi_final_matches + [final_match]
all_matches_df = pd.DataFrame(all_matches, columns=["Date", "Stage", "Home Team", "Away Team", "Tournament"])

# Save to CSV
all_matches_df.to_csv("UEFA_Euro_2024_Simulation.csv", index=False)

print(all_matches_df)
