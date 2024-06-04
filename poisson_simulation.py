import math
import pandas as pd
import numpy as np
import random
import joblib

# Define the groups and teams
teams = {
    "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
    "B": ["Spain", "Croatia", "Italy", "Albania"],
    "C": ["England", "Serbia", "Slovenia", "Denmark"],
    "D": ["Netherlands", "France", "Poland", "Austria"],
    "E": ["Ukraine", "Slovakia", "Belgium", "Romania"],
    "F": ["Turkey", "Georgia", "Portugal", "Czech Republic"]
}

# Last five games (friendlies and qualification)
team_stats = {
    "Germany": [16, 3.4, 0.4, 4, 0, 1],  # fifa rank, avg goal scored, avg goal conceded, win, loss, draw
    "Scotland": [39, 1.4, 2, 1, 2, 2],
    "Hungary": [26, 2, 1, 3, 0, 2],
    "Switzerland": [19, 0.6, 0.6, 1, 1, 3],
    "Spain": [8, 2, 1.4, 3, 1, 1],
    "Croatia": [10, 2, 0.4, 4, 1, 0],
    "Italy": [9, 2, 1.2, 3, 1, 1],
    "Albania": [66, 0.8, 1, 1, 2, 2],
    "Poland": [28, 1.8, 0.6, 2, 0, 3],
    "Netherlands": [7, 2.6, 0.4, 4, 1, 0],
    "Slovenia": [57, 1.6, 1, 3, 1, 1],
    "Denmark": [21, 1.2, 0.4, 3, 1, 1],
    "Serbia": [33, 2, 1.2, 3, 1, 1],
    "England": [4, 1.6, 0.8, 2, 1, 2],
    "Romania": [46, 2.0, 1.0, 3, 1, 1],
    "Ukraine": [22, 1.4, 0.6, 3, 0, 2],
    "Belgium": [3, 3.0, 0.8, 3, 0, 2],
    "Slovakia": [48, 1.6, 1.2, 3, 1, 1],
    "Austria": [25, 2.6, 0.8, 4, 1, 0],
    "France": [2, 4.2, 1.4, 3, 1, 1],
    "Turkey": [40, 1.4, 1.6, 2, 2, 1],
    "Georgia": [75, 1.8, 1, 2, 1, 2],
    "Portugal": [6, 2.8, 0.8, 4, 1, 0],
    "Czech Republic": [36, 1.8, 0.6, 4, 0, 1]
}

# Group stage matches
group_stage_matches = [
    ("2024-06-14", "Groupstage", "Germany", "Scotland", "UEFA Euro"),
    ("2024-06-15", "Groupstage", "Hungary", "Switzerland", "UEFA Euro"),
    ("2024-06-15", "Groupstage", "Spain", "Croatia", "UEFA Euro"),
    ("2024-06-15", "Groupstage", "Italy", "Albania", "UEFA Euro"),
    ("2024-06-16", "Groupstage", "Poland", "Netherlands", "UEFA Euro"),
    ("2024-06-16", "Groupstage", "Slovenia", "Denmark", "UEFA Euro"),
    ("2024-06-16", "Groupstage", "Serbia", "England", "UEFA Euro"),
    ("2024-06-17", "Groupstage", "Romania", "Ukraine", "UEFA Euro"),
    ("2024-06-17", "Groupstage", "Belgium", "Slovakia", "UEFA Euro"),
    ("2024-06-17", "Groupstage", "Austria", "France", "UEFA Euro"),
    ("2024-06-18", "Groupstage", "Turkey", "Georgia", "UEFA Euro"),
    ("2024-06-18", "Groupstage", "Portugal", "Czech Republic", "UEFA Euro"),
    ("2024-06-19", "Groupstage", "Croatia", "Albania", "UEFA Euro"),
    ("2024-06-19", "Groupstage", "Germany", "Hungary", "UEFA Euro"),
    ("2024-06-19", "Groupstage", "Scotland", "Switzerland", "UEFA Euro"),
    ("2024-06-20", "Groupstage", "Slovenia", "Serbia", "UEFA Euro"),
    ("2024-06-20", "Groupstage", "Denmark", "England", "UEFA Euro"),
    ("2024-06-20", "Groupstage", "Spain", "Italy", "UEFA Euro"),
    ("2024-06-21", "Groupstage", "Slovakia", "Ukraine", "UEFA Euro"),
    ("2024-06-21", "Groupstage", "Poland", "Austria", "UEFA Euro"),
    ("2024-06-21", "Groupstage", "Netherlands", "France", "UEFA Euro"),
    ("2024-06-22", "Groupstage", "Georgia", "Czech Republic", "UEFA Euro"),
    ("2024-06-22", "Groupstage", "Turkey", "Portugal", "UEFA Euro"),
    ("2024-06-22", "Groupstage", "Belgium", "Romania", "UEFA Euro"),
    ("2024-06-23", "Groupstage", "Switzerland", "Germany", "UEFA Euro"),
    ("2024-06-23", "Groupstage", "Scotland", "Hungary", "UEFA Euro"),
    ("2024-06-24", "Groupstage", "Albania", "Spain", "UEFA Euro"),
    ("2024-06-24", "Groupstage", "Croatia", "Italy", "UEFA Euro"),
    ("2024-06-25", "Groupstage", "France", "Poland", "UEFA Euro"),
    ("2024-06-25", "Groupstage", "Netherlands", "Austria", "UEFA Euro"),
    ("2024-06-25", "Groupstage", "England", "Slovenia", "UEFA Euro"),
    ("2024-06-25", "Groupstage", "Denmark", "Serbia", "UEFA Euro"),
    ("2024-06-26", "Groupstage", "Ukraine", "Belgium", "UEFA Euro"),
    ("2024-06-26", "Groupstage", "Slovakia", "Romania", "UEFA Euro"),
    ("2024-06-26", "Groupstage", "Czech Republic", "Turkey", "UEFA Euro"),
    ("2024-06-26", "Groupstage", "Georgia", "Portugal", "UEFA Euro"),
]

# Create DataFrame for group stage matches
group_stage_df = pd.DataFrame(group_stage_matches, columns=["Date", "Stage", "Home Team", "Away Team", "Tournament"])

# Load the Poisson regression models
poisson_home = joblib.load('models/poisson_home.pkl')
poisson_away = joblib.load('models/poisson_away.pkl')

# Load the label encoders
label_encoders = joblib.load('models/label_encoders.pkl')

def create_prediction_features(home_ranking, away_ranking, home_avg_goals_scored, away_avg_goals_scored, home_avg_goals_conceded, away_avg_goals_conceded, neutral):
    data = {
        'home_ranking': [home_ranking],
        'away_ranking': [away_ranking],
        'home_avg_goals_scored': [home_avg_goals_scored],
        'away_avg_goals_scored': [away_avg_goals_scored],
        'home_avg_goals_conceded': [home_avg_goals_conceded],
        'away_avg_goals_conceded': [away_avg_goals_conceded],
        'neutral': [neutral]
    }
    return pd.DataFrame(data)

def poisson_probability(lmbda, k):
    """Calculate the Poisson probability of k given lambda."""
    return (lmbda**k * np.exp(-lmbda)) / math.factorial(k)

def match_prediction(home_team, away_team):
    home_ranking, home_avg_goals_scored, home_avg_goals_conceded, _, _, _ = team_stats[home_team]
    away_ranking, away_avg_goals_scored, away_avg_goals_conceded, _, _, _ = team_stats[away_team]

    features = create_prediction_features(home_ranking, away_ranking, home_avg_goals_scored, away_avg_goals_scored, home_avg_goals_conceded, away_avg_goals_conceded, 1)

    home_goals_avg = poisson_home.predict(features)[0]
    away_goals_avg = poisson_away.predict(features)[0]

    max_goals = 10
    home_goals_prob = [poisson_probability(home_goals_avg, i) for i in range(max_goals)]
    away_goals_prob = [poisson_probability(away_goals_avg, i) for i in range(max_goals)]

    probability_matrix = np.outer(home_goals_prob, away_goals_prob)
    
    home_win_prob = np.sum(np.tril(probability_matrix, -1))
    draw_prob = np.sum(np.diag(probability_matrix))
    away_win_prob = np.sum(np.triu(probability_matrix, 1))

    return home_win_prob, draw_prob, away_win_prob, home_goals_avg, away_goals_avg

# Calculate the predicted outcomes for each group stage match
predictions = []
for index, row in group_stage_df.iterrows():
    home_team = row["Home Team"]
    away_team = row["Away Team"]
    home_win_prob, draw_prob, away_win_prob, home_goals_avg, away_goals_avg = match_prediction(home_team, away_team)
    predictions.append([home_team, away_team, home_win_prob, draw_prob, away_win_prob, home_goals_avg, away_goals_avg])

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(predictions, columns=["Home Team", "Away Team", "Home Win Probability", "Draw Probability", "Away Win Probability", "Home Goals Average", "Away Goals Average"])

# Function to simulate a match based on predictions
def simulate_match(home_win_prob, draw_prob, away_win_prob):
    outcome = random.choices(["Home Win", "Draw", "Away Win"], [home_win_prob, draw_prob, away_win_prob])[0]
    return outcome

# Simulate group stage matches and calculate points
group_points = {team: 0 for group in teams.values() for team in group}
group_stats = {team: {"GF": 0, "GA": 0} for group in teams.values() for team in group}

for index, row in predictions_df.iterrows():
    home_team = row["Home Team"]
    away_team = row["Away Team"]
    home_win_prob = row["Home Win Probability"]
    draw_prob = row["Draw Probability"]
    away_win_prob = row["Away Win Probability"]

    outcome = simulate_match(home_win_prob, draw_prob, away_win_prob)

    # Update points
    if outcome == "Home Win":
        group_points[home_team] += 3
    elif outcome == "Draw":
        group_points[home_team] += 1
        group_points[away_team] += 1
    elif outcome == "Away Win":
        group_points[away_team] += 3

    # Update goal statistics
    home_goals_avg = row["Home Goals Average"]
    away_goals_avg = row["Away Goals Average"]
    group_stats[home_team]["GF"] += home_goals_avg
    group_stats[home_team]["GA"] += away_goals_avg
    group_stats[away_team]["GF"] += away_goals_avg
    group_stats[away_team]["GA"] += home_goals_avg

# Create a DataFrame for group points and sort teams within each group
group_points_df = pd.DataFrame(list(group_points.items()), columns=["Team", "Points"])
group_points_df["Group"] = group_points_df["Team"].apply(lambda team: [group for group, teams_list in teams.items() if team in teams_list][0])
group_points_df["Goal Difference"] = group_points_df["Team"].apply(lambda team: group_stats[team]["GF"] - group_stats[team]["GA"])
group_points_df = group_points_df.sort_values(by=["Group", "Points", "Goal Difference"], ascending=[True, False, False]).reset_index(drop=True)

# Determine top two teams from each group and the best four third-placed teams
top_teams = group_points_df.groupby("Group").head(2).reset_index(drop=True)

third_placed_teams = group_points_df.groupby("Group").apply(lambda x: x.nlargest(1, "Points").iloc[2]).reset_index(drop=True)
best_four_third_placed_teams = third_placed_teams.nlargest(4, "Points").reset_index(drop=True)

# Combine top teams and best third-placed teams for the round of 16
round_of_16_teams = pd.concat([top_teams, best_four_third_placed_teams], ignore_index=True)

# Create pairs for the round of 16 matches
round_of_16_matches = [
    (round_of_16_teams.iloc[0]["Team"], round_of_16_teams.iloc[15]["Team"]),
    (round_of_16_teams.iloc[1]["Team"], round_of_16_teams.iloc[14]["Team"]),
    (round_of_16_teams.iloc[2]["Team"], round_of_16_teams.iloc[13]["Team"]),
    (round_of_16_teams.iloc[3]["Team"], round_of_16_teams.iloc[12]["Team"]),
    (round_of_16_teams.iloc[4]["Team"], round_of_16_teams.iloc[11]["Team"]),
    (round_of_16_teams.iloc[5]["Team"], round_of_16_teams.iloc[10]["Team"]),
    (round_of_16_teams.iloc[6]["Team"], round_of_16_teams.iloc[9]["Team"]),
    (round_of_16_teams.iloc[7]["Team"], round_of_16_teams.iloc[8]["Team"])
]

# Simulate knockout rounds
def simulate_knockout_round(matches):
    winners = []
    for match in matches:
        home_team, away_team = match
        home_win_prob, draw_prob, away_win_prob, _, _ = match_prediction(home_team, away_team)
        outcome = simulate_match(home_win_prob, draw_prob, away_win_prob)
        if outcome == "Home Win":
            winners.append(home_team)
        else:
            winners.append(away_team)
    return winners

# Simulate Round of 16
quarter_final_teams = simulate_knockout_round(round_of_16_matches)

# Simulate Quarter-Finals
quarter_final_matches = [
    (quarter_final_teams[0], quarter_final_teams[7]),
    (quarter_final_teams[1], quarter_final_teams[6]),
    (quarter_final_teams[2], quarter_final_teams[5]),
    (quarter_final_teams[3], quarter_final_teams[4])
]

semi_final_teams = simulate_knockout_round(quarter_final_matches)

# Simulate Semi-Finals
semi_final_matches = [
    (semi_final_teams[0], semi_final_teams[3]),
    (semi_final_teams[1], semi_final_teams[2])
]

final_teams = simulate_knockout_round(semi_final_matches)

# Simulate Final
winner = simulate_knockout_round([final_teams])[0]

print(f"The predicted winner of UEFA Euro 2024 is {winner}")
