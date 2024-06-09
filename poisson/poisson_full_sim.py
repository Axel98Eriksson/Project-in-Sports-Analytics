import math
import pandas as pd
import numpy as np
import random
import joblib
import csv

# Define the groups and teams
teams = {
    "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
    "B": ["Spain", "Croatia", "Italy", "Albania"],
    "C": ["England", "Serbia", "Slovenia", "Denmark"],
    "D": ["Netherlands", "France", "Poland", "Austria"],
    "E": ["Ukraine", "Slovakia", "Belgium", "Romania"],
    "F": ["Turkey", "Georgia", "Portugal", "Czech Republic"]
}

#last five games (friendlies and qualification)
team_stats = {
    "Germany" : [16, 3.4, 0.4, 4, 0, 1], #fifa rank, avg goal scored, avg goal conceded, win ,loss ,draw
    "Scotland" : [39,  1.4, 2, 1, 2, 2],
    "Hungary" : [26, 2, 1, 3, 0, 2],
    "Switzerland" : [19 , 0.6, 0.6, 1, 1, 3],
    
    "Spain" : [ 8, 2, 1.4, 3, 1, 1],
    "Croatia" : [10 , 2, 0.4, 4, 1, 0],
    "Italy" : [9 , 2, 1.2, 3, 1, 1],
    "Albania" : [66 , 0.8, 1, 1, 2, 2],

    "Poland" : [28 , 1.8, 0.6, 2, 0, 3],
    "Netherlands" : [7 , 2.6, 0.4, 4, 1, 0],
    "Slovenia" : [57 , 1.6, 1, 3, 1, 1],
    "Denmark" : [21 , 1.2, 0.4, 3, 1, 1],
    
    "Serbia" : [33 , 2, 1.2, 3, 1, 1],
    "England" : [4 , 1.6, 0.8, 2, 1, 2],
    "Romania" : [46 , 2.0, 1.0, 3 , 1, 1],
    "Ukraine" : [22 , 1.4, 0.6, 3, 0, 2],

    "Belgium" : [3 , 3.0 , 0.8, 3 , 0, 2],
    "Slovakia" : [48 ,  1.6, 1.2, 3, 1, 1],
    "Austria" : [25 , 2.6 , 0.8, 4 , 1, 0],
    "France" : [2 , 4.2 , 1.4, 3 , 1, 1],
    
    "Turkey" : [40 , 1.4 , 1.6, 2 , 2, 1],
    "Georgia" : [75 ,  1.8, 1, 2 , 1, 2],
    "Portugal" : [6 , 2.8 , 0.8, 4 , 1, 0],
    "Czech Republic" : [36, 1.8 , 0.6, 4 , 0, 1]
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

#functions for keeping track of progress for each team.
team_list = [
    "Germany", "Scotland", "Hungary", "Switzerland",
    "Spain", "Croatia", "Italy", "Albania",
    "Poland", "Netherlands", "Slovenia", "Denmark",
    "Serbia", "England", "Romania", "Ukraine",
    "Belgium", "Slovakia", "Austria", "France",
    "Turkey", "Georgia", "Portugal", "Czech Republic"
]

fields = ["Team", "Group Stage Exits", "Round of 16 Exits", "Quarterfinal Exits", "Semifinal Exits", "Runner-Up", "Winner"]

# Create and initialize the CSV file
with open('full_simulation.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    for team in team_list:
        writer.writerow({"Team": team, "Group Stage Exits": 0, "Round of 16 Exits": 0, "Quarterfinal Exits": 0, "Semifinal Exits": 0, "Runner-Up": 0, "Winner": 0})

def update_team_progress(team_name, stage):
    # Read current data
    with open('full_simulation.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        teams = list(reader)
    
    # Update the team's progress
    for team in teams:
        if team['Team'] == team_name:
            if stage == 'Group Stage':
                team['Group Stage Exits'] = int(team['Group Stage Exits']) + 1
            elif stage == 'Round of 16':
                team['Round of 16 Exits'] = int(team['Round of 16 Exits']) + 1
            elif stage == 'Quarterfinal':
                team['Quarterfinal Exits'] = int(team['Quarterfinal Exits']) + 1
            elif stage == 'Semifinal':
                team['Semifinal Exits'] = int(team['Semifinal Exits']) + 1
            elif stage == 'Runner-Up':
                team['Runner-Up'] = int(team['Runner-Up']) + 1
            elif stage == 'Winner':
                team['Winner'] = int(team['Winner']) + 1
            break

    # Write the updated data back to the CSV file
    with open('full_simulation.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(teams)



# Create DataFrame for group stage matches
group_stage_df = pd.DataFrame(group_stage_matches, columns=["Date", "Stage", "Home Team", "Away Team", "Tournament"])

# Load the Poisson regression models
poisson_home = joblib.load('models/new_poisson_home.pkl')
poisson_away = joblib.load('models/new_poisson_away.pkl')

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

def match_outcome(home_win_prob, draw_prob, away_win_prob):
    outcomes = ['Home Win', 'Draw', 'Away Win']
    probabilities = [home_win_prob, draw_prob, away_win_prob]
    
    # Ensure probabilities sum to 1
    assert sum(probabilities) < 1.05 and sum(probabilities) > 0.95, "Probabilities must sum to 1"
    return random.choices(outcomes, weights=probabilities)[0]

# Function to predict match result using Poisson regression models
def predict_match_result(home_team, away_team, group_stage):
    for team, stats in team_stats.items():
        if team == home_team:
            home_ranking, home_avg_goals_scored, home_avg_goals_conceded, home_win , home_loss ,home_draw = stats
        elif team == away_team:
            away_ranking, away_avg_goals_scored, away_avg_goals_conceded, away_win , away_loss , away_draw = stats

    features = create_prediction_features(
        home_ranking=home_ranking, away_ranking=away_ranking,
        home_avg_goals_scored=home_avg_goals_scored, away_avg_goals_scored=away_avg_goals_scored, 
        home_avg_goals_conceded=home_avg_goals_conceded, away_avg_goals_conceded=away_avg_goals_conceded, 
        neutral=1
    )

    
    home_lambda = poisson_home.predict(features)[0]
    away_lambda = poisson_away.predict(features)[0]

    
    max_goals = 5
    goal_distribution_matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            home_goal_prob = poisson_probability(home_lambda, i)
            away_goal_prob = poisson_probability(away_lambda, j)
            goal_distribution_matrix[i, j] = home_goal_prob * away_goal_prob
    
    goal_distribution_matrix /= goal_distribution_matrix.sum()
    
    # Calculate probabilities
    home_win_prob = np.sum(np.tril(goal_distribution_matrix, -1))
    draw_prob = np.sum(np.diag(goal_distribution_matrix))
    away_win_prob = np.sum(np.triu(goal_distribution_matrix, 1))
    
    result = match_outcome(home_win_prob, draw_prob, away_win_prob)
    
    if group_stage is True:
        if result == 'Home Win':
            return (home_team, 3), (away_team, 0), home_team
        elif result == 'Away Win':
            return (home_team, 0), (away_team, 3), away_team
        else:
            return (home_team, 1), (away_team, 1), None
    else:
        if result == 'Home Win':
            return (home_team, 3), (away_team, 0), home_team
        elif result == 'Away Win':
            return (home_team, 0), (away_team, 3), away_team
        else:
            return (home_team, 1), (away_team, 1), random.choice([home_team , away_team])
#############################################################################

n_simulations = 10000

for n in range(n_simulations):
    if n_simulations % 100: print("Simulation ", n ,"\n")

    # Simulate the group stage matches
    results = []
    for index, match in group_stage_df.iterrows():
        result = predict_match_result(match["Home Team"], match["Away Team"],group_stage=True)
        results.append(result)

    # Process group stage results
    group_points = {group: {team: 0 for team in teams[group]} for group in teams}
    for result in results:
        for team, points in result[:2]:
            for group, group_teams in teams.items():
                if team in group_teams:
                    group_points[group][team] += points

    # Calculate group standings
    group_standings = {}
    for group in teams:
        sorted_teams = sorted(group_points[group].items(), key=lambda x: x[1], reverse=True)
        group_standings[group] = sorted_teams

    # Identify and rank third-placed teams
    third_placed_teams = []
    for group in group_standings:
        third_placed_teams.append(group_standings[group][2])

    # Sort third-placed teams by points, then by goal difference, then by goals scored
    third_placed_teams.sort(key=lambda x: (x[1]), reverse=True)

    # Select top 4 third-placed teams
    top_4_third_placed_teams = third_placed_teams[:4]

    # Determine knockout stage positions
    knockout_positions = {}
    for group in group_standings:
        knockout_positions[f"1{group}"] = group_standings[group][0][0]
        knockout_positions[f"2{group}"] = group_standings[group][1][0]

    # Define knockout stage matches with top 4 third-placed teams
    knockout_stage_matches = [
        ("2024-06-29", "Round of 16", knockout_positions["1B"], top_4_third_placed_teams[0][0], "UEFA Euro"),
        ("2024-06-29", "Round of 16", knockout_positions["1A"], knockout_positions["2C"], "UEFA Euro"),
        ("2024-06-30", "Round of 16", knockout_positions["1F"], top_4_third_placed_teams[1][0], "UEFA Euro"),
        ("2024-06-30", "Round of 16", knockout_positions["2D"], knockout_positions["2E"], "UEFA Euro"),
        ("2024-07-01", "Round of 16", knockout_positions["1E"], top_4_third_placed_teams[2][0], "UEFA Euro"),
        ("2024-07-01", "Round of 16", knockout_positions["1D"], knockout_positions["2F"], "UEFA Euro"),
        ("2024-07-02", "Round of 16", knockout_positions["1C"], top_4_third_placed_teams[3][0], "UEFA Euro"),
        ("2024-07-02", "Round of 16", knockout_positions["2A"], knockout_positions["2B"], "UEFA Euro")
    ]

    knockout_stage_teams = set()
    for match in knockout_stage_matches:
        knockout_stage_teams.add(match[2])
        knockout_stage_teams.add(match[3])

    # Update the group stage exits
    for team in team_list:
        if team not in knockout_stage_teams:
            update_team_progress(team, "Group Stage")
        


    # Function to simulate knockout match
    def simulate_knockout_match(home_team, away_team):
        result = predict_match_result(home_team, away_team,group_stage=False)
        winner = result[2] if result[2] is not None else random.choice([home_team, away_team])
        
        return winner

    # Simulate knockout stage matches with debug info
    round_of_16_winners = []
    print("Ro16 Matches")
    for match in knockout_stage_matches:
        print(match[2] , " - ", match[3])
        winner = simulate_knockout_match(match[2], match[3])
        round_of_16_winners.append(winner)
        loser = match[3] if winner == match[2] else match[2]
        update_team_progress(loser, match[1]) # Update the progress for the losing team
    print("Round of 16 winners: ", round_of_16_winners)

    # Define quarter-finals with debug info
    quarter_final_matches = [
        ("2024-07-05", "Quarter-finals", round_of_16_winners[0], round_of_16_winners[1], "UEFA Euro"),
        ("2024-07-05", "Quarter-finals", round_of_16_winners[2], round_of_16_winners[3], "UEFA Euro"),
        ("2024-07-06", "Quarter-finals", round_of_16_winners[4], round_of_16_winners[5], "UEFA Euro"),
        ("2024-07-06", "Quarter-finals", round_of_16_winners[6], round_of_16_winners[7], "UEFA Euro")
    ]

    quarter_final_winners = []
    for match in quarter_final_matches:
        winner = simulate_knockout_match(match[2], match[3])
        quarter_final_winners.append(winner)
        loser = match[3] if winner == match[2] else match[2]
        update_team_progress(loser, "Quarterfinal") # Update the progress for the losing team
    print("Quarter-final winners: ", quarter_final_winners)

    # Define semi-finals with debug info
    semi_final_matches = [
        ("2024-07-09", "Semi-finals", quarter_final_winners[0], quarter_final_winners[1], "UEFA Euro"),
        ("2024-07-10", "Semi-finals", quarter_final_winners[2], quarter_final_winners[3], "UEFA Euro")
    ]

    semi_final_winners = []
    for match in semi_final_matches:
        winner = simulate_knockout_match(match[2], match[3])
        semi_final_winners.append(winner)
        loser = match[3] if winner == match[2] else match[2]
        update_team_progress(loser, "Semifinal") # Update the progress for the losing team
    print("Semi-final winners: ", semi_final_winners)

    # Define Final with debug info
    final_match = ("2024-07-14", "Final", semi_final_winners[0], semi_final_winners[1], "UEFA Euro")
    final_winner = simulate_knockout_match(final_match[2], final_match[3])
    loser = match[3] if winner == match[2] else match[2]
    update_team_progress(loser, "Runner-Up") # Update the progress for the losing team
    update_team_progress(winner, "Winner") # Update the progress for the losing team
    print("Final winner: ", final_winner)

    # Combine all matches into a DataFrame
    #all_matches = group_stage_df.values.tolist() + knockout_stage_matches + quarter_final_matches + semi_final_matches + [final_match]
    #all_matches_df = pd.DataFrame(all_matches, columns=["Date", "Stage", "Home Team", "Away Team", "Tournament"])