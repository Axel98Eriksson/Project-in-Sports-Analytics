import math
import pandas as pd
import numpy as np
import random
import joblib
from oskars_poisson import simulate_match as oskar_predictor


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
    "Germany" : [16, 3.4, 0.4, 4, 0, 1], 
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

# Create DataFrame for group stage matches
group_stage_df = pd.DataFrame(group_stage_matches, columns=["Date", "Stage", "Home Team", "Away Team", "Tournament"])

matches_odds = []

number_of_iterations = 100
winners = []

def predict_match_result(home_team, away_team):

    for team, stats in team_stats.items():
        if team == home_team:
            home_ranking,home_avg_goals_scored, home_avg_goals_conceded, home_win , home_loss ,home_draw = stats
        elif team == away_team:
            away_ranking, away_avg_goals_scored, away_avg_goals_conceded, away_win , away_loss , away_draw = stats

    print(home_team, " - ", away_team)

    home_win_prob, draw_prob, away_win_prob = oskar_predictor(home_team, away_team, "Germany", home_ranking, away_ranking, home_avg_goals_scored, away_avg_goals_scored, home_win, away_win, home_loss, away_loss, home_draw, away_draw, n_simulations=10000)

    matches_odds.append((home_team, away_team, home_win_prob, draw_prob, away_win_prob))

    #random winner based on the probabilities
    result = np.random.choice(["Home Win", "Draw", "Away Win"], p=[home_win_prob, draw_prob, away_win_prob])
    

    if result == 'Home Win':
        return (home_team, 3), (away_team, 0), home_team
    elif result == 'Away Win':
        return (home_team, 0), (away_team, 3), away_team
    else:
        return (home_team, 1), (away_team, 1), None

# Simulate the group stage matches
for i in range(number_of_iterations):

    results = []
    for index, match in group_stage_df.iterrows():
        result = predict_match_result(match["Home Team"], match["Away Team"])
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

    # Function to simulate knockout match
    def simulate_knockout_match(home_team, away_team):
        result = predict_match_result(home_team, away_team)
        winner = result[2] if result[2] is not None else random.choice([home_team, away_team])
        return winner




    # Simulate knockout stage matches
    round_of_16_winners = []
    for match in knockout_stage_matches:
        winner = simulate_knockout_match(match[2], match[3])
        round_of_16_winners.append(winner)

    # Define quarter-finals
    quarter_final_matches = [
        ("2024-07-05", "Quarter-finals", round_of_16_winners[0], round_of_16_winners[1], "UEFA Euro"),
        ("2024-07-05", "Quarter-finals", round_of_16_winners[2], round_of_16_winners[3], "UEFA Euro"),
        ("2024-07-06", "Quarter-finals", round_of_16_winners[4], round_of_16_winners[5], "UEFA Euro"),
        ("2024-07-06", "Quarter-finals", round_of_16_winners[6], round_of_16_winners[7], "UEFA Euro")
    ]

    # Simulate Quarter-finals
    quarter_final_winners = []
    for match in quarter_final_matches:
        winner = simulate_knockout_match(match[2], match[3])
        quarter_final_winners.append(winner)

    # Define semi-finals
    semi_final_matches = [
        ("2024-07-09", "Semi-finals", quarter_final_winners[0], quarter_final_winners[1], "UEFA Euro"),
        ("2024-07-10", "Semi-finals", quarter_final_winners[2], quarter_final_winners[3], "UEFA Euro")
    ]

    # Simulate Semi-finals
    semi_final_winners = []
    for match in semi_final_matches:
        winner = simulate_knockout_match(match[2], match[3])
        semi_final_winners.append(winner)

    # Define Final
    final_match = ("2024-07-14", "Final", semi_final_winners[0], semi_final_winners[1], "UEFA Euro")

    # Combine all matches into a DataFrame
    all_matches = group_stage_df.values.tolist() + knockout_stage_matches + quarter_final_matches + semi_final_matches + [final_match]
    all_matches_df = pd.DataFrame(all_matches, columns=["Date", "Stage", "Home Team", "Away Team", "Tournament"])

    # Add results and winners
    results_with_winners = results + [(match[2], match[3], simulate_knockout_match(match[2], match[3])) for match in knockout_stage_matches] \
                        + [(match[2], match[3], simulate_knockout_match(match[2], match[3])) for match in quarter_final_matches] \
                        + [(match[2], match[3], simulate_knockout_match(match[2], match[3])) for match in semi_final_matches] \
                        + [(final_match[2], final_match[3], simulate_knockout_match(final_match[2], final_match[3]))]

    all_matches_df["Winner"] = [result[2] for result in results_with_winners]
    winners.append(final_match[2])

# Calculate the probability of each team winning the tournament
winner_probabilities = [(team, winners.count(team) / number_of_iterations) for team in set(winners)]
# Sort the probabilities in descending order
winner_probabilities = sorted(winner_probabilities, key=lambda x: x[1], reverse=True)


# Print the probabilities of each team winning the tournament
print("Winner Probabilities:")
for team, probability in winner_probabilities:
    print(f"{team}: {probability:.2f}")





def output():
    #print(all_matches_df)

    # visualize the results for the group stage with matches and winners
    print("\nGroup Stage Results:")
    for group, teams in group_standings.items():
        print(f"Group {group} Standings:")
        for i, (team, points) in enumerate(teams, 1):
            print(f"{i}. {team}: {points} points")
            # Print matches
            for match in group_stage_matches:
                if match[2] == team or match[3] == team:
                    # Find the played matches and odds
                    for odds in matches_odds:
                        if odds[0] == match[2] and odds[1] == match[3]:
                            print(f"{match[2]} - {match[3]}: {odds[2]:.2f} - {odds[3]:.2f} - {odds[4]:.2f}")
        print()


    # Print knockout stage matches
    print("\nKnockout Stage Matches:")
    for match in knockout_stage_matches:
        print(f"{match[1]} - {match[2]} vs {match[3]}")
        print(f"Winner: {round_of_16_winners[knockout_stage_matches.index(match)]}")

    # Print Quarter-finals
    print("\nQuarter-finals:")
    for match in quarter_final_matches:
        print(f"{match[1]} - {match[2]} vs {match[3]}")
        print(f"Winner: {quarter_final_winners[quarter_final_matches.index(match)]}")

    # Print Semi-finals
    print("\nSemi-finals:")
    for match in semi_final_matches:
        print(f"{match[1]} - {match[2]} vs {match[3]}")
        print(f"Winner: {semi_final_winners[semi_final_matches.index(match)]}")


    # Print Final
    print("\nFinal:")
    print(f"{final_match[1]} - {final_match[2]} vs {final_match[3]}")

    # Print Winner
    print(f"\nWinner: {final_match[2]}")

    # Save to CSV
    all_matches_df.to_csv("UEFA_Euro_2024_Simulation.csv", index=False)


#output()



