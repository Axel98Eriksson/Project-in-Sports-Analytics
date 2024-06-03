import pandas as pd
from datetime import datetime

# Step 1: Define the Groups and Teams
teams = {
    "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
    "B": ["Spain", "Croatia", "Italy", "Albania"],
    "C": ["Poland", "Netherlands", "Slovenia", "Denmark"],
    "D": ["Serbia", "England", "Romania", "Ukraine"],
    "E": ["Belgium", "Slovakia", "Austria", "France"],
    "F": ["Türkiye", "Georgia", "Portugal", "Czechia"]
}

# Step 2: Input Group Stage Matches
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
    ("2024-06-18", "Türkiye", "Georgia", "UEFA Euro"),
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
    ("2024-06-22", "Türkiye", "Portugal", "UEFA Euro"),
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
    ("2024-06-26", "Czechia", "Türkiye", "UEFA Euro"),
    ("2024-06-26", "Georgia", "Portugal", "UEFA Euro")
]

# Convert the group stage matches to a DataFrame
group_stage_df = pd.DataFrame(group_stage_matches, columns=["Date", "Home Team", "Away Team", "Tournament"])

# Step 3: Define Knockout Stage Structure
# Placeholder for knockout stage matches
knockout_stage_matches = []

# Function to add knockout stage matches
def add_knockout_match(date, stage, home_team, away_team):
    knockout_stage_matches.append((date, stage, home_team, away_team, "UEFA Euro"))

# Round of 16
add_knockout_match("2024-06-29", "Round of 16", "1A", "2B")
add_knockout_match("2024-06-29", "Round of 16", "1C", "2D")
add_knockout_match("2024-06-30", "Round of 16", "1B", "2A")
add_knockout_match("2024-06-30", "Round of 16", "1D", "2C")
add_knockout_match("2024-07-01", "Round of 16", "1E", "2F")
add_knockout_match("2024-07-01", "Round of 16", "1G", "2H")
add_knockout_match("2024-07-02", "Round of 16", "1F", "2E")
add_knockout_match("2024-07-02", "Round of 16", "1H", "2G")

# Quarter-finals
add_knockout_match("2024-07-05", "Quarter-finals", "W49", "W50")
add_knockout_match("2024-07-05", "Quarter-finals", "W51", "W52")
add_knockout_match("2024-07-06", "Quarter-finals", "W53", "W54")
add_knockout_match("2024-07-06", "Quarter-finals", "W55", "W56")

# Semi-finals
add_knockout_match("2024-07-09", "Semi-finals", "W57", "W58")
add_knockout_match("2024-07-10", "Semi-finals", "W59", "W60")

# Third place match
add_knockout_match("2024-07-13", "Third place", "L61", "L62")

# Final
add_knockout_match("2024-07-14", "Final", "W61", "W62")

# Convert the knockout stage matches to a DataFrame
knockout_stage_df = pd.DataFrame(knockout_stage_matches, columns=["Date", "Stage", "Home Team", "Away Team", "Tournament"])

# Combine group stage and knockout stage dataframes
all_matches_df = pd.concat([group_stage_df, knockout_stage_df], ignore_index=True)

# Output the DataFrame to CSV
all_matches_df.to_csv("UEFA_Euro_2024_Matches.csv", index=False)

print("File created successfully: UEFA_Euro_2024_Matches.csv")
