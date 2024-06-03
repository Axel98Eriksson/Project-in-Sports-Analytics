import numpy as np
import pandas as pd

# Load data
data_file = 'Results/results_with_rankings_wo_conference.csv'
data = pd.read_csv(data_file)

# Sort data by date to ensure the correct chronological order
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date')

# Initialize columns for average goals
data['home_avg_goals_scored'] = np.nan
data['home_avg_goals_conceded'] = np.nan
data['away_avg_goals_scored'] = np.nan
data['away_avg_goals_conceded'] = np.nan

# Function to calculate the average goals from the last 5 games
def calculate_average_goals(df, team, is_home, last_n=5):
    if is_home:
        games = df[(df['home_team'] == team)]
        goals_scored = games['home_score']
        goals_conceded = games['away_score']
    else:
        games = df[(df['away_team'] == team)]
        goals_scored = games['away_score']
        goals_conceded = games['home_score']
    
    # Only take the last `last_n` games
    games = games.tail(last_n)
    avg_goals_scored = goals_scored.tail(last_n).mean()
    avg_goals_conceded = goals_conceded.tail(last_n).mean()
    
    return avg_goals_scored, avg_goals_conceded

# Iterate through the data to compute averages
for idx, row in data.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    
    # Calculate averages for home team
    home_avg_scored, home_avg_conceded = calculate_average_goals(data.loc[:idx], home_team, is_home=True)
    data.at[idx, 'home_avg_goals_scored'] = home_avg_scored
    data.at[idx, 'home_avg_goals_conceded'] = home_avg_conceded
    
    # Calculate averages for away team
    away_avg_scored, away_avg_conceded = calculate_average_goals(data.loc[:idx], away_team, is_home=False)
    data.at[idx, 'away_avg_goals_scored'] = away_avg_scored
    data.at[idx, 'away_avg_goals_conceded'] = away_avg_conceded

# Fill any remaining NaN values with 0 (or another appropriate value if desired)
data.fillna(0, inplace=True)

# Save the updated DataFrame to a new CSV file
data.to_csv('Results/results_with_averages.csv', index=False)

print(data.head(10))
