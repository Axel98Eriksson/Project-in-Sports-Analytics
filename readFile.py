import csv
import pandas
csvFile = pandas.read_csv('all_games/results.csv')
#filter out rows with missing values
csvFile = csvFile.dropna()

filter = csvFile[(csvFile['tournament'] == 'UEFA Euro') | (csvFile['tournament'] == 'Friendly') | (csvFile['tournament'] == 'FIFA World Cup') | (csvFile['tournament'] == 'FIFA World Cup qualification') | (csvFile['tournament'] == 'FIFA World Cup qualification (UEFA)')  | (csvFile['tournament'] == 'UEFA Euro qualification') | (csvFile['tournament'] == 'FIFA World Cup qualification (UEFA)') ]

euro = csvFile[(csvFile['tournament'] == 'UEFA Euro') | (csvFile['tournament'] == 'UEFA Euro qualification')]

#filter to only include rows that have the teams i want to predict
csvFile = csvFile[(csvFile['home_team'].isin(filter['home_team'])) & (csvFile['away_team'].isin(filter['away_team']))]

#Filter to only include teams from UEFA
csvFile = csvFile[(csvFile['home_team'].isin(euro['home_team'])) & (csvFile['away_team'].isin(euro['away_team']))]


#filter to only include rows with date after 1980-01-01
csvFile = csvFile[csvFile['date'] > '2010-01-01']
csvFile['date'] = pandas.to_datetime(csvFile['date'])

rankings = pandas.read_csv('fifa_rankings/fifa_ranking-2024-04-04.csv')
rankings = rankings.dropna()
rankings = rankings[rankings['confederation'] == 'UEFA']
rankings['rank_date'] = pandas.to_datetime(rankings['rank_date'])

# add a column to the csvFile that contains the fifa ranking of the home team
csvFile['home_ranking'] = 0
csvFile['away_ranking'] = 0

#add a column that contains the average goals scored the last 5 games, and how many wins, losses and draws the team has had in the last 5 games
csvFile['home_avg_goals_scored'] = 0
csvFile['away_avg_goals_scored'] = 0
csvFile['home_avg_goals_conceded'] = 0
csvFile['away_avg_goals_conceded'] = 0
csvFile['home_wins'] = 0
csvFile['away_wins'] = 0
csvFile['home_losses'] = 0
csvFile['away_losses'] = 0
csvFile['home_draws'] = 0
csvFile['away_draws'] = 0

def get_last_5_games(team, match_date):
    last_5_games = csvFile[((csvFile['home_team'] == team) | (csvFile['away_team'] == team)) & (csvFile['date'] < match_date)].tail(5)
    return last_5_games

def get_avg_goals(team, match_date):
    last_5_games = get_last_5_games(team, match_date)
    home_goals = 0
    away_goals = 0
    for index, row in last_5_games.iterrows():
        if row['home_team'] == team:
            home_goals += row['home_score']
            away_goals += row['away_score']
        else:
            home_goals += row['away_score']
            away_goals += row['home_score']
    return home_goals/5, away_goals/5

def get_wins_losses_draws(team, match_date):
    last_5_games = get_last_5_games(team, match_date)
    wins = 0
    losses = 0
    draws = 0
    for index, row in last_5_games.iterrows():
        if row['home_team'] == team:
            if row['home_score'] > row['away_score']:
                wins += 1
            elif row['home_score'] < row['away_score']:
                losses += 1
            else:
                draws += 1
        else:
            if row['home_score'] < row['away_score']:
                wins += 1
            elif row['home_score'] > row['away_score']:
                losses += 1
            else:
                draws += 1
    return wins, losses, draws



def get_team_ranking(team, match_date):
    team_rankings = rankings[rankings['country_full'] == team]
    closest_date = team_rankings[team_rankings['rank_date'] <= match_date]['rank_date'].max()
    if pandas.isna(closest_date):
        #return the first found rank for the team
        if len(team_rankings) > 0:
            return team_rankings['rank'].values[0]
        else:
            return 0
        
    return team_rankings[team_rankings['rank_date'] == closest_date]['rank'].values[0]




for index, row in csvFile.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    match_date = row['date']
    
    home_ranking = get_team_ranking(home_team, match_date)
    away_ranking = get_team_ranking(away_team, match_date)
    
    csvFile.at[index, 'home_ranking'] = home_ranking
    csvFile.at[index, 'away_ranking'] = away_ranking

    home_avg_goals_scored, home_average_conceded = get_avg_goals(home_team, match_date)
    csvFile.at[index, 'home_avg_goals_scored'] = home_avg_goals_scored
    csvFile.at[index, 'home_avg_goals_conceded'] = home_average_conceded

    away_avg_goals_scored, away_avg_goals_conceded = get_avg_goals(away_team, match_date)
    csvFile.at[index, 'away_avg_goals_scored'] = away_avg_goals_scored
    csvFile.at[index, 'away_avg_goals_conceded'] = away_avg_goals_conceded

    home_wins, home_losses, home_draws = get_wins_losses_draws(home_team, match_date)
    csvFile.at[index, 'home_wins'] = home_wins
    csvFile.at[index, 'home_losses'] = home_losses
    csvFile.at[index, 'home_draws'] = home_draws

    away_wins, away_losses, away_draws = get_wins_losses_draws(away_team, match_date)
    csvFile.at[index, 'away_wins'] = away_wins
    csvFile.at[index, 'away_losses'] = away_losses
    csvFile.at[index, 'away_draws'] = away_draws



print(csvFile)
csvFile = csvFile[csvFile['date'] > '2014-01-01']
#save the updated csvFile to a new file
csvFile.to_csv('Results/oskars_special_mix.csv', index=False)
