import csv
import pandas
csvFile = pandas.read_csv('all_games/results.csv')
#filter out rows with missing values
csvFile = csvFile.dropna()

euro = csvFile[(csvFile['tournament'] == 'UEFA Euro')]

#filter to only include rows with tournament name "UEFA Euro"
csvFile = csvFile[(csvFile['tournament'] == 'Friendly')]

#filter to only include rows that have teams from UEFA
csvFile = csvFile[(csvFile['home_team'].isin(euro['home_team'])) & (csvFile['away_team'].isin(euro['away_team']))]

#filter to only include rows with date after 1980-01-01
csvFile = csvFile[csvFile['date'] > '2020-01-01']
csvFile['date'] = pandas.to_datetime(csvFile['date'])

rankings = pandas.read_csv('fifa_rankings/fifa_ranking-2024-04-04.csv')
rankings = rankings.dropna()
rankings = rankings[rankings['confederation'] == 'UEFA']
rankings['rank_date'] = pandas.to_datetime(rankings['rank_date'])

# add a column to the csvFile that contains the fifa ranking of the home team
csvFile['home_ranking'] = 0
csvFile['away_ranking'] = 0



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


print(csvFile)

#save the updated csvFile to a new file
csvFile.to_csv('Results/results_friendly.csv', index=False)
