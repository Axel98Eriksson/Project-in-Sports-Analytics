import csv
import pandas
csvFile = pandas.read_csv('all_games/results.csv')
#filter out rows with missing values
csvFile = csvFile.dropna()
#filter to only include rows with tournament name "UEFA Euro"
csvFile = csvFile[(csvFile['tournament'] == 'UEFA Euro') | (csvFile['tournament'] == 'UEFA Euro Qualifying') | (csvFile['tournament'] == 'UEFA Nations League')]
#filter to only include rows with date after 1980-01-01
csvFile = csvFile[csvFile['date'] > '1992-01-01']
csvFile['date'] = pandas.to_datetime(csvFile['date'])

rankings = pandas.read_csv('fifa_rankings/fifa_ranking-2024-04-04.csv')
rankings = rankings.dropna()
rankings = rankings[rankings['confederation'] == 'UEFA']
rankings['rank_date'] = pandas.to_datetime(rankings['rank_date'])

# add a column to the csvFile that contains the fifa ranking of the home team
csvFile['home_ranking'] = 0
csvFile['away_ranking'] = 0



for index, row in csvFile.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    
    date = str(row['date'])
    #show only year
    #date = date[:4]
    

    home_ranking = rankings[(str(rankings['rank_date'][:4]) == date[:4]) & (rankings['country_full'] == home_team)]['rank'].rank
    away_ranking = rankings[(str(rankings['rank_date'][:4]) == date[:4]) & (rankings['country_full'] == away_team)]['rank'].rank

    # home_ranking = rankings[(rankings['rank_date'] == date) & (rankings['country_full'] == home_team)]['rank']
    # away_ranking = rankings[(rankings['rank_date'] == date) & (rankings['country_full'] == away_team)]['rank']
    

    print("home: ", home_ranking)
    print("away: ", away_ranking)
    csvFile.at[index, 'home_ranking'] = home_ranking
    csvFile.at[index, 'away_ranking'] = away_ranking






#print(csvFile)


# with open('data/1872-2024/results.csv', mode ='r')as file:
#   csvFile = csv.reader(file)
#   for lines in csvFile:
#         print(lines)