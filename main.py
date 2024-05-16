import pandas as pd

# Path to the CSV file
file_path = "euro_games/Uefa Euro Cup All Matches.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Filter matches for a specific year
desired_year = 2016  # Change this to the desired year
matches_in_year = df[df['Year'] == desired_year]

# Display the filtered matches
print(matches_in_year)
