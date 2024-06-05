import pandas as pd
import matplotlib.pyplot as plt


def viz_win_percentage(file_path):
    # Calculate the percentage of each team winning the tournament
    df = pd.read_csv(file_path)
    df['Win Percentage'] = (df['Winner'] / 10000) * 100

    # Sort the dataframe by win percentage in descending order
    df = df.sort_values(by='Win Percentage', ascending=False)

    # Create a bar chart for the win percentage
    plt.figure(figsize=(12, 8))
    plt.bar(df['Team'], df['Win Percentage'], color='skyblue')
    plt.ylabel('Win Percentage')
    plt.xlabel('Team')
    plt.title('Percentage of Each Team Winning the Tournament')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.show()


def viz_most_likely_outcome(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Define color codes for each stage
    color_codes = {
        "Group Stage Exits": "#000000",  # Black
        "Round of 16 Exits": "#404040",  # Dark Gray
        "Quarterfinal Exits": "#808080", # Gray
        "Semifinal Exits": "#BFBFBF",    # Light Gray
        "Runner-Up": "#DFDFDF",          # Very Light Gray
        "Winner": "#FFD700"              #Gold
    }
    
    # Normalize the values to get percentages
    stages = ["Group Stage Exits", "Round of 16 Exits", "Quarterfinal Exits", "Semifinal Exits", "Runner-Up", "Winner"]
    df_percentages = df.copy()
    df_percentages[stages] = df_percentages[stages].div(df_percentages[stages].sum(axis=1), axis=0) * 100
    
    # Create a stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 10))
    
    bottom = [0] * len(df)
    
    for stage in stages:
        values = df_percentages[stage]
        ax.bar(df['Team'], values, bottom=bottom, color=color_codes[stage], label=stage)
        bottom += values
    
    plt.xlabel('Team')
    plt.ylabel('Percentage')
    plt.title('Distribution of How Far Each Team Reaches in the Tournament')
    plt.xticks(rotation=90)
    plt.legend(title='Stage')
    plt.show()


file_path = '10000_simulations.csv'

viz_win_percentage(file_path)
viz_most_likely_outcome(file_path)


