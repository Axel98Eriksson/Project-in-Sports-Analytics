import csv
from collections import defaultdict

def combine_results(files):
    # Initialize a defaultdict to store aggregated results
    aggregated_results = defaultdict(lambda: [0]*7)  # 7 columns for the different stages

    # Iterate through each file
    for file in files:
        with open(file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header if present
            for row in reader:
                team = row[0]
                results = list(map(int, row[1:]))  # Convert results to integers
                # Add results to aggregated_results
                aggregated_results[team] = [agg + res for agg, res in zip(aggregated_results[team], results)]

    # Write the aggregated results to a new file
    with open('Oskars_predictor/combined_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Team', 'Group Stage Exits', 'Round of 16 Exits', 
                         'Quarterfinal Exits', 'Semifinal Exits', 'Runner-Up', 'Winner'])
        for team, results in aggregated_results.items():
            writer.writerow([team] + results)

if __name__ == "__main__":
    # List of input files to be combined
    files = ['Oskars_predictor/full_simulation.csv', 'Oskars_predictor/full_simulation2.csv', 'Oskars_predictor/full_simulation3.csv']  # Replace with your actual file names
    
    # Call the function to combine results
    combine_results(files)
