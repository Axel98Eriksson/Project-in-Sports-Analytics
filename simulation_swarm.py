import numpy as np

# Define the teams
teams = ['Germany', 'Scotland', 'Hungary', 'Switzerland',
            'Spain', 'Croatia', 'Italy', 'Albania',
            'Slovenia', 'Denmark', 'Serbia', 'England',
            'Poland', 'Netherlands', 'Austria', 'France',
            'Romania', 'Ukraine', 'Belgium', 'Slovakia',
            'Turkey', 'Georgia', 'Portugal', 'Czech Republic']

# Initialize a matrix to store the results
num_teams = len(teams)
num_stages = 6  # Group stage, Round of 16, Quarterfinals, Semifinals, Second place, First place
results_matrix = np.zeros((num_teams, num_stages))


num_simulations = 1000
for _ in range(num_simulations):
    # Simulate the tournament and determine the outcome for each team
    # For simplicity, let's assume random outcomes for each team in each simulation
    # Update the results matrix accordingly
    # Example:
    # results_matrix[0, 3] += 1  # Team A reached semifinals in this simulation
    results_matrix[0,0] += 1

# After all simulations, you can analyze the results
# For example, you can calculate the average number of times each team reached each stage
average_results = np.mean(results_matrix, axis=0)

# You can also analyze the distribution of outcomes for each team
# For example, you can calculate the percentage of simulations in which each team reached each stage
percentage_results = (results_matrix / num_simulations) * 100

# Print or visualize the results as needed
print("Average Results:")
print(average_results)
print("\nPercentage of Simulations:")
print(percentage_results)
