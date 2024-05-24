import pandas as pd

# URL for the European Championship fixtures
euc = 'https://fbref.com/en/comps/676/schedule/European-Championship-Scores-and-Fixtures'

# Reading fixtures
df = pd.read_html(euc)[0]

# Drop rows with only NaN values
df = df.dropna(how='all')  # or alternatively, df.dropna(how='all', inplace=True)

# Displaying the combined data frame (optional)
print(df.head(36))

# To inspect the DataFrame structure and size
#print(df.info())
#print(df.shape)
