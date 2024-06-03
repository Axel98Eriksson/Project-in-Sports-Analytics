import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = 'Results/results_with_rankings_wo_conference.csv'
data = pd.read_csv(file_path)

# Convert date to datetime and extract additional features
data['match_date'] = pd.to_datetime(data['date'])
data['day_of_week'] = data['match_date'].dt.dayofweek
data['month'] = data['match_date'].dt.month

# Define the features and target variables
features = data[['home_ranking', 'away_ranking', 'neutral', 'day_of_week', 'month']]
target_home = data['home_score']
target_away = data['away_score']

# One-hot encode categorical features
categorical_features = ['neutral']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

# Split the data into training and testing sets
X_train, X_test, y_train_home, y_test_home, y_train_away, y_test_away = train_test_split(
    features, target_home, target_away, test_size=0.2, random_state=42
)

from sklearn.feature_selection import SequentialFeatureSelector

# Ensure that the input to the SequentialFeatureSelector is a DataFrame
X_train_df = pd.DataFrame(X_train, columns=features.columns)
X_test_df = pd.DataFrame(X_test, columns=features.columns)

# Create a pipeline that includes preprocessing and the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', PoissonRegressor())
])

# Perform sequential feature selection
sfs = SequentialFeatureSelector(model_pipeline, n_features_to_select='auto', direction='forward', scoring='neg_mean_squared_error', cv=5)

# Fit the selector for home_score
sfs.fit(X_train_df, y_train_home)

# Get the selected features
selected_features_home = sfs.get_support()
selected_features_names_home = features.columns[selected_features_home]

print("Selected features for home_score predictions:", selected_features_names_home)

# Transform the training and testing sets to include only the selected features
X_train_selected_home = sfs.transform(X_train_df)
X_test_selected_home = sfs.transform(X_test_df)

# Train the model on the selected features
model_pipeline.fit(X_train_selected_home, y_train_home)

# Make predictions
preds_home = model_pipeline.predict(X_test_selected_home)

# Calculate mean squared error
mse_home = mean_squared_error(y_test_home, preds_home)
print(f"Mean Squared Error for home_score predictions with selected features: {mse_home}")


# Perform sequential feature selection for away_score
sfs.fit(X_train_df, y_train_away)

# Get the selected features
selected_features_away = sfs.get_support()
selected_features_names_away = features.columns[selected_features_away]

print("Selected features for away_score predictions:", selected_features_names_away)

# Transform the training and testing sets to include only the selected features
X_train_selected_away = sfs.transform(X_train_df)
X_test_selected_away = sfs.transform(X_test_df)

# Train the model on the selected features
model_pipeline.fit(X_train_selected_away, y_train_away)

# Make predictions
preds_away = model_pipeline.predict(X_test_selected_away)

# Calculate mean squared error
mse_away = mean_squared_error(y_test_away, preds_away)
print(f"Mean Squared Error for away_score predictions with selected features: {mse_away}")

