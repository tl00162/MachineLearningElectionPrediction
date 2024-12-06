import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Initialize progress bar
progress_bar = tqdm(total=100, desc="Random Forest Pipeline", unit="step")

# Load the cleaned dataset
polls_data_cleaned = pd.read_csv('cleaned_condensed_polls_final.csv')
progress_bar.update(10)  # Step 1: Data loaded

# Drop non-informative columns and the target-like 'classification' column
columns_to_drop = ['target', 'classification_numeric', 'classification']
X = polls_data_cleaned.drop(columns=columns_to_drop, errors='ignore')

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)
progress_bar.update(10)  # Step 2: Data preprocessed

# Extract target variable
y = polls_data_cleaned['classification']

# Split data into training (with classifications) and test (without classifications)
train_rows = ~y.isna()
test_rows = y.isna()

X_train = X[train_rows]
y_train = y[train_rows]
X_test = X[test_rows]
progress_bar.update(10)  # Step 3: Data split

# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model on the training data
print("Training the Random Forest model...")
model.fit(X_train, y_train)
progress_bar.update(30)  # Step 4: Model trained

# Make predictions on the test data
print("Making predictions on the test data...")
test_predictions = model.predict(X_test)
progress_bar.update(10)  # Step 5: Test predictions made

# Add predictions to the original dataset
polls_data_cleaned.loc[test_rows, 'predicted'] = test_predictions

# Save test predictions
test_output_path = 'test_predictions_random_forest.csv'
polls_data_cleaned[test_rows][['state', 'predicted']].to_csv(test_output_path, index=False)
print(f"Test predictions saved to {test_output_path}")
progress_bar.update(10)  # Step 6: Test predictions saved

# Save training predictions for evaluation
train_predictions = model.predict(X_train)
polls_data_cleaned.loc[train_rows, 'predicted'] = train_predictions

# Evaluate the model on the training data
print("Evaluating the model on the training data...")
print("Training Accuracy:", accuracy_score(y_train, train_predictions))
print("\nClassification Report (Training):\n", classification_report(y_train, train_predictions))
progress_bar.update(10)  # Step 7: Model evaluated

# Aggregate state-level predictions for test data
test_state_predictions = (
    polls_data_cleaned[test_rows]
    .groupby('state')
    .agg(
        final_prediction=('predicted', lambda x: x.value_counts().idxmax())  # Most frequent prediction
    )
    .reset_index()
)

# Save state-level predictions for the test data
test_state_output_path = 'test_state_level_predictions_random_forest_state_included.csv'
test_state_predictions.to_csv(test_state_output_path, index=False)
print(f"Test state-level predictions saved to {test_state_output_path}")
progress_bar.update(10)  # Step 8: State-level predictions saved

# Save the full dataset with predictions
full_output_path = 'full_predictions_random_forest_state_included.csv'
polls_data_cleaned.to_csv(full_output_path, index=False)
print(f"Full dataset with predictions saved to {full_output_path}")
progress_bar.update(10)  # Step 9: Full dataset saved

# Calculate feature importances
importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Print the top features
print("\nTop Features:\n", feature_importances.head(10))

# Plot the top features
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'][:10], feature_importances['Importance'][:10])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Features in Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
progress_bar.update(10)  # Step 10: Feature importance plotted

# Close the progress bar
progress_bar.close()
