import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score

# Load the cleaned dataset
polls_data_cleaned = pd.read_csv('cleaned_condensed_polls_final.csv')

# Drop non-informative columns and the target-like 'classification' column
columns_to_drop = ['target', 'classification_numeric', 'classification']
X = polls_data_cleaned.drop(columns=columns_to_drop, errors='ignore')

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Extract target variable
y = polls_data_cleaned['classification']

# Drop rows with missing target values
valid_rows = ~y.isna()
X = X[valid_rows]
y = y[valid_rows]

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Fit the Random Forest model
model.fit(X, y)

# Implement Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
predictions = cross_val_predict(model, X, y, cv=skf)

# Output predictions with corresponding states
polls_data_cleaned = polls_data_cleaned[valid_rows]
polls_data_cleaned['predicted'] = predictions

# Evaluate the model
print("Accuracy:", accuracy_score(y, predictions))
print("\nClassification Report:\n", classification_report(y, predictions))

# Aggregate predictions to make a final prediction for each state
state_predictions = (
    polls_data_cleaned.groupby('state')
    .agg(
        classification=('classification', 'first'),  # Use the first value as the actual target (assuming it's consistent across rows)
        final_prediction=('predicted', lambda x: x.value_counts().idxmax())  # Most frequent prediction
    )
    .reset_index()
)

# Save the aggregated state-level predictions
state_output_path = 'state_level_predictions_with_classification.csv'
state_predictions.to_csv(state_output_path, index=False)

print(f"State-level predictions with classification saved to {state_output_path}")

# Save individual predictions with corresponding states
row_output_path = 'state_predictions_no_classification.csv'
polls_data_cleaned[['state', 'classification', 'predicted']].to_csv(row_output_path, index=False)

print(f"Row-level predictions saved to {row_output_path}")

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
