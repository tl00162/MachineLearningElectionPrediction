import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Initialize progress bar
progress_bar = tqdm(total=100, desc="Logistic Regression Pipeline", unit="step")

# Load the cleaned dataset
polls_data_cleaned = pd.read_csv('cleaned_condensed_polls_final.csv')
progress_bar.update(10)  # Progress update

# Drop non-informative columns and the target-like 'classification' column
columns_to_drop = ['target', 'classification_numeric', 'classification']
X = polls_data_cleaned.drop(columns=columns_to_drop, errors='ignore')

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)
progress_bar.update(10)  # Progress update

# Extract target variable
y = polls_data_cleaned['classification']

# Drop rows with missing target values
valid_rows = ~y.isna()
X = X[valid_rows]
y = y[valid_rows]

# Impute missing values in X
imputer = SimpleImputer(strategy='mean')  # Replace 'mean' with 'median' or 'most_frequent' if preferred
X_imputed = imputer.fit_transform(X)
progress_bar.update(20)  # Progress update

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)
progress_bar.update(10)  # Progress update

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the Logistic Regression model
model.fit(X_train, y_train)
progress_bar.update(30)  # Progress update

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
progress_bar.update(10)  # Progress update

# Predict on the full dataset for final predictions
final_predictions = model.predict(X_imputed)
polls_data_cleaned = polls_data_cleaned[valid_rows]
polls_data_cleaned['predicted'] = final_predictions

# Aggregate predictions to make a final prediction for each state
state_predictions = (
    polls_data_cleaned.groupby('state')
    .agg(
        classification=('classification', 'first'),  # Use the first value as the actual target
        final_prediction=('predicted', lambda x: x.value_counts().idxmax())  # Most frequent prediction
    )
    .reset_index()
)

# Save the aggregated state-level predictions
model_type = "logistic_regression"
state_output_path = f'state_level_predictions_{model_type}.csv'
state_predictions.to_csv(state_output_path, index=False)

print(f"State-level predictions saved to {state_output_path}")
progress_bar.update(10)  # Progress update

# Save individual predictions with corresponding states
row_output_path = f'state_predictions_{model_type}.csv'
polls_data_cleaned[['state', 'classification', 'predicted']].to_csv(row_output_path, index=False)

print(f"Row-level predictions saved to {row_output_path}")

# Print the top coefficients
feature_names = X.columns
coefficients = model.coef_[0]
feature_importances = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
feature_importances = feature_importances.sort_values(by='Coefficient', ascending=False)

# Print the top features
print("\nTop Features:\n", feature_importances.head(10))

# Plot the top coefficients
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'][:10], feature_importances['Coefficient'][:10])
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title(f'Top 10 Features in {model_type.replace("_", " ").title()}')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Close the progress bar
progress_bar.close()
