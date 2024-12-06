import pandas as pd

# Load the cleaned dataset
condensed_polls_cleaned = pd.read_csv('cleaned_condensed_polls_classification.csv')

# Move the 'classification' column to the last position
if 'classification' in condensed_polls_cleaned.columns:
    classification_column = condensed_polls_cleaned.pop('classification')  # Remove and store the column
    condensed_polls_cleaned['classification'] = classification_column      # Add it back at the end

# Save the updated dataset
output_path = 'cleaned_condensed_polls_final.csv'
condensed_polls_cleaned.to_csv(output_path, index=False)

# Print confirmation and the new shape of the dataset
print(f"Dataset saved with classification as the last column to {output_path}")
print("Shape of the dataset after reordering columns:", condensed_polls_cleaned.shape)
