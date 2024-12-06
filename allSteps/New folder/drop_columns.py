import pandas as pd

# Load the dataset
polls_data = pd.read_csv('polls_with_numeric_classifications.csv')

# Define columns to drop
columns_to_drop = [
    'url',                   # Unnecessary link to source
    'biden_percentage',      # Already represented by classification
    'trump_percentage',      # Already represented by classification
    'created_at',            # Metadata, not useful for ML
    'notes',                 # Typically contains unstructured or noisy data
    'sponsors',              # Irrelevant for most analyses
    'source',                # Metadata, not useful for ML
    'internal',              # Internal flag, typically not predictive
    'sponsor_candidate',     # Redundant information
    'sponsor_candidate_party',  # Redundant information
    'nationwide_batch',      # Not relevant to state-level data
    'transparency_score',    # May not correlate with target variable
]

# Drop the columns
polls_data_cleaned = polls_data.drop(columns=columns_to_drop, errors='ignore')

# Save the cleaned dataset to a new CSV
output_path = 'polls_cleaned.csv'
polls_data_cleaned.to_csv(output_path, index=False)

print(f"Cleaned dataset saved to {output_path}")