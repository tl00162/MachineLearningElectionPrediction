import pandas as pd

# Load the dataset
polls_data = pd.read_csv('combined_polls_with_results.csv')

# Function to classify results as numbers
def classify_results_numeric(row):
    biden_percentage = row['biden_percentage']
    trump_percentage = row['trump_percentage']

    if pd.notnull(biden_percentage) and pd.notnull(trump_percentage):
        # Standard classifications
        if biden_percentage >= .60:
            return -3  # safe dem
        elif .55 <= biden_percentage < .60:
            return -2  # likely dem
        elif .50 <= biden_percentage < .55:
            return -1  # leans dem
        elif .50 <= trump_percentage < .55:
            return 1   # leans rep
        elif .55 <= trump_percentage < .60:
            return 2   # likely rep
        elif trump_percentage >= .60:
            return 3   # safe rep
        # Less than 50% for both candidates
        elif biden_percentage < .50 and trump_percentage < .50:
            return -1 if biden_percentage > trump_percentage else 1
    return None  # No classification for rows with missing data

# Apply numeric classification to the dataset
polls_data['classification_numeric'] = polls_data.apply(classify_results_numeric, axis=1)

# Save the updated dataset to a new CSV
output_path = 'polls_with_numeric_classifications.csv'
polls_data.to_csv(output_path, index=False)

print(f"Dataset with numeric classifications saved to {output_path}")
