import pandas as pd

# Load the CSV files
polls_2017 = pd.read_csv('2017_polls.csv')
polls_2021_filtered = pd.read_csv('2021_polls_filtered.csv')

# Add the `poll_type` column to both datasets
polls_2017['poll_type'] = 'Presidential'
polls_2021_filtered['poll_type'] = 'Presidential'

# Reorder the columns of 2021 polls to match the 2017 polls
polls_2021_reordered = polls_2021_filtered[polls_2017.columns]

# Combine the datasets
combined_polls = pd.concat([polls_2017, polls_2021_reordered], ignore_index=True)

# Save the combined DataFrame to a new CSV file
output_path = 'combined_polls_2017_2021.csv'
combined_polls.to_csv(output_path, index=False)

# Print the shape of the combined DataFrame
print(f"Combined polls data saved to {output_path}")
print(f"Shape of the combined dataset: {combined_polls.shape}")
