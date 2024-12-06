import pandas as pd

# Load the CSV files
combined_polls_2017_2021 = pd.read_csv('combined_polls_2017_2021.csv')
combined_polls_with_type = pd.read_csv('combined_polls_with_type.csv')

# Align columns
all_columns = list(set(combined_polls_2017_2021.columns).union(combined_polls_with_type.columns))

# Ensure both DataFrames have all columns
combined_polls_2017_2021 = combined_polls_2017_2021.reindex(columns=all_columns, fill_value=None)
combined_polls_with_type = combined_polls_with_type.reindex(columns=all_columns, fill_value=None)

# Combine the datasets
full_combined_polls = pd.concat([combined_polls_2017_2021, combined_polls_with_type], ignore_index=True)

# Fill missing states with "Nation-wide" for presidential polls
presidential_polls = full_combined_polls['poll_type'] == 'Presidential'
full_combined_polls.loc[presidential_polls & full_combined_polls['state'].isna(), 'state'] = 'Nation-wide'

# Count rows with missing 'state' and 'end_date'
missing_state = full_combined_polls['state'].isna().sum()
missing_date = full_combined_polls['end_date'].isna().sum()
total_missing = full_combined_polls[full_combined_polls['state'].isna() | full_combined_polls['end_date'].isna()].shape[0]

print(f"Rows with missing 'state': {missing_state}")
print(f"Rows with missing 'end_date': {missing_date}")
print(f"Total rows with missing 'state' or 'end_date': {total_missing}")

# Drop rows with missing 'end_date' only
full_combined_polls = full_combined_polls.dropna(subset=['end_date'])

# Ensure 'state' and 'end_date' columns are in correct formats
full_combined_polls['state'] = full_combined_polls['state'].astype(str)
full_combined_polls['end_date'] = pd.to_datetime(full_combined_polls['end_date'], errors='coerce')

# Sort by state and date
full_combined_polls_sorted = full_combined_polls.sort_values(by=['state', 'end_date']).reset_index(drop=True)

# Save the combined and sorted DataFrame to a new CSV file
output_path = 'full_combined_polls.csv'
full_combined_polls_sorted.to_csv(output_path, index=False)

print(f"Combined and sorted polls data saved to {output_path}")
print(f"Shape of the combined dataset: {full_combined_polls_sorted.shape}")
