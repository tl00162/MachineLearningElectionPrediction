import pandas as pd

# Load the CSV files
senate_polls = pd.read_csv('senate_polls.csv')
senate_polls_historic = pd.read_csv('senate_polls_historical.csv')
governor_polls = pd.read_csv('governor_polls.csv')
governor_polls_historic = pd.read_csv('governor_polls_historical.csv')
house_polls = pd.read_csv('house_polls.csv')
house_polls_historic = pd.read_csv('house_polls_historical.csv')

# Add a poll_type column to each DataFrame
senate_polls['poll_type'] = 'Senate'
senate_polls_historic['poll_type'] = 'Senate'
governor_polls['poll_type'] = 'Governor'
governor_polls_historic['poll_type'] = 'Governor'
house_polls['poll_type'] = 'House'
house_polls_historic['poll_type'] = 'House'

# Standardize column names (ensure all files have a consistent 'date' column)
for df in [senate_polls, senate_polls_historic, governor_polls, governor_polls_historic, house_polls, house_polls_historic]:
    if 'end_date' in df.columns:
        df.rename(columns={'end_date': 'date'}, inplace=True)
    elif 'start_date' in df.columns:
        df.rename(columns={'start_date': 'date'}, inplace=True)

# Combine all DataFrames
combined_polls = pd.concat([
    senate_polls,
    senate_polls_historic,
    governor_polls,
    governor_polls_historic,
    house_polls,
    house_polls_historic
], ignore_index=True)

# Ensure 'state' and 'date' columns are in correct formats
if 'state' in combined_polls.columns and 'date' in combined_polls.columns:
    combined_polls['state'] = combined_polls['state'].astype(str)
    combined_polls['date'] = pd.to_datetime(combined_polls['date'], errors='coerce')
else:
    raise ValueError("The required columns 'state' and 'date' are missing in the combined dataset.")

# Sort by state and date
combined_polls_sorted = combined_polls.sort_values(by=['state', 'date']).reset_index(drop=True)

# Save the combined and sorted DataFrame to a new CSV file
output_path = 'combined_polls_with_type.csv'
combined_polls_sorted.to_csv(output_path, index=False)

print(f"Combined and sorted polls data saved to {output_path}")
