import pandas as pd

# Load the datasets
presidential_results = pd.read_csv('2020_presidential_results.csv')
polls_data = pd.read_csv('combined_polls.csv')  # Combined polls data

# Reshape presidential results to one row per state
presidential_results_pivot = presidential_results.pivot(
    index=['state', 'totalvotes'],
    columns='candidate',
    values='percentage_of_votes'
).reset_index()

# Rename columns for clarity
presidential_results_pivot.rename(columns={
    'BIDEN, JOSEPH R. JR': 'biden_percentage',
    'TRUMP, DONALD J.': 'trump_percentage'
}, inplace=True)

# Ensure consistency in state names
polls_data['state'] = polls_data['state'].str.upper()  # Standardize case
presidential_results_pivot['state'] = presidential_results_pivot['state'].str.upper()

# Convert date column to datetime
polls_data['end_date'] = pd.to_datetime(polls_data['end_date'], errors='coerce')

# Add results columns to all rows, filling only for relevant dates
polls_data = pd.merge(
    polls_data,
    presidential_results_pivot,
    on=['state'],
    how='left'
)

# Mask results for dates after 12/1/2020
polls_data.loc[polls_data['end_date'] >= '2020-12-01', ['biden_percentage', 'trump_percentage']] = None

# Save the combined dataset to a new CSV file
output_path = 'combined_polls_with_results_conditional.csv'
polls_data.to_csv(output_path, index=False)

print(f"Combined data saved to {output_path}")
