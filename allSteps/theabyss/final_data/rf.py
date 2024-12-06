import pandas as pd

# Load the dataset
polls_data = pd.read_csv('polls_cleaned.csv')

# Filter for Democratic and Republican candidates
polls_data_filtered = polls_data[polls_data['party'].isin(['DEM', 'REP'])]

# Find the highest `pct` for each `race_id` and `question_id` for DEM and REP
max_pct_dem = polls_data_filtered[polls_data_filtered['party'] == 'DEM'].groupby(
    ['race_id', 'question_id'], as_index=False
).apply(lambda x: x.loc[x['pct'].idxmax()])

max_pct_rep = polls_data_filtered[polls_data_filtered['party'] == 'REP'].groupby(
    ['race_id', 'question_id'], as_index=False
).apply(lambda x: x.loc[x['pct'].idxmax()])

# Combine DEM and REP results into one DataFrame
combined_data = pd.merge(
    max_pct_dem, max_pct_rep,
    on=['race_id', 'question_id'],
    suffixes=('_dem', '_rep')
)

# Print the shape of the output dataset
print("Shape of the condensed dataset:", combined_data.shape)

# Print the number of unique question_ids
unique_question_ids = combined_data['question_id'].nunique()
print("Number of unique question_ids:", unique_question_ids)

# Save the condensed dataset
output_path = 'condensed_polls.csv'
combined_data.to_csv(output_path, index=False)

print(f"Condensed dataset saved to {output_path}")
