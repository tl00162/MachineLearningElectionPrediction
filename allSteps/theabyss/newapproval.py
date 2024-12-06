import pandas as pd

# Load the dataset
trump_approval = pd.read_csv('2016_approval.csv')

# Ensure 'end_date' and 'sample_size' are in correct formats
trump_approval['end_date'] = pd.to_datetime(trump_approval['end_date'], errors='coerce')
trump_approval['sample_size'] = pd.to_numeric(trump_approval['sample_size'], errors='coerce')
trump_approval['yes'] = pd.to_numeric(trump_approval['yes'], errors='coerce')
trump_approval['no'] = pd.to_numeric(trump_approval['no'], errors='coerce')

# Drop rows with missing data in relevant columns
trump_approval = trump_approval.dropna(subset=['end_date', 'sample_size', 'yes', 'no'])

# Group by 'end_date' and calculate the weighted average approval percentage
grouped_approval = trump_approval.groupby('end_date').apply(
    lambda x: pd.Series({
        'weighted_yes': (x['yes'] * x['sample_size']).sum() / x['sample_size'].sum(),
        'weighted_no': (x['no'] * x['sample_size']).sum() / x['sample_size'].sum(),
        'total_sample_size': x['sample_size'].sum()
    })
).reset_index()

# Rename columns for clarity
grouped_approval.rename(columns={'weighted_yes': 'approval_percentage'}, inplace=True)

# Save to a new CSV file
output_path = '2017_approval.csv'
grouped_approval.to_csv(output_path, index=False)

print(f"Simplified approval data saved to {output_path}")
