import pandas as pd

# Load datasets
trump_approval = pd.read_csv('2017_approval.csv')  # Trump data
biden_harris_approval = pd.read_csv('2021_approval.csv')  # Biden and Harris data

# Convert dates to datetime
trump_approval['end_date'] = pd.to_datetime(trump_approval['end_date'], errors='coerce')
biden_harris_approval['date'] = pd.to_datetime(biden_harris_approval['date'], errors='coerce')

# Process Trump data
trump_start_date = pd.Timestamp("2017-01-20")
trump_approval['days_in_office'] = (trump_approval['end_date'] - trump_start_date).dt.days
trump_approval_simplified = trump_approval[['end_date', 'days_in_office', 'approval_percentage']].copy()
trump_approval_simplified.rename(columns={
    'end_date': 'date',
    'approval_percentage': 'approval_rating'
}, inplace=True)
trump_approval_simplified['politician'] = 'Donald Trump'

# Process Biden and Harris data
biden_start_date = pd.Timestamp("2021-01-20")
biden_harris_approval['days_in_office'] = (biden_harris_approval['date'] - biden_start_date).dt.days

biden_approval = biden_harris_approval[
    (biden_harris_approval['politician/institution'].str.contains("Biden", case=False)) &
    (biden_harris_approval['answer'] == "Approve")
][['date', 'days_in_office', 'pct_estimate']].copy()
biden_approval.rename(columns={'pct_estimate': 'approval_rating'}, inplace=True)
biden_approval['politician'] = 'Joe Biden'

harris_approval = biden_harris_approval[
    (biden_harris_approval['politician/institution'].str.contains("Harris", case=False)) &
    (biden_harris_approval['answer'] == "Approve")
][['date', 'days_in_office', 'pct_estimate']].copy()
harris_approval.rename(columns={'pct_estimate': 'approval_rating'}, inplace=True)
harris_approval['politician'] = 'Kamala Harris'

# Combine all data
combined_approval = pd.concat([trump_approval_simplified, biden_approval, harris_approval])

# Sort by date
combined_approval.sort_values(by='date', inplace=True)

# Save to a new CSV file
output_path = 'combined_approval.csv'
combined_approval.to_csv(output_path, index=False)

print(f"Combined approval data saved to {output_path}")
