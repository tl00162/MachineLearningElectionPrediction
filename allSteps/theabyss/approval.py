import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
trump_approval = pd.read_csv('2017_approval.csv')  # Processed data for Trump
biden_harris_approval = pd.read_csv('2021_approval.csv')  # Raw data for Biden and Harris

# Convert dates to datetime
trump_approval['end_date'] = pd.to_datetime(trump_approval['end_date'], errors='coerce')
biden_harris_approval['date'] = pd.to_datetime(biden_harris_approval['date'], errors='coerce')

# Calculate days in office
trump_start_date = pd.Timestamp("2017-01-20")
biden_start_date = pd.Timestamp("2021-01-20")

trump_approval['days_in_office'] = (trump_approval['end_date'] - trump_start_date).dt.days
biden_harris_approval['days_in_office'] = (biden_harris_approval['date'] - biden_start_date).dt.days

# Filter for Biden and Harris
biden_approval = biden_harris_approval[
    (biden_harris_approval['politician/institution'].str.contains("Biden", case=False)) &
    (biden_harris_approval['answer'] == "Approve")
]
harris_approval = biden_harris_approval[
    (biden_harris_approval['politician/institution'].str.contains("Harris", case=False)) &
    (biden_harris_approval['answer'] == "Approve")
]

# Plot the data
plt.figure(figsize=(12, 6))

# Plot Trump data
plt.plot(
    trump_approval['days_in_office'], 
    trump_approval['approval_percentage'], 
    label="Donald Trump Approval", marker='o', color='red'
)

# Plot Biden data
plt.plot(
    biden_approval['days_in_office'], 
    biden_approval['pct_estimate'], 
    label="Joe Biden Approval", marker='o', color='blue'
)

# Plot Harris data
plt.plot(
    harris_approval['days_in_office'], 
    harris_approval['pct_estimate'], 
    label="Kamala Harris Approval", marker='o', color='green'
)

# Customize plot
plt.title("Approval Ratings Over Days in Office")
plt.xlabel("Days in Office")
plt.ylabel("Approval Rating (%)")
plt.legend()
plt.grid(True)
plt.show()
