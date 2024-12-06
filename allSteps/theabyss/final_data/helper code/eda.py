import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

presidential_polls = pd.read_csv('2021_polls.csv')
presidential_polls_historic = pd.read_csv('president_polls_historical.csv')
senate_polls = pd.read_csv('senate_polls.csv')
senate_polls_historic = pd.read_csv('senate_polls_historical.csv')
governor_polls = pd.read_csv('governor_polls.csv')
governor_polls_historic = pd.read_csv('governor_polls_historical.csv')
house_polls = pd.read_csv('house_polls.csv')
house_polls_historic = pd.read_csv('house_polls_historical.csv')
approval_ratings = pd.read_csv('2021_approval.csv')
approval_ratings_historic = pd.read_csv('2016_approval.csv')

# Display the first few rows of each dataset
print("Presidential Polls:")
print(presidential_polls.head(), "\n")
print("Senate Polls:")
print(senate_polls.head(), "\n")
print("Governor Polls:")
print(governor_polls.head(), "\n")
print("House Polls:")
print(house_polls.head(), "\n")
print("Approval Ratings:")
print(approval_ratings.head(), "\n")

print("historic Presidential Polls:")
print(presidential_polls_historic.head(), "\n")
print("historic Senate Polls:")
print(senate_polls_historic.head(), "\n")
print("historic Governor Polls:")
print(governor_polls_historic.head(), "\n")
print("historic House Polls:")
print(house_polls_historic.head(), "\n")
print("historic Approval Ratings:")
print(approval_ratings_historic.head(), "\n")

# Check for missing values
print("Missing Values:")
print("Presidential Polls:\n", presidential_polls.isnull().sum(), "\n")
print("Senate Polls:\n", senate_polls.isnull().sum(), "\n")
print("Governor Polls:\n", governor_polls.isnull().sum(), "\n")
print("House Polls:\n", house_polls.isnull().sum(), "\n")
print("Approval Ratings:\n", approval_ratings.isnull().sum(), "\n")

# Describe datasets
print("Data Description:")
print(presidential_polls.describe(), "\n")

trump_approval = pd.read_csv('2016_approval.csv')
biden_harris_approval = pd.read_csv('2021_approval.csv')

# Filter for relevant columns and individuals
trump_approval = trump_approval[
    trump_approval['politician'].str.contains("Trump", case=False, na=False)
]
biden_harris_approval = biden_harris_approval[
    biden_harris_approval['politician/institution'].str.contains("Biden|Harris", case=False, na=False)
]

# Convert dates to datetime
trump_approval['end_date'] = pd.to_datetime(trump_approval['end_date'])
biden_harris_approval['date'] = pd.to_datetime(biden_harris_approval['date'])

# Calculate days in office
trump_start_date = pd.Timestamp("2017-01-20")
biden_start_date = pd.Timestamp("2021-01-20")

trump_approval['days_in_office'] = (trump_approval['end_date'] - trump_start_date).dt.days
biden_harris_approval['days_in_office'] = (biden_harris_approval['date'] - biden_start_date).dt.days

# Separate Biden and Harris
biden_approval = biden_harris_approval[
    biden_harris_approval['politician/institution'].str.contains("Biden", case=False)
]
harris_approval = biden_harris_approval[
    biden_harris_approval['politician/institution'].str.contains("Harris", case=False)
]

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(trump_approval['days_in_office'], trump_approval['yes'], label="Donald Trump Approval", marker='o', color='red')
plt.plot(biden_approval['days_in_office'], biden_approval['yes'], label="Joe Biden Approval", marker='o', color='blue')
plt.plot(harris_approval['days_in_office'], harris_approval['yes'], label="Kamala Harris Approval", marker='o', color='green')

# Customize plot
plt.title("Approval Ratings Over Days in Office")
plt.xlabel("Days in Office")
plt.ylabel("Approval Rating (%)")
plt.legend()
plt.grid(True)
plt.show()

# # Updated Visualization: Approval Ratings Over Time
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=approval_ratings, x='date', y='pct_estimate', hue='politician/institution')
# plt.title('Approval Ratings Over Time')
# plt.xlabel('Date')
# plt.ylabel('Approval Rating (%)')
# plt.xticks(rotation=45)
# plt.legend(title='Politician/Institution')
# plt.show()

# plt.figure(figsize=(12, 8))
# sns.boxplot(data=presidential_polls, x='state', y='pct', hue='answer')
# plt.title('Polling Distribution by State')
# plt.xlabel('State')
# plt.ylabel('Polling Percentage')
# plt.xticks(rotation=90)
# plt.legend(title='Candidate')
# plt.show()