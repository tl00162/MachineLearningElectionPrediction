import pandas as pd

# Load the 2021 polls dataset
polls_2021 = pd.read_csv('2021_polls.csv')

# Define the mapping from numeric_grade to fte_grade
numeric_to_fte_grade = {
    (0.5, 1.0): 'C',
    (1.1, 1.5): 'B-',
    (1.6, 2.0): 'B',
    (2.2, 2.8): 'A',
    (2.9, 3.0): 'A+',
}

# Function to map numeric_grade to fte_grade
def map_numeric_to_fte(numeric_grade):
    for (lower, upper), grade in numeric_to_fte_grade.items():
        if lower <= numeric_grade <= upper:
            return grade
    return None  # If no match, return None

# Apply the mapping to create the fte_grade column
if 'numeric_grade' in polls_2021.columns:
    polls_2021['fte_grade'] = polls_2021['numeric_grade'].apply(map_numeric_to_fte)

# Save the updated DataFrame to a new CSV file
output_path = '2021_polls_with_fte_grade.csv'
polls_2021.to_csv(output_path, index=False)

print(f"Updated 2021 polls data saved to {output_path}")
