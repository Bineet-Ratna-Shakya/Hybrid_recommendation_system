import pandas as pd

# Load the dataset
file_path = '/Users/soul/Documents/hybrid_recommendation_system/data/Netflix-Cleaned-Data.csv'
df = pd.read_csv(file_path)

# Handle 'date_added' conversion
df['date_added'] = df['date_added'].replace('Unknown', pd.NaT)
df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y', errors='coerce')

# Normalize the 'duration' column
def normalize_duration(duration):
    if 'Season' in duration:
        return int(duration.split()[0])  # Number of seasons
    elif 'min' in duration:
        return int(duration.split()[0])  # Duration in minutes
    else:
        return None  # If not recognized

df['normalized_duration'] = df['duration'].apply(normalize_duration)

# Encoding categorical variables
df_encoded = pd.get_dummies(df, columns=['type', 'rating', 'country'], drop_first=True)

# Save the encoded DataFrame to a CSV file
df_encoded.to_csv('/Users/soul/Documents/hybrid_recommendation_system/data/Netflix-encoded-Data.csv', index=False)
