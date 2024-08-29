import pandas as pd

# Load the dataset
file_path = '/Users/soul/Documents/hybrid_recommendation_system/data/Netflix-Cleaned-Data.csv'
df = pd.read_csv(file_path)

def handle_date_conversion(df):
    """
    Converts the 'date_added' column to datetime format, replacing 'Unknown' values with NaT.
    
    Args:
        df (pd.DataFrame): DataFrame containing the movie data.
    
    Returns:
        pd.DataFrame: DataFrame with 'date_added' column converted to datetime.
    """
    df['date_added'] = df['date_added'].replace('Unknown', pd.NaT)
    df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y', errors='coerce')
    return df

def normalize_duration(duration):
    """
    Normalizes the 'duration' column to extract the numeric value of duration in seasons or minutes.
    
    Args:
        duration (str): Duration string, e.g., '30 min' or '2 Season'.
    
    Returns:
        int or None: Normalized duration as an integer or None if the duration is not recognized.
    """
    if 'Season' in duration:
        return int(duration.split()[0])  # Number of seasons
    elif 'min' in duration:
        return int(duration.split()[0])  # Duration in minutes
    else:
        return None  # If not recognized

# Apply date conversion
df = handle_date_conversion(df)

# Normalize the 'duration' column
df['normalized_duration'] = df['duration'].apply(normalize_duration)

def encode_categorical_variables(df):
    """
    Encodes categorical variables using one-hot encoding, dropping the first category to avoid multicollinearity.
    
    Args:
        df (pd.DataFrame): DataFrame containing the movie data with categorical variables.
    
    Returns:
        pd.DataFrame: DataFrame with categorical variables encoded.
    """
    df_encoded = pd.get_dummies(df, columns=['type', 'rating', 'country'], drop_first=True)
    return df_encoded

# Encode categorical variables
df_encoded = encode_categorical_variables(df)
output_path = '/Users/soul/Documents/hybrid_recommendation_system/data/Netflix-encoded-Data.csv'
df_encoded.to_csv(output_path, index=False)
