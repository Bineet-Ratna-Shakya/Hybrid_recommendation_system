# utils.py
import pandas as pd
import logging

def save_feedback(feedback, file_path='user_feedback.csv'):
    """
    Save user feedback to a CSV file.

    Args:
        feedback (dict): Dictionary containing movie titles and user ratings.
        file_path (str): Path to the CSV file where feedback will be saved.
    """
    df = pd.DataFrame(list(feedback.items()), columns=['movie', 'rating'])
    df.to_csv(file_path, mode='a', header=False, index=False)
    logging.info("User feedback saved.")
