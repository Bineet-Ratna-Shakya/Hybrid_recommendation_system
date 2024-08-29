import pandas as pd
from collaborative_filtering import CollaborativeFiltering, generate_synthetic_user_item_matrix
from content_based_filtering import ContentBasedFiltering
from hybrid_model import HybridFiltering

# Load the cleaned data
data_cleaned = pd.read_csv('/Users/soul/Documents/hybrid_recommendation_system/data/Netflix-Cleaned-Data.csv')  # Update path accordingly

# Initialize the Content-Based Filtering
content_filter = ContentBasedFiltering(data_cleaned)

# Generate a synthetic user-item interaction matrix for collaborative filtering
user_item_matrix, movie_titles = generate_synthetic_user_item_matrix(data_cleaned)

# Initialize the Collaborative Filtering with movie titles
collab_filter = CollaborativeFiltering(user_item_matrix, movie_titles)

# Initialize the Hybrid Filtering with movie titles
hybrid_filter = HybridFiltering(data_cleaned, user_item_matrix, movie_titles)

# Function to get user input and check for valid movie title
def get_movie_title(data):
    while True:
        movie_title = input("Enter a movie title: ").strip()
        if movie_title.lower() in data['title'].str.lower().values:
            return movie_title
        else:
            print(f"Movie title '{movie_title}' not found in the dataset. Please try again.")

# Get a valid movie title from the user
movie_title = get_movie_title(data_cleaned)

# Content-Based Filtering Recommendations
content_recommendations = content_filter.recommend(movie_title)
print("Content-Based Recommendations:", content_recommendations)

# Collaborative Filtering Recommendations for a sample user (user_idx=0)
collaborative_recommendations = collab_filter.recommend(user_idx=0)
print("Collaborative Recommendations:", collaborative_recommendations)

# Hybrid Filtering Recommendations
hybrid_recommendations = hybrid_filter.recommend(movie_title, user_idx=0)
print("Hybrid Recommendations:", hybrid_recommendations)
