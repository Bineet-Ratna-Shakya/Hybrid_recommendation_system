import pandas as pd
from collaborative_filtering import CollaborativeFiltering, generate_synthetic_user_item_matrix
from content_based_filtering import ContentBasedFiltering
from hybrid_model import HybridFiltering

data_cleaned = pd.read_csv('/Users/soul/Documents/hybrid_recommendation_system/data/Netflix-encoded-Data.csv')
content_filter = ContentBasedFiltering(data_cleaned)
user_item_matrix, movie_titles = generate_synthetic_user_item_matrix(data_cleaned)
collab_filter = CollaborativeFiltering(user_item_matrix, movie_titles)
hybrid_filter = HybridFiltering(data_cleaned, user_item_matrix, movie_titles)

def get_movie_title(data):
    """
    Prompts the user to enter a movie title and checks if it exists in the dataset.
    
    Args:
        data (pd.DataFrame): DataFrame containing the movie data.
        
    Returns:
        str: Valid movie title from the dataset.
    """
    while True:
        movie_title = input("Enter a movie title (or type 'exit' to quit): ").strip()
        if movie_title.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            exit()
        elif movie_title.lower() in data['title'].str.lower().values:
            return movie_title
        else:
            print(f"Movie title '{movie_title}' not found in the dataset. Please try again.")

def display_recommendations(title, recommendations):
    """
    Displays a list of movie recommendations with a title header.
    
    Args:
        title (str): Title for the recommendation list.
        recommendations (list): List of recommended movie titles.
    """
    print(f"\n{'='*40}")
    print(f"--- {title} ---")
    print(f"{'='*40}")
    for i, rec in enumerate(recommendations, start=1):
        print(f"{i}. {rec}")
    print(f"{'='*40}")

def continue_or_exit():
    """
    Prompts the user to decide whether to get more recommendations or exit the program.
    
    Returns:
        bool: True if the user wants to continue, False if the user wants to exit.
    """
    while True:
        choice = input("Do you want to get recommendations for another movie? (yes/no): ").strip().lower()
        if choice == 'yes':
            return True
        elif choice == 'no':
            print("Thank you for using the recommendation system. Goodbye!")
            exit()
        else:
            print("Invalid choice. Please enter 'yes' or 'no'.")

# Main program loop
print("\n" + "="*40)
print("WELCOME TO THE HYBRID RECOMMENDATION SYSTEM")
print("="*40)

while True:
    movie_title = get_movie_title(data_cleaned)

    print("\n" + "="*40)
    print("GENERATING RECOMMENDATIONS...")
    print("="*40)

    content_recommendations = content_filter.recommend(movie_title)
    display_recommendations("Content-Based Recommendations", content_recommendations)

    collaborative_recommendations = collab_filter.recommend(user_idx=0)
    display_recommendations("Collaborative Recommendations", collaborative_recommendations)

    # Hybrid Filtering Recommendations
    hybrid_recommendations = hybrid_filter.recommend(movie_title, user_idx=0)
    display_recommendations("Hybrid Recommendations", hybrid_recommendations)

    if not continue_or_exit():
        break

print("\n" + "="*40)
print("END OF RECOMMENDATIONS")
print("="*40)
