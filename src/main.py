import pandas as pd
from collaborative_filtering import CollaborativeFiltering, generate_synthetic_user_item_matrix
from content_based_filtering import ContentBasedFiltering
from deep_learning_model import NeuralCollaborativeFiltering
from hybrid_model import HybridFiltering
from utils import save_feedback
import visualizations  # Import the visualization module

# Load and preprocess the data
data_cleaned = pd.read_csv('/Users/soul/Documents/hybrid_recommendation_system/data/Netflix-encoded-Data.csv')

# Generate user-item matrix and movie titles
num_synthetic_users = 10  # Number of synthetic users
user_item_matrix, movie_titles = generate_synthetic_user_item_matrix(data_cleaned, num_users=num_synthetic_users)

# Initialize models
content_filter = ContentBasedFiltering(data_cleaned)
collab_filter = CollaborativeFiltering(user_item_matrix, movie_titles)
hybrid_filter = HybridFiltering(data_cleaned, user_item_matrix, movie_titles, embedding_dim=20)
hybrid_filter.prepare_models()

# Initialize and train Neural Collaborative Filtering model
ncf = NeuralCollaborativeFiltering(user_item_matrix, movie_titles, embedding_dim=20)
ncf.train(epochs=5)

# Print model weights
ncf.print_model_weight_summary()


def get_movie_title(data):
    while True:
        movie_title = input("Enter a movie title (or type 'exit' to quit): ").strip()
        if movie_title.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            exit()
        elif movie_title.lower() in data['title'].str.lower().values:
            correct_title = data[data['title'].str.lower() == movie_title.lower()]['title'].values[0]
            return correct_title
        else:
            print(f"Movie title '{movie_title}' not found in the dataset. Please try again.")

def get_user_index():
    while True:
        try:
            user_index = int(input(f"Enter a synthetic user index (0 to {num_synthetic_users - 1}): "))
            if 0 <= user_index < num_synthetic_users:
                return user_index
            else:
                print(f"Invalid index. Please enter a number between 0 and {num_synthetic_users - 1}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def display_recommendations(title, recommendations):
    print(f"\n{'='*40}")
    print(f"--- {title} ---")
    print(f"{'='*40}")
    for i, rec in enumerate(recommendations, start=1):
        print(f"{i}. {rec}")
    print(f"{'='*40}")

def continue_or_exit():
    while True:
        choice = input("Do you want to get recommendations for another movie or user? (yes/no): ").strip().lower()
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
    user_index = get_user_index()

    print("\n" + "="*40)
    print("GENERATING RECOMMENDATIONS...")
    print("="*40)

    # Content-Based Recommendations
    content_recommendations = content_filter.recommend(movie_title)
    display_recommendations("Content-Based Recommendations", content_recommendations)

    # Collaborative Recommendations
    collaborative_recommendations = collab_filter.recommend(user_idx=user_index)
    display_recommendations("Collaborative Recommendations", collaborative_recommendations)

    # Hybrid Filtering Recommendations
    hybrid_recommendations = hybrid_filter.recommend(movie_title, user_idx=user_index)
    display_recommendations("Hybrid Recommendations", hybrid_recommendations)

    # Generate and display visualizations
    #visualizations.plot_top_n_recommendations(hybrid_recommendations, user_index)
    #visualizations.plot_recommendation_comparison(content_recommendations, collaborative_recommendations, hybrid_recommendations)

    # Collect user feedback
    feedback = hybrid_filter.collect_user_feedback(hybrid_recommendations)
    save_feedback(feedback)  # Save the feedback for future use

    # Update models with user feedback
    collab_filter.update_user_item_matrix(feedback)  # Update collaborative filtering model
    hybrid_filter.ncf_model.incorporate_feedback(feedback)  # Update NCF model with feedback
    
    print("Thank you for your feedback!")

    if not continue_or_exit():
        break

print("\n" + "="*40)
print("END OF RECOMMENDATIONS")
print("="*40)
