import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.style.use('ggplot')  # Use a more aesthetic style



def plot_user_item_interaction_matrix(user_item_matrix):
    """Plot a heatmap of user-item interactions."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(user_item_matrix, cmap='coolwarm', cbar=True)
    plt.title('User-Item Interaction Matrix')
    plt.xlabel('Items')
    plt.ylabel('Users')
    plt.tight_layout()
    plt.show()

def plot_user_item_ratings_heatmap(user_item_matrix):
    """Plot a heatmap of the user-item ratings."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(user_item_matrix, annot=True, fmt=".1f", cmap='coolwarm', cbar=True)
    plt.title('Heatmap of User-Item Ratings')
    plt.xlabel('Items')
    plt.ylabel('Users')
    plt.tight_layout()
    plt.show()

def plot_top_n_recommendations(recommendations, user_id):
    """Plot the top N recommendations for a specific user."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=recommendations, y=np.arange(len(recommendations)), palette='viridis')
    plt.title(f'Top-N Recommendations for User {user_id}')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Items')
    plt.tight_layout()
    plt.show()

def plot_recommendation_comparison(content_recommendations, collaborative_recommendations, hybrid_recommendations):
    """Compare recommendations from different models."""
    data = {
        'Content-Based': content_recommendations,
        'Collaborative': collaborative_recommendations,
        'Hybrid': hybrid_recommendations
    }
    rec_df = pd.DataFrame(data)

    plt.figure(figsize=(14, 7))
    rec_df.plot(kind='bar', figsize=(14, 7), color=['#FF6F61', '#6B5B95', '#88B04B'])
    plt.title('Comparison of Recommendations from Different Models')
    plt.xlabel('Recommended Items')
    plt.ylabel('Score/Rank')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
