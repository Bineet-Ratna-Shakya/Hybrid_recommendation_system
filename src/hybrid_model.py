from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering

class HybridFiltering:
    """
    A class to implement hybrid movie recommendation using both content-based 
    and collaborative filtering techniques.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing movie data.
        user_item_matrix (pd.DataFrame): User-item interaction matrix.
        movie_titles (list): List of movie titles.
        collab_filter (CollaborativeFiltering): Instance of CollaborativeFiltering class.
        content_filter (ContentBasedFiltering): Instance of ContentBasedFiltering class.
    """
    
    def __init__(self, data, user_item_matrix, movie_titles):
        """
        Initializes the HybridFiltering with the provided data, user-item matrix, 
        and movie titles. Sets up both content-based and collaborative filtering instances.
        
        Args:
            data (pd.DataFrame): DataFrame containing movie data.
            user_item_matrix (pd.DataFrame): User-item interaction matrix.
            movie_titles (list): List of movie titles.
        """
        self.data = data
        self.user_item_matrix = user_item_matrix
        self.movie_titles = movie_titles
        #collaborative filtering with both user_item_matrix and movie_titles
        self.collab_filter = CollaborativeFiltering(user_item_matrix, movie_titles)
        self.content_filter = ContentBasedFiltering(data)
    
    def recommend(self, movie_title, user_idx=0, top_n=10):
        """
        Generates movie recommendations based on a hybrid approach using both content-based 
        and collaborative filtering.
        
        Args:
            movie_title (str): The movie title to base the recommendations on.
            user_idx (int, optional): Index of the user for collaborative filtering recommendations. Defaults to 0.
            top_n (int, optional): Number of top recommendations to return. Defaults to 10.
        
        Returns:
            list: List of top-n recommended movie titles.
        """
        # Content-Based Recommendations
        content_recommendations = self.content_filter.recommend(movie_title, top_n=top_n)
        
        # Collaborative Recommendations
        collab_recommendations = self.collab_filter.recommend(user_idx=user_idx, top_n=top_n)
        
        # Combine recommendations could be refined further to merge intelligently)
        hybrid_recommendations = list(set(content_recommendations + collab_recommendations))[:top_n]
        
        return hybrid_recommendations
