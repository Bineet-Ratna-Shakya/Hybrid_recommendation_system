from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering

class HybridFiltering:
    def __init__(self, data, user_item_matrix, movie_titles):
        self.data = data
        self.user_item_matrix = user_item_matrix
        self.movie_titles = movie_titles
        # Initialize collaborative filtering with both user_item_matrix and movie_titles
        self.collab_filter = CollaborativeFiltering(user_item_matrix, movie_titles)
        self.content_filter = ContentBasedFiltering(data)
    
    def recommend(self, movie_title, user_idx=0, top_n=10):
        # Content-Based Recommendations
        content_recommendations = self.content_filter.recommend(movie_title, top_n=top_n)
        
        # Collaborative Recommendations
        collab_recommendations = self.collab_filter.recommend(user_idx=user_idx, top_n=top_n)
        
        # Combine recommendations (this could be refined further to merge intelligently)
        hybrid_recommendations = list(set(content_recommendations + collab_recommendations))[:top_n]
        
        return hybrid_recommendations
