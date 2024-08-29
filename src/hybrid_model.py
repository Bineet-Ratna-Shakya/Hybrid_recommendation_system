from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering

class HybridModel:
    def __init__(self, data):
        self.data = data
        self.collab_model = CollaborativeFiltering(data)
        self.content_model = ContentBasedFiltering(data)
    
    def recommend(self, movie_title, top_n=10):
        # Get recommendations from both models
        collab_recommendations = set(self.collab_model.recommend(movie_title, top_n))
        content_recommendations = set(self.content_model.recommend(movie_title, top_n))
        
        # Combine recommendations
        combined_recommendations = list(collab_recommendations.union(content_recommendations))
        
        # Return top N unique recommendations
        return combined_recommendations[:top_n]
