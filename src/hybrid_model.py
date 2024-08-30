from content_based_filtering import ContentBasedFiltering
from collaborative_filtering import CollaborativeFiltering
from deep_learning_model import NeuralCollaborativeFiltering

class HybridFiltering:
    def __init__(self, data, user_item_matrix, movie_titles, embedding_dim=20):
        self.data = data
        self.user_item_matrix = user_item_matrix
        self.movie_titles = movie_titles
        self.embedding_dim = embedding_dim
        self.content_filter = ContentBasedFiltering(data)
        self.collab_filter = CollaborativeFiltering(user_item_matrix, movie_titles)
        self.ncf_model = NeuralCollaborativeFiltering(user_item_matrix, movie_titles, embedding_dim)
    
    def prepare_models(self):
        self.content_filter.prepare_data()
        self.collab_filter.prepare_data()
        self.ncf_model.prepare_data()
    
    def recommend(self, movie_title, user_idx=0, top_n=10):
        # Content-Based Recommendations
        content_recommendations = self.content_filter.recommend(movie_title, top_n=top_n)
        
        # Collaborative Filtering Recommendations
        collab_recommendations = self.collab_filter.recommend(user_idx=user_idx, top_n=top_n)
        
        # Neural Collaborative Filtering Recommendations
        ncf_recommendations = self.ncf_model.recommend(user_idx=user_idx, top_n=top_n)
        
        # Combine recommendations (could be refined further)
        all_recommendations = set(content_recommendations + collab_recommendations + ncf_recommendations)
        hybrid_recommendations = list(all_recommendations)[:top_n]
        
        return hybrid_recommendations
