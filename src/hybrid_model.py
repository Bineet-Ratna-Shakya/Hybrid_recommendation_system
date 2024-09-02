from content_based_filtering import ContentBasedFiltering
from collaborative_filtering import CollaborativeFiltering
from deep_learning_model import NeuralCollaborativeFiltering
import numpy as np
import logging

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
        logging.info("Hybrid Filtering models prepared.")

    def recommend(self, movie_title, user_idx=0, top_n=10, content_weight=0.4, collab_weight=0.3, ncf_weight=0.3):
        # Content-Based Recommendations
        content_recommendations = self.content_filter.recommend(movie_title, top_n=top_n)
        
        # Collaborative Filtering Recommendations
        collab_recommendations = self.collab_filter.recommend(user_idx=user_idx, top_n=top_n)
        
        # Neural Collaborative Filtering Recommendations
        ncf_recommendations = self.ncf_model.recommend(user_idx=user_idx, top_n=top_n)
        
        # Combine recommendations with weights
        weighted_recommendations = {}
        for idx, title in enumerate(content_recommendations):
            weighted_recommendations[title] = weighted_recommendations.get(title, 0) + content_weight
        for idx, title in enumerate(collab_recommendations):
            weighted_recommendations[title] = weighted_recommendations.get(title, 0) + collab_weight
        for idx, title in enumerate(ncf_recommendations):
            weighted_recommendations[title] = weighted_recommendations.get(title, 0) + ncf_weight
        
        # Sort and get top recommendations
        sorted_recommendations = sorted(weighted_recommendations.items(), key=lambda x: x[1], reverse=True)
        hybrid_recommendations = [title for title, _ in sorted_recommendations[:top_n]]
        
        return hybrid_recommendations

    def collect_user_feedback(self, recommended_movies):
        return self.content_filter.collect_user_feedback(recommended_movies)
