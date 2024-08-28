# hybrid_model.py
import pandas as pd

class HybridModel:
    def __init__(self, collaborative_model, content_model):
        self.collaborative_model = collaborative_model
        self.content_model = content_model

    def combine_results(self, show_id, weight_collab=0.5, weight_content=0.5):
        # Combine recommendations from both models
        collab_recs = self.collaborative_model.recommend_items(show_id)
        content_recs = self.content_model.recommend_similar_items(show_id)

        combined_recs = collab_recs.add(content_recs, fill_value=0)
        combined_recs = (weight_collab * collab_recs).add(weight_content * content_recs, fill_value=0)
        return combined_recs.sort_values(ascending=False)

    def recommend(self, show_id, num_recommendations=5):
        # Get hybrid recommendations
        recommendations = self.combine_results(show_id)
        return recommendations.head(num_recommendations)
