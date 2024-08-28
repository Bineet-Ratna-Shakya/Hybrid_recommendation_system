# content_based_filtering.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFiltering:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.tfidf_matrix = self.create_tfidf_matrix()
        self.similarity_matrix = self.calculate_similarity()

    def create_tfidf_matrix(self):
        # Use 'description' or 'listed_in' for TF-IDF vectorization
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.data['description'])  # Adjust column as necessary
        return tfidf_matrix

    def calculate_similarity(self):
        # Calculate cosine similarity between items
        cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        return pd.DataFrame(cosine_sim, index=self.data['show_id'], columns=self.data['show_id'])

    def recommend_similar_items(self, show_id, num_recommendations=5):
        if show_id not in self.similarity_matrix.index:
            raise ValueError(f"Show ID {show_id} not found in the data.")
        
        # Sort similar items
        similar_items = self.similarity_matrix[show_id].sort_values(ascending=False).drop(show_id)
        return similar_items.head(num_recommendations)
