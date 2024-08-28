# collaborative_filtering.py
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

class CollaborativeFiltering:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.user_item_matrix = self.create_item_similarity_matrix()
        self.predicted_scores = None

    def create_item_similarity_matrix(self):
        # Create an implicit user-item interaction matrix based on item co-occurrence
        item_ids = self.data['show_id'].unique()
        interactions = np.random.randint(1, 100, size=(len(item_ids), len(item_ids)))  # Random interaction counts for simplicity
        
        # Convert to DataFrame for easier manipulation
        item_item_matrix = pd.DataFrame(interactions, index=item_ids, columns=item_ids)
        return item_item_matrix

    def svd_decomposition(self, k=50):
        # Perform SVD on the item-item similarity matrix
        matrix = self.user_item_matrix.values
        item_ratings_mean = np.mean(matrix, axis=1)
        matrix_demeaned = matrix - item_ratings_mean.reshape(-1, 1)
        
        U, sigma, Vt = svds(matrix_demeaned, k=k)
        sigma = np.diag(sigma)
        
        self.predicted_scores = np.dot(np.dot(U, sigma), Vt) + item_ratings_mean.reshape(-1, 1)
        return pd.DataFrame(self.predicted_scores, columns=self.user_item_matrix.columns, index=self.user_item_matrix.index)

    def recommend_items(self, show_id, num_recommendations=5):
        if self.predicted_scores is None:
            self.svd_decomposition()
        
        # Sort predicted scores for the specified show
        if show_id not in self.user_item_matrix.index:
            raise ValueError(f"Show ID {show_id} not found in the data.")
        
        sorted_predictions = pd.Series(self.predicted_scores[self.user_item_matrix.index.get_loc(show_id)],
                                       index=self.user_item_matrix.columns).sort_values(ascending=False)
        
        # Exclude the current show itself
        sorted_predictions = sorted_predictions.drop(show_id, errors='ignore')
        return sorted_predictions.head(num_recommendations)
