import pandas as pd
from scipy.sparse.linalg import svds
import numpy as np

class CollaborativeFiltering:
    def __init__(self, data):
        self.data = data
        self.user_item_matrix = None
        self.U, self.sigma, self.Vt = None, None, None
        self.movie_indices = None
    
    def prepare_data(self):
        # Generate a user-item interaction matrix
        self.user_item_matrix = pd.get_dummies(self.data['title']).values
        
        # Convert to float type for SVD
        self.user_item_matrix = self.user_item_matrix.astype(float)
        
        # Decompose the matrix using SVD with a smaller k value
        num_users, num_movies = self.user_item_matrix.shape
        k = min(num_users - 1, num_movies - 1, 10)  # Adjust k based on matrix size
        
        self.U, self.sigma, self.Vt = svds(self.user_item_matrix, k=k)
        
        # Convert sigma to a diagonal matrix
        self.sigma = np.diag(self.sigma)
        
        # Store movie indices for lookup
        self.movie_indices = pd.get_dummies(self.data['title']).columns
    
    def recommend(self, movie_title, top_n=10):
        if self.U is None or self.sigma is None or self.Vt is None:
            self.prepare_data()
        
        # Find the index of the movie
        if movie_title not in self.movie_indices:
            return []
        movie_idx = self.movie_indices.get_loc(movie_title)
        
        # Predict scores by multiplying the decomposed matrices
        predicted_scores = np.dot(np.dot(self.U, self.sigma), self.Vt)
        
        # Sort and get the indices of the top similar movies
        similar_movies_idx = np.argsort(predicted_scores[:, movie_idx])[::-1]
        
        # Recommend top N movies
        recommended_movies = self.movie_indices[similar_movies_idx[1:top_n+1]].tolist()
        
        return recommended_movies
