import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

class CollaborativeFiltering:
    def __init__(self, user_item_matrix, movie_titles):
        self.user_item_matrix = user_item_matrix
        self.movie_titles = movie_titles
        self.U, self.sigma, self.Vt = None, None, None

    def prepare_data(self):
        # Ensure the user-item matrix is of float type
        self.user_item_matrix = self.user_item_matrix.astype(float)
        
        # Convert the user-item matrix to a sparse matrix format (e.g., CSR format)
        sparse_matrix = csr_matrix(self.user_item_matrix)
        
        # Decompose the matrix using SVD with a smaller k value
        num_users, num_movies = sparse_matrix.shape
        k = min(num_users - 1, num_movies - 1, 10)  # Adjust k based on matrix size
        
        # Perform SVD
        self.U, self.sigma, self.Vt = svds(sparse_matrix, k=k)
        
        # Convert sigma to a diagonal matrix
        self.sigma = np.diag(self.sigma)

    def recommend(self, user_idx, top_n=10):
        if self.U is None or self.sigma is None or self.Vt is None:
            self.prepare_data()
        
        # Predict scores for the user by multiplying the decomposed matrices
        predicted_scores = np.dot(np.dot(self.U, self.sigma), self.Vt)[user_idx, :]
        
        # Sort and get the indices of the top N recommendations
        recommended_idx = np.argsort(predicted_scores)[::-1]
        
        # Map indices to movie titles
        recommended_items = self.movie_titles[recommended_idx[:top_n]].tolist()
        
        return recommended_items


import numpy as np
import pandas as pd

# Generate a synthetic user-item interaction matrix with random ratings
def generate_synthetic_user_item_matrix(data, num_users=100):
    # Create a matrix with random ratings between 1 and 5
    user_item_matrix = np.random.randint(1, 6, size=(num_users, len(data)))
    
    # Convert the matrix to float type
    user_item_matrix = user_item_matrix.astype(float)
    
    # Convert to a DataFrame for easier handling
    user_item_df = pd.DataFrame(user_item_matrix, columns=data['title'])
    return user_item_df, data['title']

