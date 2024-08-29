import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import pandas as pd

class CollaborativeFiltering:
    """
    A class to implement collaborative filtering using Singular Value Decomposition (SVD).
    
    Attributes:
        user_item_matrix (pd.DataFrame): DataFrame containing the user-item interaction matrix.
        movie_titles (pd.Index): Index of movie titles.
        U (np.ndarray, optional): User matrix from SVD decomposition.
        sigma (np.ndarray, optional): Diagonal matrix of singular values from SVD decomposition.
        Vt (np.ndarray, optional): Movie matrix from SVD decomposition.
    """
    
    def __init__(self, user_item_matrix, movie_titles):
        """
        Initializes the CollaborativeFiltering with the user-item matrix and movie titles.
        
        Args:
            user_item_matrix (pd.DataFrame): DataFrame containing the user-item interaction matrix.
            movie_titles (pd.Index): Index of movie titles.
        """
        self.user_item_matrix = user_item_matrix
        self.movie_titles = movie_titles
        self.U, self.sigma, self.Vt = None, None, None

    def prepare_data(self):
        """
        Prepares the data by converting the user-item matrix to a sparse matrix format, performing SVD,
        and storing the decomposed matrices.
        """
        self.user_item_matrix = self.user_item_matrix.astype(float)
        
        sparse_matrix = csr_matrix(self.user_item_matrix)
        
        # Decompose the matrix using SVD with a smaller k value
        num_users, num_movies = sparse_matrix.shape
        k = min(num_users - 1, num_movies - 1, 10)  
        
        # Perform SVD
        self.U, self.sigma, self.Vt = svds(sparse_matrix, k=k)
        
        # Convert sigma to a diagonal matrix
        self.sigma = np.diag(self.sigma)

    def recommend(self, user_idx, top_n=10):
        """
        Generates movie recommendations for a given user based on collaborative filtering.
        
        Args:
            user_idx (int): Index of the user for whom recommendations are to be generated.
            top_n (int, optional): Number of top recommendations to return. Defaults to 10.
        
        Returns:
            list: List of top-n recommended movie titles.
        """
        if self.U is None or self.sigma is None or self.Vt is None:
            self.prepare_data()
        
        # Predict scores for the user by multiplying the decomposed matrices
        predicted_scores = np.dot(np.dot(self.U, self.sigma), self.Vt)[user_idx, :]
        
        # Sort and get the indices of the top N recommendations
        recommended_idx = np.argsort(predicted_scores)[::-1]
        
        # Map indices to movie titles
        recommended_items = self.movie_titles[recommended_idx[:top_n]].tolist()
        
        return recommended_items

def generate_synthetic_user_item_matrix(data, num_users=100, seed=42):
    """
    Generates a synthetic user-item interaction matrix with random ratings.
    
    Args:
        data (pd.DataFrame): DataFrame containing movie data with a 'title' column.
        num_users (int, optional): Number of synthetic users to generate. Defaults to 100.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    
    Returns:
        pd.DataFrame: DataFrame with synthetic user-item interaction matrix.
        pd.Index: Index of movie titles.
    """
    np.random.seed(seed)

    user_item_matrix = np.random.randint(1, 6, size=(num_users, len(data)))
    
    user_item_matrix = user_item_matrix.astype(float)
    
    user_item_df = pd.DataFrame(user_item_matrix, columns=data['title'])
    return user_item_df, data['title']
