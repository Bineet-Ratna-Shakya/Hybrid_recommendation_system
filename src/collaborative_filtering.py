import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CollaborativeFiltering:
    def __init__(self, user_item_matrix, movie_titles):
        self.user_item_matrix = user_item_matrix
        self.movie_titles = movie_titles
        self.U, self.sigma, self.Vt = None, None, None

    def prepare_data(self):
        """
        Prepare the collaborative filtering model by performing matrix factorization.
        """
        self.user_item_matrix = self.user_item_matrix.astype(float)
        sparse_matrix = csr_matrix(self.user_item_matrix)
        num_users, num_movies = sparse_matrix.shape
        k = min(num_users - 1, num_movies - 1, 10)
        self.U, self.sigma, self.Vt = svds(sparse_matrix, k=k)
        self.sigma = np.diag(self.sigma)
        logging.info("Collaborative Filtering model prepared.")

    def recommend(self, user_idx, top_n=10):
        """
        Recommend movies for a given user index based on the collaborative filtering model.

        Args:
            user_idx (int): The index of the user for whom recommendations are to be generated.
            top_n (int): The number of top recommendations to return.

        Returns:
            list: List of recommended movie titles.
        """
        
        if self.U is None or self.sigma is None or self.Vt is None:
            self.prepare_data()
        predicted_scores = np.dot(np.dot(self.U, self.sigma), self.Vt)[user_idx, :]
        recommended_idx = np.argsort(predicted_scores)[::-1]
        recommended_items = self.movie_titles[recommended_idx[:top_n]].tolist()
        return recommended_items


    def evaluate_performance(self, test_data):
        """
        Evaluate the performance of the collaborative filtering model using mean squared error.

        Args:
            test_data (pd.DataFrame): DataFrame containing the test user-item matrix.

        Returns:
            float: Mean Squared Error (MSE) of the model.
        """
        test_matrix = csr_matrix(test_data)
        num_users, num_movies = test_matrix.shape
        k = min(num_users - 1, num_movies - 1, 10)
        self.U, self.sigma, self.Vt = svds(csr_matrix(self.user_item_matrix), k=k)
        self.sigma = np.diag(self.sigma)
        predicted_ratings = np.dot(np.dot(self.U, self.sigma), self.Vt)
        test_ratings = test_matrix.toarray()
        mse = np.mean((test_ratings - predicted_ratings) ** 2)
        logging.info(f"Collaborative Filtering model performance: MSE = {mse:.4f}")
        return mse

    def update_user_item_matrix(self, feedback):
        """
        Update the user-item matrix based on user feedback.

        Args:
            feedback (dict): Dictionary with movie titles as keys and feedback ratings as values.
        """
        for movie, rating in feedback.items():
            if movie in self.user_item_matrix.columns:
                movie_idx = self.user_item_matrix.columns.get_loc(movie)
                self.user_item_matrix.iloc[0, movie_idx] = rating  # Update for user 0 as an example
        logging.info("User feedback incorporated into the user-item matrix.")
        # Recompute the model with updated matrix
        self.prepare_data()
        
def generate_synthetic_user_item_matrix(data, num_users=10, seed=42):
    np.random.seed(seed)
    user_item_matrix = np.random.randint(1, 6, size=(num_users, len(data)))
    user_item_matrix = user_item_matrix.astype(float)
    user_item_df = pd.DataFrame(user_item_matrix, columns=data['title'])
    return user_item_df, data['title']

