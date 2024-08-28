import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds

class CollaborativeFiltering:
    def __init__(self, data_path):
        # Load and preprocess the data
        self.data = pd.read_csv(data_path)
        self.user_item_matrix = None
        self.U = None
        self.sigma = None
        self.Vt = None

    def create_user_item_matrix(self):
        """
        Create the user-item interaction matrix.
        """
        self.user_item_matrix = self.data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        return self.user_item_matrix

    def matrix_factorization(self, num_features=50):
        """
        Apply Singular Value Decomposition (SVD) to factorize the user-item interaction matrix.
        """
        # Decompose the user-item matrix
        U, sigma, Vt = svds(self.user_item_matrix, k=num_features)
        
        # Convert sigma to a diagonal matrix
        self.sigma = np.diag(sigma)
        self.U = U
        self.Vt = Vt

    def predict_ratings(self):
        """
        Predict ratings by reconstructing the user-item interaction matrix.
        """
        user_predicted_ratings = np.dot(np.dot(self.U, self.sigma), self.Vt)
        return pd.DataFrame(user_predicted_ratings, columns=self.user_item_matrix.columns, index=self.user_item_matrix.index)

    def recommend_items(self, user_id, num_recommendations=5):
        """
        Recommend items to a user based on predicted ratings.
        """
        user_row_number = self.user_item_matrix.index.get_loc(user_id)
        sorted_user_ratings = self.predict_ratings().iloc[user_row_number].sort_values(ascending=False)
        return sorted_user_ratings.head(num_recommendations)

    def evaluate(self):
        """
        Evaluate the collaborative filtering model using RMSE.
        """
        # Split data into training and test sets
        train_data, test_data = train_test_split(self.data, test_size=0.2)
        
        # Create the user-item matrix from the training data
        self.user_item_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        
        # Apply SVD
        self.matrix_factorization()
        
        # Predict ratings
        predicted_ratings = self.predict_ratings()
        
        # Prepare test data for evaluation
        test_user_item_matrix = test_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        test_user_item_matrix = test_user_item_matrix.reindex_like(predicted_ratings).fillna(0)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_user_item_matrix.values.flatten(), predicted_ratings.values.flatten()))
        return rmse

# Example usage:
# cf = CollaborativeFiltering(data_path='/path/to/your/Netflix-encoded-Data.csv')
# cf.create_user_item_matrix()
# cf.matrix_factorization()
# print(cf.recommend_items(user_id=1))
# print("RMSE: ", cf.evaluate())
