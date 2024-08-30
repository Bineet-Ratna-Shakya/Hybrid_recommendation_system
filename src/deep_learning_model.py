import numpy as np
import pandas as pd
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras._tf_keras.keras.optimizers import Adam

class NeuralCollaborativeFiltering:
    """
    A class to implement Neural Collaborative Filtering (NCF) using TensorFlow/Keras.
    
    Attributes:
        num_users (int): Number of unique users.
        num_movies (int): Number of unique movies.
        embedding_dim (int): Dimension of the embedding vectors.
        model (tf.keras.Model): The Keras model for NCF.
    """
    def prepare_data(self):
        """
        Prepare data for training the Neural Collaborative Filtering model.
        
        This might include tasks such as normalizing data, creating training 
        and validation splits, or encoding user and movie IDs.
        """
        # Add code to prepare data here, like encoding user/movie indices
        interactions = self.user_item_matrix.stack().reset_index()
        interactions.columns = ['user', 'movie', 'rating']
        
        # Convert user and movie indices to numeric encodings if needed
        self.user_mapping = {user: idx for idx, user in enumerate(interactions['user'].unique())}
        self.movie_mapping = {movie: idx for idx, movie in enumerate(interactions['movie'].unique())}
        
        # Store prepared data
        self.interactions = interactions
        self.X = [interactions['user'].map(self.user_mapping).values, 
                  interactions['movie'].map(self.movie_mapping).values]
        self.y = interactions['rating'].values    
        
    def __init__(self, user_item_matrix, movie_titles=None, embedding_dim=20):
        """
        Initializes the NeuralCollaborativeFiltering with user-item interaction matrix and embedding dimension.
        
        Args:
            user_item_matrix (pd.DataFrame): DataFrame containing the user-item interaction matrix.
            embedding_dim (int, optional): Dimension of the embedding vectors. Defaults to 20.
        """
        self.user_item_matrix = user_item_matrix
        self.movie_titles = movie_titles
        self.num_users = user_item_matrix.shape[0]
        self.num_movies = user_item_matrix.shape[1]
        self.embedding_dim = embedding_dim
        self.model = self.build_model()

        # Build the model
        self.model = self.build_model()

    def build_model(self):
        """
        Builds the Neural Collaborative Filtering model using Keras.
        
        Returns:
            tf.keras.Model: Compiled NCF model.
        """
        user_input = Input(shape=(1,), name='user')
        movie_input = Input(shape=(1,), name='movie')
        
        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.embedding_dim)(user_input)
        movie_embedding = Embedding(input_dim=self.num_movies, output_dim=self.embedding_dim)(movie_input)
        
        user_vector = Flatten()(user_embedding)
        movie_vector = Flatten()(movie_embedding)
        
        concat = Concatenate()([user_vector, movie_vector])
        hidden = Dense(128, activation='relu')(concat)
        output = Dense(1, activation='sigmoid')(hidden)
        
        model = Model(inputs=[user_input, movie_input], outputs=output)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def train(self, epochs=5, batch_size=32):
        """
        Trains the NCF model using the user-item interaction matrix.
        
        Args:
            epochs (int, optional): Number of training epochs. Defaults to 5.
            batch_size (int, optional): Batch size for training. Defaults to 32.
        """
        interactions = self.user_item_matrix.stack().reset_index()
        interactions.columns = ['user', 'movie', 'rating']
        
        X = [interactions['user'].values, interactions['movie'].values]
        y = interactions['rating'].values
        
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    def recommend(self, user_idx, top_n=10):
        """
        Generates movie recommendations for a given user based on the trained NCF model.
        
        Args:
            user_idx (int): Index of the user for whom recommendations are to be generated.
            top_n (int, optional): Number of top recommendations to return. Defaults to 10.
        
        Returns:
            list: List of top-n recommended movie titles.
        """
        # Create a list of all movie indices
        movie_indices = np.arange(self.num_movies)
        
        # Predict scores for the user
        predictions = []
        for movie_idx in movie_indices:
            prediction = self.model.predict([np.array([user_idx]), np.array([movie_idx])])
            predictions.append((movie_idx, prediction[0, 0]))
        
        # Sort and get the indices of the top N recommendations
        recommended_idx = sorted(predictions, key=lambda x: x[1], reverse=True)
        recommended_idx = [idx for idx, _ in recommended_idx[:top_n]]
        
        # Map indices to movie titles
        recommended_items = self.user_item_matrix.columns[recommended_idx].tolist()
        
        return recommended_items
