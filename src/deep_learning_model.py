import numpy as np
import pandas as pd
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.regularizers import l2
from sklearn.model_selection import train_test_split
#deep learning
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
        """
        interactions = self.user_item_matrix.stack().reset_index()
        interactions.columns = ['user', 'movie', 'rating']
        
        # Normalize ratings (if needed)
        interactions['rating'] = interactions['rating'].astype(np.float32)
        
        # Convert user and movie indices to numeric encodings
        self.user_mapping = {user: idx for idx, user in enumerate(interactions['user'].unique())}
        self.movie_mapping = {movie: idx for idx, movie in enumerate(interactions['movie'].unique())}
        
        # Store prepared data
        self.interactions = interactions
        user_indices = interactions['user'].map(self.user_mapping).values
        movie_indices = interactions['movie'].map(self.movie_mapping).values
        ratings = interactions['rating'].values
        
        # Convert to numpy arrays if not already
        self.user_indices = np.array(user_indices)
        self.movie_indices = np.array(movie_indices)
        self.ratings = np.array(ratings)

        # Check the shapes again
        print(f"Shape of user_indices: {len(self.user_indices)}")
        print(f"Shape of movie_indices: {len(self.movie_indices)}")
        print(f"Shape of ratings: {len(self.ratings)}")
        
        # Split data into training and validation sets
        X_train_user, X_val_user, X_train_movie, X_val_movie, y_train, y_val = train_test_split(
            self.user_indices, self.movie_indices, self.ratings, test_size=0.1, random_state=42
        )
        
        # Combine the training and validation data into tf.data.Dataset
        self.train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_user, X_train_movie), y_train)).batch(self.batch_size)
        self.val_dataset = tf.data.Dataset.from_tensor_slices(((X_val_user, X_val_movie), y_val)).batch(self.batch_size)

    def __init__(self, user_item_matrix, movie_titles=None, embedding_dim=20, batch_size=32):
        """
        Initializes the NeuralCollaborativeFiltering with user-item interaction matrix and embedding dimension.
        
        Args:
            user_item_matrix (pd.DataFrame): DataFrame containing the user-item interaction matrix.
            embedding_dim (int, optional): Dimension of the embedding vectors. Defaults to 20.
            batch_size (int, optional): Batch size for training. Defaults to 32.
        """
        self.user_item_matrix = user_item_matrix
        self.movie_titles = movie_titles
        self.num_users = user_item_matrix.shape[0]
        self.num_movies = user_item_matrix.shape[1]
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.model = self.build_model()
        self.prepare_data()

    def build_model(self):
        """
        Builds the Neural Collaborative Filtering model using Keras.
        
        Returns:
            tf.keras.Model: Compiled NCF model.
        """
        user_input = Input(shape=(1,), name='user')
        movie_input = Input(shape=(1,), name='movie')
        
        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.embedding_dim, 
                                   embeddings_regularizer=l2(0.01))(user_input)
        movie_embedding = Embedding(input_dim=self.num_movies, output_dim=self.embedding_dim, 
                                    embeddings_regularizer=l2(0.01))(movie_input)
        
        user_vector = Flatten()(user_embedding)
        movie_vector = Flatten()(movie_embedding)
        
        concat = Concatenate()([user_vector, movie_vector])
        hidden = Dense(64, activation='relu')(concat)  # Simplified model
        output = Dense(1, activation='sigmoid')(hidden)
        
        model = Model(inputs=[user_input, movie_input], outputs=output)
        
        optimizer = Adam(learning_rate=0.001)  # Optimizer with a learning rate
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def train(self, epochs=5):
        """
        Trains the NCF model using the user-item interaction matrix.
        
        Args:
            epochs (int, optional): Number of training epochs. Defaults to 5.
        """
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.val_dataset, 
                       callbacks=[early_stopping], verbose=1)

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
        
        # Batch prediction for faster performance
        user_indices = np.full_like(movie_indices, user_idx)
        predictions = self.model.predict([user_indices, movie_indices])
        
        # Sort and get the indices of the top N recommendations
        recommended_idx = np.argsort(predictions.flatten())[::-1][:top_n]
        
        # Map indices to movie titles
        recommended_items = self.user_item_matrix.columns[recommended_idx].tolist()
        
        return recommended_items
