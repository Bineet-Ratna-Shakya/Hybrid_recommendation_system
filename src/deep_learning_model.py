import numpy as np
import pandas as pd
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NeuralCollaborativeFiltering:
    def __init__(self, user_item_matrix, movie_titles=None, embedding_dim=20, batch_size=32):
        self.user_item_matrix = user_item_matrix
        self.movie_titles = movie_titles
        self.num_users = user_item_matrix.shape[0]
        self.num_movies = user_item_matrix.shape[1]
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.batch_size = batch_size
        self.model = self.build_model()
        self.prepare_data()

    def prepare_data(self):
        interactions = self.user_item_matrix.stack().reset_index()
        interactions.columns = ['user', 'movie', 'rating']
        interactions['rating'] = interactions['rating'].astype(np.float32)
        self.user_mapping = {user: idx for idx, user in enumerate(interactions['user'].unique())}
        self.movie_mapping = {movie: idx for idx, movie in enumerate(interactions['movie'].unique())}
        self.interactions = interactions
        user_indices = interactions['user'].map(self.user_mapping).values
        movie_indices = interactions['movie'].map(self.movie_mapping).values
        ratings = interactions['rating'].values
        self.user_indices = np.array(user_indices)
        self.movie_indices = np.array(movie_indices)
        self.ratings = np.array(ratings)
        X_train_user, X_val_user, X_train_movie, X_val_movie, y_train, y_val = train_test_split(
            self.user_indices, self.movie_indices, self.ratings, test_size=0.1, random_state=42
        )
        self.train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_user, X_train_movie), y_train)).batch(self.batch_size)
        self.val_dataset = tf.data.Dataset.from_tensor_slices(((X_val_user, X_val_movie), y_val)).batch(self.batch_size)
        logging.info("Neural Collaborative Filtering data prepared.")

    def build_model(self):
        user_input = Input(shape=(1,), name='user')
        movie_input = Input(shape=(1,), name='movie')
        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.embedding_dim, 
                                   embeddings_regularizer=l2(0.01))(user_input)
        movie_embedding = Embedding(input_dim=self.num_movies, output_dim=self.embedding_dim, 
                                    embeddings_regularizer=l2(0.01))(movie_input)
        user_vector = Flatten()(user_embedding)
        movie_vector = Flatten()(movie_embedding)
        concat = Concatenate()([user_vector, movie_vector])
        hidden = Dense(64, activation='relu')(concat)
        output = Dense(1, activation='sigmoid')(hidden)
        model = Model(inputs=[user_input, movie_input], outputs=output)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        logging.info("Neural Collaborative Filtering model built.")
        return model

    def train(self, epochs=5):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.val_dataset, 
                       callbacks=[early_stopping], verbose=1)
        logging.info("Neural Collaborative Filtering model trained.")

    def recommend(self, user_idx, top_n=10):
        movie_indices = np.arange(self.num_movies)
        user_indices = np.full_like(movie_indices, user_idx)
        predictions = self.model.predict([user_indices, movie_indices])
        recommended_idx = np.argsort(predictions.flatten())[::-1][:top_n]
        recommended_items = self.user_item_matrix.columns[recommended_idx].tolist()
        return recommended_items

    def evaluate_performance(self):
        interactions = self.user_item_matrix.stack().reset_index()
        interactions.columns = ['user', 'movie', 'rating']
        user_indices = interactions['user'].map(self.user_mapping).values
        movie_indices = interactions['movie'].map(self.movie_mapping).values
        ratings = interactions['rating'].values
        predictions = self.model.predict([user_indices, movie_indices])
        mse = mean_squared_error(ratings, predictions)
        rmse = np.sqrt(mse)
        logging.info(f"Neural Collaborative Filtering model performance: RMSE = {rmse:.4f}")

    def incorporate_feedback(self, feedback):
        """
        Incorporate user feedback into the neural collaborative filtering model.
        
        Args:
            feedback (dict): Dictionary with movie titles as keys and feedback ratings as values.
        """
        feedback_data = []
        for movie_title, rating in feedback.items():
            if movie_title in self.movie_mapping:
                movie_idx = self.movie_mapping[movie_title]
                for user in self.user_mapping:
                    user_idx = self.user_mapping[user]
                    feedback_data.append((user_idx, movie_idx, rating))
        
        if feedback_data:
            feedback_df = pd.DataFrame(feedback_data, columns=['user', 'movie', 'rating'])
            feedback_user_indices = feedback_df['user'].values
            feedback_movie_indices = feedback_df['movie'].values
            feedback_ratings = feedback_df['rating'].values
            
            # Update training data
            self.user_indices = np.concatenate([self.user_indices, feedback_user_indices])
            self.movie_indices = np.concatenate([self.movie_indices, feedback_movie_indices])
            self.ratings = np.concatenate([self.ratings, feedback_ratings])
            
            # Prepare updated datasets
            X_train_user, X_val_user, X_train_movie, X_val_movie, y_train, y_val = train_test_split(
                self.user_indices, self.movie_indices, self.ratings, test_size=0.1, random_state=42
            )
            self.train_dataset = tf.data.Dataset.from_tensor_slices(((X_train_user, X_train_movie), y_train)).batch(self.batch_size)
            self.val_dataset = tf.data.Dataset.from_tensor_slices(((X_val_user, X_val_movie), y_val)).batch(self.batch_size)
            logging.info("User feedback incorporated into the model data.")
            
            # Retrain the model with updated data
            self.train()
    
    def print_model_weight_summary(self):
        print("Model Weights Summary:")
        for layer in self.model.layers:
            weights = layer.get_weights()
            if weights:
                print(f"Layer: {layer.name}")
                for weight in weights:
                    weight_array = np.array(weight)
                    print(f"Shape: {weight_array.shape}")
                    print(f"Mean: {weight_array.mean():.4f}")
                    print(f"Std Dev: {weight_array.std():.4f}")
                    print(f"Min: {weight_array.min():.4f}")
                    print(f"Max: {weight_array.max():.4f}")
                print("-" * 40)