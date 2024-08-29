import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, data):
        self.data = data
        self.genre_matrix = None
        self.director_matrix = None
    
    def prepare_data(self):
        # Create a matrix for genres
        genre_data = self.data['listed_in'].str.get_dummies(sep=', ')
        director_data = self.data['director'].str.get_dummies(sep=', ')
        
        # Combine the matrices
        self.genre_matrix = genre_data
        self.director_matrix = director_data
        
    def get_similarity(self, matrix):
        # Calculate the cosine similarity between items
        return cosine_similarity(matrix)
    
    def recommend(self, movie_title, top_n=10):
        if self.genre_matrix is None or self.director_matrix is None:
            self.prepare_data()
        
        # Get index of the movie
        movie_idx = self.data[self.data['title'].str.lower() == movie_title.lower()].index[0]
        
        # Compute similarity based on genres and directors
        genre_sim = self.get_similarity(self.genre_matrix)
        director_sim = self.get_similarity(self.director_matrix)
        
        # Combine the similarities
        combined_sim = (genre_sim + director_sim) / 2
        
        # Get similar movies
        similar_movies = list(enumerate(combined_sim[movie_idx]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        
        # Recommend top N movies
        recommended_movies = [self.data['title'].iloc[i[0]] for i in similar_movies[1:top_n+1]]
        
        return recommended_movies
