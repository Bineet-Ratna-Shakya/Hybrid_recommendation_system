import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFiltering:
    def __init__(self, data):
        self.data = data
        self.tfidf_matrix = None
    
    def prepare_data(self):
        # Combine relevant features into a single string for each movie
        self.data['content'] = self.data['description'].fillna('') + ' ' + \
                               self.data['cast'].fillna('') + ' ' + \
                               self.data['listed_in'].fillna('') + ' ' + \
                               self.data['director'].fillna('')
        
        # Initialize TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.data['content'])
    
    def recommend(self, movie_title, top_n=10):
        if self.tfidf_matrix is None:
            self.prepare_data()
        
        # Get index of the movie
        movie_idx = self.data[self.data['title'].str.lower() == movie_title.lower()].index[0]
        
        # Compute similarity based on content
        cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Get similar movies
        similar_movies = list(enumerate(cosine_sim[movie_idx]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        
        # Recommend top N movies
        recommended_movies = [self.data['title'].iloc[i[0]] for i in similar_movies[1:top_n+1]]
        
        return recommended_movies
