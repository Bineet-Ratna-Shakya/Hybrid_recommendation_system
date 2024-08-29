import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFiltering:
    """
    A class to implement content-based movie recommendation using TF-IDF and cosine similarity.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing movie data with columns such as 'description', 'cast', 
                             'listed_in', and 'director'.
        tfidf_matrix (scipy.sparse.csr.csr_matrix, optional): TF-IDF matrix computed from movie content.
    """
    
    def __init__(self, data):
        """
        Initializes the ContentBasedFiltering with the provided data and prepares the TF-IDF matrix.
        
        Args:
            data (pd.DataFrame): DataFrame containing movie data with relevant columns.
        """
        self.data = data
        self.tfidf_matrix = None
    
    def prepare_data(self):
        """
        Prepares the data by combining relevant text features into a single string for each movie and 
        computes the TF-IDF matrix for these combined features.
        """
        self.data['content'] = self.data['description'].fillna('') + ' ' + \
                               self.data['cast'].fillna('') + ' ' + \
                               self.data['listed_in'].fillna('') + ' ' + \
                               self.data['director'].fillna('')
        
        #TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.data['content'])
    
    def recommend(self, movie_title, top_n=10):
        """
        Generates movie recommendations based on content similarity to the given movie title.
        
        Args:
            movie_title (str): The movie title to base the recommendations on.
            top_n (int, optional): Number of top recommendations to return. Defaults to 10.
        
        Returns:
            list: List of top-n recommended movie titles based on content similarity.
        """
        if self.tfidf_matrix is None:
            self.prepare_data()
        
        movie_idx = self.data[self.data['title'].str.lower() == movie_title.lower()].index[0]
        
        cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # similar movies
        similar_movies = list(enumerate(cosine_sim[movie_idx]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        
        # Recommend top N movies
        recommended_movies = [self.data['title'].iloc[i[0]] for i in similar_movies[1:top_n+1]]
        
        return recommended_movies
