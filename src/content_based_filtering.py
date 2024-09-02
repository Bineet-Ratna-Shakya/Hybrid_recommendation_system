import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

class ContentBasedFiltering:
    def __init__(self, data):
        self.data = data
        self.tfidf_matrix = None
        self.feedback = {}

    def prepare_data(self):
        self.data['content'] = self.data['description'].fillna('') + ' ' + \
                               self.data['cast'].fillna('') + ' ' + \
                               self.data['listed_in'].fillna('') + ' ' + \
                               self.data['director'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.data['content'])
        logging.info("Content-Based Filtering model prepared.")

    def recommend(self, movie_title, top_n=10):
        if self.tfidf_matrix is None:
            self.prepare_data()
        movie_idx = self.data[self.data['title'].str.lower() == movie_title.lower()].index[0]
        cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        similar_movies = list(enumerate(cosine_sim[movie_idx]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        recommended_movies = [self.data['title'].iloc[i[0]] for i in similar_movies[1:top_n+1]]
        
        # Adjust recommendations based on feedback
        if self.feedback:
            feedback_adjusted = {movie: score + self.feedback.get(movie, 0) for movie, score in zip(recommended_movies, range(1, top_n+1))}
            sorted_feedback = sorted(feedback_adjusted.items(), key=lambda x: x[1], reverse=True)
            recommended_movies = [movie for movie, _ in sorted_feedback[:top_n]]
        
        return recommended_movies

    def collect_user_feedback(self, recommended_movies):
        print("\nPlease rate the following recommendations (1-5):")
        feedback = {}
        for movie in recommended_movies:
            while True:
                try:
                    rating = int(input(f"{movie}: "))
                    if 1 <= rating <= 5:
                        feedback[movie] = rating
                        break
                    else:
                        print("Please enter a rating between 1 and 5.")
                except ValueError:
                    print("Invalid input. Please enter an integer between 1 and 5.")
        self.feedback.update(feedback)
        return feedback

