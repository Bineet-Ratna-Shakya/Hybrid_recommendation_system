# src/content_based_filtering.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFiltering:
    def __init__(self, data_path):
        """
        Initialize the ContentBasedFiltering model.
        
        :param data_path: Path to the preprocessed data CSV file.
        """
        self.data = pd.read_csv(data_path)
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def fit(self):
        """
        Fit the TF-IDF model and calculate the cosine similarity matrix based on the 'description' column.
        """
        # Step 1: TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['description'].fillna(''))
        
        # Step 2: Cosine Similarity
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def recommend(self, show_id, top_n=5):
        """
        Recommend items similar to the given show_id.
        
        :param show_id: ID of the show to base recommendations on.
        :param top_n: Number of similar items to recommend.
        :return: DataFrame of recommended items.
        """
        if self.similarity_matrix is None:
            raise ValueError("Model has not been fitted. Please call the fit method before recommending.")
        
        # Get the index of the item based on show_id
        item_index = self.data[self.data['show_id'] == show_id].index[0]
        
        # Get similarity scores for the specified item
        similarity_scores = self.similarity_matrix[item_index]
        
        # Get the indices of the top_n most similar items
        similar_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]
        
        # Return the most similar items
        return self.data.iloc[similar_indices]

# Example usage
if __name__ == "__main__":
    # Load and preprocess data
    data_path = '/Users/soul/Documents/hybrid_recommendation_system/data/Netflix-encoded-Data.csv'

    # Initialize and fit the content-based filtering model
    cbf_model = ContentBasedFiltering(data_path=data_path)
    cbf_model.fit()
    
    # Recommend similar items to the first item in the dataset
    recommendations = cbf_model.recommend(show_id='s1', top_n=5)  # Adjust show_id as needed
    print(recommendations[['show_id', 'title', 'description']])
