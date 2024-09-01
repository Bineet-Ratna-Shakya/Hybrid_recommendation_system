
# Hybrid Recommendation System

This project implements a hybrid recommendation system that combines collaborative filtering (using Singular Value Decomposition, SVD) and content-based filtering (using TF-IDF and Cosine Similarity) to provide personalized movie recommendations. The repository has two branches:

- **main branch**: Contains the core hybrid model using collaborative filtering and content-based filtering.
- **deep-learning branch**: Adds an advanced deep learning-based model (e.g., Neural Collaborative Filtering or Autoencoders) to enhance the hybrid system.

## Table of Contents

- [Getting Started](#getting-started)
- [Features](#features)
- [File Structure](#file-structure)
- [How To Run](#how-to-run)
- [Logic Overview](#logic-overview)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Content-Based Filtering](#content-based-filtering)
  - [Hybrid Model](#hybrid-model)
- [Built With](#built-with)
- [Usage](#usage)
- [Branches](#branches)


## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.7+
- pip (Python package installer)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/hybrid-recommendation-system.git
    cd hybrid-recommendation-system
    ```

2. Switch to the desired branch:

    - For the hybrid model only:

        ```bash
        git checkout main
        ```

    - For the deep learning enhanced model:

        ```bash
        git checkout deep-learning
        ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:

    ```bash
    python main.py
    ```

## How To Run
Usage
Prepare your dataset:

Make sure you have a dataset in CSV format. The file should be located at /your_path/hybrid_recommendation_system/data/Netflix-encoded-Data.csv. Adjust the path in the script if necessary.

Run the recommendation system:

Execute the main.py script to start the recommendation system:

```
python main.py
```
Interact with the system:

Enter a movie title when prompted.
The system will display recommendations based on content-based filtering, collaborative filtering, and hybrid filtering.
You can continue to get recommendations for other movies or exit the program.

```
WELCOME TO THE HYBRID RECOMMENDATION SYSTEM
========================================

Enter a movie title (or type 'exit' to quit): Inception

========================================
GENERATING RECOMMENDATIONS...
========================================

--- Content-Based Recommendations ---
1. The Dark Knight
2. Interstellar
3. Memento
...

--- Collaborative Recommendations ---
1. Tenet
2. The Prestige
3. Insomnia
...

--- Hybrid Recommendations ---
1. Dunkirk
2. Shutter Island
3. The Matrix
...

Do you want to get recommendations for another movie? (yes/no): no
Thank you for using the recommendation system. Goodbye!
```
Notes
Ensure your dataset is properly encoded and preprocessed as expected by the scripts.
You may need to adjust model parameters based on your specific requirements and dataset.

## Features

- **Collaborative Filtering** (`collaborative_filtering.py`): Implements Singular Value Decomposition for matrix factorization of user-item interaction data.
- **Content-Based Filtering** (`content_based_filtering.py`): Provides recommendations based on TF-IDF vectorization and cosine similarity.
- **Data Preprocessing** (`data_preprocessing.py`): Handles data cleaning, missing value handling, and normalization.
- **Hybrid Model** (`hybrid_model.py`): Combines collaborative and content-based filtering results for enhanced recommendations.
- **Deep Learning Integration** (`deep_learning_model.py` in `deep-learning` branch): Adds advanced models like Neural Collaborative Filtering or Autoencoders.

## File Structure

- `collaborative_filtering.py`: Contains the collaborative filtering implementation using matrix factorization (e.g., SVD).
- `content_based_filtering.py`: Handles content-based filtering using TF-IDF and cosine similarity.
- `data_preprocessing.py`: Manages data preprocessing tasks such as cleaning and normalization.
- `deep_learning_model.py`: Implements the deep learning-based model (only in `deep-learning` branch).
- `hybrid_model.py`: Integrates both collaborative and content-based models into a hybrid recommendation system.
- `main.py`: The main script to run the recommendation system.

## Logic Overview
# Synthetic User Generation in Collaborative Filtering

Synthetic User Generation is a technique used to enhance the performance of Collaborative Filtering, especially when there is limited data or sparse interactions (i.e., few ratings per user). The idea is to generate synthetic users or ratings to augment the existing dataset, making the recommendation model more robust.

## How It Works

### Problem with Sparsity:
- In a typical user-item matrix, many users may have rated only a few items. This leads to sparsity, which makes it difficult to learn meaningful patterns for recommendation.

### Generating Synthetic Users:
- **Synthetic users** are artificial entities created to mimic potential user behaviors. These users are generated based on certain rules or by sampling from the existing distribution of user ratings.
- Synthetic users can be designed to represent a specific segment of users (e.g., movie enthusiasts who rate high, critics who rate low, etc.).

### Augmenting the Dataset:
- By adding these synthetic users, the density of the user-item matrix increases, allowing the model to learn more robust patterns.
- For example, synthetic ratings can be generated for items that are underrepresented (few users have rated them) or for new items that haven't been rated yet.

### Methodology for Creating Synthetic Users:
1. **Use clustering algorithms** like K-Means to group existing users based on their rating patterns. Each cluster represents a unique user behavior profile.
2. **Generate synthetic users** by sampling from these clusters or by creating new users who rate according to the centroid of each cluster.
3. **Assign these synthetic users** with ratings based on average cluster behavior, noise addition, or domain-specific rules (e.g., favoring certain genres or directors).

### Benefits:
- This approach helps mitigate cold-start problems (e.g., new users with no ratings or new items with few ratings).
- Improves model training by providing more data points, leading to better generalization and accuracy of recommendations.

# 1. Collaborative Filtering

Collaborative Filtering is a recommendation approach that relies on user-item interactions to predict user preferences. In this case, it uses Singular Value Decomposition (SVD) to decompose the user-item matrix, which represents how users rate or interact with different movies.

### How It Works

#### Data Preparation:
- The input is a user-item matrix where rows represent users and columns represent movies. The matrix values are user ratings or some form of interaction score (e.g., how much a user likes or watches a movie).
- This matrix is converted to a sparse format using `csr_matrix` from `scipy.sparse` to handle the typically large size and sparsity (many missing values) efficiently.
- Singular Value Decomposition (SVD) is applied to the matrix. SVD decomposes the matrix M into three matrices: U, Σ (sigma), and Vt.
  - **U:** Represents the relationship between users and the latent factors. It is a user matrix where each row represents a user, and each column represents a latent factor.
  - **Σ (sigma):** A diagonal matrix containing singular values that capture the importance of each latent factor.
  - **Vt:** Represents the relationship between movies and latent factors. It is a transposed movie matrix where each row represents a movie, and each column represents a latent factor.
- The value of k determines the number of latent factors to retain (a smaller number to reduce complexity and noise).

#### Recommendation Generation:
- To generate recommendations for a specific user, the algorithm reconstructs the user-item matrix using the formula:

  \[
  \text{Predicted Scores} = U \cdot Σ \cdot V^T
  \]

- This multiplication gives the predicted score for each movie for the user.
- The movies are sorted based on their predicted scores for the user, and the top N movies with the highest scores are recommended.


**Code snippet**:

```python
# Collaborative Filtering using SVD
class CollaborativeFiltering:
    def __init__(self, user_item_matrix, movie_titles):
        self.user_item_matrix = user_item_matrix
        self.movie_titles = movie_titles
        self.U, self.sigma, self.Vt = None, None, None

    def prepare_data(self):
        self.user_item_matrix = self.user_item_matrix.astype(float)
        sparse_matrix = csr_matrix(self.user_item_matrix)
        num_users, num_movies = sparse_matrix.shape
        k = min(num_users - 1, num_movies - 1, 10)  
        self.U, self.sigma, self.Vt = svds(sparse_matrix, k=k)
        self.sigma = np.diag(self.sigma)

    def recommend(self, user_idx, top_n=10):
        if self.U is None or self.sigma is None or self.Vt is None:
            self.prepare_data()
        predicted_scores = np.dot(np.dot(self.U, self.sigma), self.Vt)[user_idx, :]
        recommended_idx = np.argsort(predicted_scores)[::-1]
        recommended_items = self.movie_titles[recommended_idx[:top_n]].tolist()
        return recommended_items
```

## 2. Content-Based Filtering

Content-Based Filtering recommends movies based on their features, such as the movie's description, cast, genres, and director. It uses TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to find similar items.

### How It Works

#### Data Preparation:
- The content features (e.g., description, cast, genres, director) of each movie are combined into a single string. This step ensures that all relevant information about a movie is considered when calculating its similarity to other movies.
- The TF-IDF Vectorizer converts the combined content strings into a TF-IDF matrix:
  - TF-IDF calculates the importance of words in the context of each movie's content. It considers both the frequency of a word in a specific movie and its rarity across all movies.
  - This matrix is high-dimensional, with each row representing a movie and each column representing a word in the movie descriptions.

#### Recommendation Generation:
- **Cosine Similarity** is computed between the target movie's TF-IDF vector and all other movies' vectors:
  - Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space. It provides a metric of how similar two movies are based on their content.
- Movies are sorted based on their similarity scores to the target movie. The top N most similar movies are recommended.


**Code snippet**:

```python
# Content-Based Filtering using TF-IDF and Cosine Similarity
class ContentBasedFiltering:
    def __init__(self, data):
        self.data = data
        self.tfidf_matrix = None
    
    def prepare_data(self):
        self.data['content'] = self.data['description'].fillna('') + ' ' +                                self.data['cast'].fillna('') + ' ' +                                self.data['listed_in'].fillna('') + ' ' +                                self.data['director'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.data['content'])
    
    def recommend(self, movie_title, top_n=10):
        if self.tfidf_matrix is None:
            self.prepare_data()
        movie_idx = self.data[self.data['title'].str.lower() == movie_title.lower()].index[0]
        cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        similar_movies = list(enumerate(cosine_sim[movie_idx]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        recommended_movies = [self.data['title'].iloc[i[0]] for i in similar_movies[1:top_n+1]]
        return recommended_movies
```

## 3. Hybrid Model

The Hybrid Model combines both Collaborative Filtering and Content-Based Filtering approaches to leverage the strengths of both methods and provide more comprehensive recommendations.

### How It Works

#### Combining the Approaches:
- **Content-Based Recommendations:** Uses the Content-Based Filtering method to generate recommendations based on the similarity of content to a given movie.
- **Collaborative Recommendations:** Uses the Collaborative Filtering method to generate recommendations based on user interactions.
- The model merges both recommendation sets to provide a list that captures both user preferences (from collaborative filtering) and content similarities (from content-based filtering).

#### Recommendation Generation:
- The hybrid model calls both the recommend methods from `ContentBasedFiltering` and `CollaborativeFiltering`.
- It merges these recommendations into a single list (with or without additional weighting or prioritization logic). The merged list is truncated to the top N recommendations.

# Deep Learning Model
# Neural Collaborative Filtering (NCF) Logic

Neural Collaborative Filtering (NCF) uses neural networks to learn user and item embeddings for making personalized recommendations. Here's a brief overview of the logic and important code points:

## Key Logic

1. **Data Preparation:**
   - Convert the user-item interaction matrix to a DataFrame.
   - Encode user and movie IDs to numeric values.

2. **Model Building:**
   - **Inputs:**
     - `user_input`: Input for user IDs.
     - `movie_input`: Input for movie IDs.
   - **Embedding Layers:**
     - `user_embedding`: Maps user IDs to dense vectors.
     - `movie_embedding`: Maps movie IDs to dense vectors.
   - **Concatenation:**
     - Combine user and movie embeddings.
   - **Hidden Layers:**
     - Dense layer with ReLU activation.
   - **Output Layer:**
     - Dense layer with sigmoid activation for binary rating prediction.

4. **Recommendation Generation:**
   - Predict ratings for all movies for a given user.
   - Return top-N movies based on predicted ratings.

**Code snippet**:

```python
# Hybrid Filtering combining Collaborative and Content-Based Filtering
class HybridFiltering:
    def __init__(self, data, user_item_matrix, movie_titles):
        self.data = data
        self.user_item_matrix = user_item_matrix
        self.movie_titles = movie_titles
        self.collab_filter = CollaborativeFiltering(user_item_matrix, movie_titles)
        self.content_filter = ContentBasedFiltering(data)
    
    def recommend(self, movie_title, user_idx=0, top_n=10):
        content_recommendations = self.content_filter.recommend(movie_title, top_n=top_n)
        collab_recommendations = self.collab_filter.recommend(user_idx=user_idx, top_n=top_n)
        hybrid_recommendations = list(set(content_recommendations + collab_recommendations))[:top_n]
        return hybrid_recommendations
```
### Summary of Logic:

- **Collaborative Filtering:** Learns from user behavior (e.g., movie ratings) to predict preferences for unseen movies.
- **Content-Based Filtering:** Utilizes movie metadata (e.g., description, cast) to find movies similar to what a user already likes.
- **Hybrid Model:** Integrates both methods to provide a more robust recommendation by capturing both user preferences and movie content similarities. This approach helps in scenarios where either collaborative or content-based filtering alone might fail (e.g., when a user has rated very few movies or when new movies lack sufficient user interaction data).

## Built With

- **Python**: The core programming language.
- **scikit-learn**: For TF-IDF vectorization and cosine similarity.
- **Surprise**: For collaborative filtering and matrix factorization.
- **TensorFlow/Keras**: For deep learning models (optional).

## Usage

1. Prepare your data: Place your dataset in the `data/` folder in the required format (e.g., CSV, JSON).

2. Run the hybrid model: Execute the `main.py` script to generate recommendations:

    ```bash
    python main.py
    ```

3. Adjust parameters directly in the scripts: Modify the parameters in the respective script files (`collaborative_filtering.py`, `content_based_filtering.py`, etc.) to customize the model settings and data paths.

## Branches

- **main branch**: Implements the hybrid recommendation system using collaborative filtering (SVD) and content-based filtering (TF-IDF and Cosine Similarity).
- **deep-learning branch**: Enhances the hybrid model with deep learning techniques like Neural Collaborative Filtering (NCF) or Autoencoders.

# References

## General Recommender System Resources

- [IJISRT Paper on Recommender Systems](https://ijisrt.com/assets/upload/files/IJISRT22APR1053_(1).pdf)
- [Comprehensive Guide to Recommendation Engines in Python](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/)
- [Recommender Systems Tutorial on DataCamp](https://www.datacamp.com/tutorial/recommender-systems-python)
- [Movie Recommendation System Dataset on Kaggle](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)
- [How to Build a Movie Recommendation System Based on Collaborative Filtering](https://www.freecodecamp.org/news/how-to-build-a-movie-recommendation-system-based-on-collaborative-filtering/)
- [Deep Learning-Based Recommender Systems on Kaggle](https://www.kaggle.com/code/jamesloy/deep-learning-based-recommender-systems)
- [Content-Based and Collaborative Filtering on Kaggle](https://www.kaggle.com/code/indralin/try-content-based-and-collaborative-filtering)
- [Building a Hybrid Content-Collaborative Movie Recommender Using Deep Learning](https://towardsdatascience.com/building-a-hybrid-content-collaborative-movie-recommender-using-deep-learning-22f99c838f55)

## Data Cleaning

- [Data Cleaning with Pandas and Python - Real Python](https://realpython.com/python-data-cleaning-numpy-pandas/)
- [Data Cleaning in Python - Towards Data Science](https://towardsdatascience.com/the-art-of-data-cleaning-in-python-dbc944b8f8e2)
- [Handling Missing Data - Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)

## Data Normalization

- [Data Normalization Techniques - Towards Data Science](https://towardsdatascience.com/data-normalization-in-machine-learning-395fdec69d02)

## Collaborative Filtering

- [Collaborative Filtering Model - Towards Data Science](https://towardsdatascience.com/building-a-collaborative-filtering-model-from-scratch-in-python-7ff992a12d9e)
- [Matrix Factorization Algorithms - Surprise Library](https://surprise.readthedocs.io/en/stable/matrix_factorization.html)
- [Matrix Factorization for Recommender Systems - Towards Data Science](https://towardsdatascience.com/matrix-factorization-for-recommender-systems-3b5922d3c5c8)
- [SVD for Collaborative Filtering - Medium](https://medium.com/analytics-vidhya/collaborative-filtering-using-singular-value-decomposition-in-python-743d1577f7a2)

## Content-Based Filtering

- [Content-Based Recommender Systems - Towards Data Science](https://towardsdatascience.com/building-a-content-based-book-recommendation-engine-for-an-amazon-like-bookstore-d9b78d7de93)
- [TF-IDF Vectorizer - Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [Feature Extraction with TF-IDF - Real Python](https://realpython.com/python-keras-text-classification/)
- [Cosine Similarity - Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

## Hybrid Recommendation Systems

- [Hybrid Content-Collaborative Recommender System - Towards Data Science](https://towardsdatascience.com/building-a-hybrid-content-collaborative-movie-recommender-using-deep-learning-22f99c838f55)
- [Hybrid Recommendation Engine - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/)
- [Combining Models for Hybrid Recommendations - Medium](https://medium.com/@sarojkarna/hybrid-recommender-systems-using-python-7f74a7d5e4b2)

## Evaluation and Visualization

- [Evaluation Metrics for Recommender Systems - Towards Data Science](https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093)
- [A Practical Guide to Evaluating Recommender Systems - Medium](https://medium.com/@shaileshkramcharran/which-evaluation-metrics-should-i-use-to-evaluate-recommendation-systems-7e2cf1301d0c)

## Optional Features

- [Neural Collaborative Filtering (PDF)](https://arxiv.org/pdf/1708.05031.pdf)
- [Neural Collaborative Filtering with Keras - Towards Data Science](https://towardsdatascience.com/neural-collaborative-filtering-and-implementation-with-keras-5a8a3a5a7ab2)
- [Autoencoders for Collaborative Filtering - Medium](https://medium.com/analytics-vidhya/autoencoders-collaborative-filtering-for-recommendation-systems-d4f6bff6d10a)

