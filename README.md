
# Hybrid Recommendation System

This project implements a hybrid recommendation system that combines collaborative filtering (using Singular Value Decomposition, SVD) and content-based filtering (using TF-IDF and Cosine Similarity) to provide personalized movie recommendations. The repository has two branches:

- **main branch**: Contains the core hybrid model using collaborative filtering and content-based filtering.
- **deep-learning branch**: Adds an advanced deep learning-based model (e.g., Neural Collaborative Filtering or Autoencoders) to enhance the hybrid system.

## Table of Contents

- [Getting Started](#getting-started)
- [Features](#features)
- [File Structure](#file-structure)
- [Logic Overview](#logic-overview)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Content-Based Filtering](#content-based-filtering)
  - [Hybrid Model](#hybrid-model)
- [Built With](#built-with)
- [Usage](#usage)
- [Branches](#branches)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

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

### Collaborative Filtering

The Collaborative Filtering approach is based on Singular Value Decomposition (SVD). It uses a user-item interaction matrix to make recommendations.

- **Data Preparation**: Converts the user-item matrix to a sparse format and performs SVD.
- **Recommendation**: Predicts movie scores for users by reconstructing the user-item matrix using the decomposed matrices. Recommends the top N movies with the highest predicted scores.

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

### Content-Based Filtering

The Content-Based Filtering approach uses TF-IDF vectorization and cosine similarity to recommend movies similar to a given movie.

- **Data Preparation**: Combines multiple text-based features (e.g., description, cast, genres) into a single content string for each movie. Computes the TF-IDF matrix for these features.
- **Recommendation**: Calculates the cosine similarity between the TF-IDF vectors of the target movie and all other movies. Recommends the top N most similar movies.

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

### Hybrid Model

The Hybrid Model combines both collaborative and content-based filtering approaches to provide a more comprehensive recommendation system.

- **Recommendation**: Merges the results from both collaborative and content-based methods. This approach captures both user preferences and content similarities to improve recommendation quality.

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

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact [Your Name] at [your.email@example.com].
