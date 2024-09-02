
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

# TASK FOUR
# Neural Collaborative Filtering (NCF) Workflow :
**deep-learning branch**

# deep_learning_model.py changes
## Data Preparation
1. **Convert Interaction Matrix**: Convert the user-item interaction matrix into a dataframe and process it for use in training.
2. **Map Identifiers**: Map user and movie identifiers to numerical indices.
3. **Data Splitting**: Split the data into training and validation sets and prepare it for TensorFlow processing.

## Model Building
1. **Define Neural Network**:
   - **Embedding Layers**: Use embedding layers for users and movies.
   - **Flatten and Concatenate**: Flatten embeddings and concatenate them.
   - **Dense Hidden Layer**: Add a dense hidden layer.
   - **Output Layer**: Add an output layer.
2. **Compile Model**:
   - **Loss Function**: Use binary cross-entropy loss function.
   - **Optimizer**: Use Adam optimizer.

## Training
1. **Train Model**: Train the model on the training data.
2. **Early Stopping**: Use early stopping to avoid overfitting.

## Recommendation
1. **Predict Ratings**: Predict ratings for all items for a given user.
2. **Top N Items**: Return the top N items based on these predictions.

## Performance Evaluation
1. **Evaluate Model**:
   - **Mean Squared Error (MSE)**: Calculate MSE.
   - **Root Mean Squared Error (RMSE)**: Calculate RMSE.

## Feedback Incorporation
1. **Update Model**: Incorporate new user feedback into the model.
2. **Retrain and Adjust**: Retrain the model and adjust the datasets accordingly.

## Model Weight Summary
1. **Weights Summary**: Provide a summary of the model’s weights, including their shape and statistical properties.

# collaborative_filtering.py changes

# Matrix Factorization Workflow

## Data Preparation
1. **Convert Interaction Matrix**: Convert the user-item interaction matrix into a sparse matrix.
2. **Matrix Factorization**: Perform matrix factorization using Singular Value Decomposition (SVD) to decompose the matrix into latent factors.

## Recommendation
1. **Predict Ratings**: For a given user index, predict ratings for all items by reconstructing the matrix from the latent factors.
2. **Top N Recommendations**: Return the top N movie recommendations based on predicted scores.

## Performance Evaluation
1. **Evaluate Model**: Calculate the Mean Squared Error (MSE) between predicted ratings and actual ratings from a test dataset.

## Feedback Incorporation
1. **Update Matrix**: Update the user-item matrix with new user feedback.
2. **Recompute Model**: Recompute the model to reflect the updated feedback.

## Synthetic Data Generation
1. **Generate Synthetic Data**: Generate a synthetic user-item matrix for testing purposes.
2. **Create Random Ratings**: Create random ratings and convert them into a DataFrame with movie titles.

# content_based_filtering.py changes
# Content-Based Recommendation Workflow

## Data Preparation
1. **Content Aggregation**: Combine several text fields (description, cast, listed_in, director) into a single content field.
2. **TF-IDF Matrix**: Convert the aggregated content into a TF-IDF matrix, which represents the importance of words in the context of the dataset.

## Recommendation
1. **Similarity Computation**: Compute cosine similarity between movies based on the TF-IDF matrix.
2. **Retrieve Recommendations**: Find and sort movies similar to the given movie based on similarity scores.
3. **Feedback Adjustment**: Adjust recommendations if user feedback is available, modifying scores based on feedback to influence the recommendations.

## User Feedback Collection
1. **Prompt for Ratings**: Ask users to rate the recommended movies on a scale from 1 to 5.
2. **Update Feedback**: Update the internal feedback dictionary with the user’s ratings.

# hybrid_model.py changes
# Hybrid System Workflow

## Initialization
1. **Models Setup**: Initialize instances of three different filtering models:
   - `ContentBasedFiltering` for content-based recommendations.
   - `CollaborativeFiltering` for collaborative recommendations.
   - `NeuralCollaborativeFiltering` for deep learning-based recommendations.

## Model Preparation
1. **Data Preparation**: Call `prepare_data` on each model to ensure they are ready for generating recommendations.

## Recommendation Generation
1. **Retrieve Recommendations**:
   - Get top N movie recommendations from each filtering model:
     - Content-based
     - Collaborative
     - Neural collaborative

2. **Combine Recommendations**:
   - **Aggregating Recommendations**:
     - Content-based recommendations have a weight of `content_weight`.
     - Collaborative filtering recommendations have a weight of `collab_weight`.
     - Neural collaborative filtering recommendations have a weight of `ncf_weight`.
   - **Weight Aggregation**: Add weights for each movie based on its presence in the recommendations from different models.

3. **Sort and Select Top N**:
   - Sort the aggregated recommendations by weight.
   - Select the top N movies.

## User Feedback Collection
1. **Feedback Collection**:
   - Use the `collect_user_feedback` method from the `ContentBasedFiltering` model to gather user ratings for recommended movies.

# main.py changes

## Data Loading and Initialization
1. **Load Data**: Read a CSV file containing the data.
2. **Generate User-Item Matrix**: Create a synthetic user-item matrix and movie titles for recommendations.

## Model Initialization
1. **Content-Based Filtering**: Initialize the content-based recommendation model.
2. **Collaborative Filtering**: Initialize the collaborative filtering model with the synthetic user-item matrix.
3. **Hybrid Filtering**: Initialize the hybrid recommendation model combining different filtering approaches.
4. **Neural Collaborative Filtering**: Initialize and train a neural collaborative filtering model.

## User Input Functions
1. **Get Movie Title**: Prompt the user to enter a movie title and validate the input.
2. **Get User Index**: Prompt the user to enter a synthetic user index and validate the input.

## Recommendation Generation
1. **Content-Based Recommendations**: Generate movie recommendations based on content similarity.
2. **Collaborative Recommendations**: Generate movie recommendations based on user-item interactions.
3. **Hybrid Recommendations**: Combine content-based, collaborative, and neural recommendations using weights.

## Display and Visualizations
1. **Display Recommendations**: Print out the top movie recommendations for each filtering approach.
2. **Visualizations (Commented Out)**: Optionally generate and display visualizations for the recommendations (code for plotting is commented out).

## Feedback Collection and Model Update
1. **Collect Feedback**: Prompt the user to rate the recommended movies and save the feedback.
2. **Update Models**: Incorporate the user feedback into the collaborative filtering model and neural collaborative filtering model.

## Loop and Exit
1. **Main Program Loop**: Repeat the process of getting recommendations and collecting feedback until the user decides to exit.

## End of Program
1. **Exit Message**: Print a message indicating the end of the recommendation session.

# Additonal Files
## utils.py
# Function Definition: `save_feedback`

## Purpose
To save user feedback on movie recommendations to a CSV file.

## Arguments
- **feedback**: A dictionary where keys are movie titles and values are user ratings.
- **file_path**: The path to the CSV file where the feedback will be appended.

## Process
1. **Convert Feedback to DataFrame**:
   - Converts the feedback dictionary into a Pandas DataFrame with two columns: `movie` and `rating`.

2. **Save to CSV**:
   - Appends the DataFrame to a CSV file at the specified path.
   - If the file does not exist, it will be created.
   - The header is not written again if the file already exists.

3. **Logging**:
   - Logs a message indicating that the user feedback has been successfully saved.

# GUI Application: Hybrid Movie Recommendation System

## Data and Model Initialization
1. **Load Data**:
   - Loads movie data and generates synthetic user-item matrices.
2. **Model Initialization**:
   - Initializes and prepares content-based, collaborative filtering, and neural collaborative filtering models.

## Recommendation Generation
1. **User Input**:
   - Users input a movie title and a synthetic user index.
2. **Generate Recommendations**:
   - The application generates recommendations using content-based, collaborative, and hybrid methods.
3. **Display Recommendations**:
   - Recommendations are displayed in a structured format.

## Feedback Collection
1. **Rate Movies**:
   - Allows users to rate the recommended movies.
2. **Update Model**:
   - Updates the neural collaborative filtering model with the feedback.

## GUI Components
1. **Tkinter Interface**:
   - Uses Tkinter to create a full-screen application.
   - Includes input fields, buttons, and tables to display recommendations and feedback options.

# Overall changes
# Features of the Hybrid Movie Recommendation System

## 1. Data and Model Initialization
- **Load Movie Data**: Imports movie data and creates a synthetic user-item interaction matrix.
- **Initialize Models**:
  - **Content-Based Filtering**: Uses movie attributes for recommendations.
  - **Collaborative Filtering**: Utilizes user-item interaction data for recommendations.
  - **Neural Collaborative Filtering**: Applies deep learning techniques for recommendations.

## 2. Recommendation Generation
- **User Input**:
  - **Movie Title**: Users can enter a movie title to base recommendations on.
  - **User Index**: Users provide a synthetic user index for personalized recommendations.
- **Generate Recommendations**:
  - **Content-Based**: Recommends movies similar to the input title based on content features.
  - **Collaborative**: Suggests movies based on user-item interactions and preferences.
  - **Hybrid**: Combines content-based, collaborative, and neural methods to provide comprehensive recommendations.
- **Display Recommendations**:
  - Presents recommendations in a clear and organized format.

## 3. Feedback Collection
- **Rate Movies**:
  - Users can rate the recommended movies on a scale from 1 to 5.
- **Update Model**:
  - Integrates user feedback into the neural collaborative filtering model to refine future recommendations.

## 4. GUI Components
- **Tkinter Interface**:
  - **Full-Screen Application**: Provides a user-friendly, full-screen interface.
  - **Input Fields**: Allows users to input movie titles and user indices.
  - **Buttons**: Facilitates actions such as generating recommendations and submitting ratings.
  - **Tables**: Displays recommendations and feedback options in an organized manner.

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

