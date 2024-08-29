import pandas as pd
from hybrid_model import HybridModel

def main():
    # Load the dataset
    file_path = '/Users/soul/Documents/hybrid_recommendation_system/data/Netflix-encoded-Data.csv'
    data = pd.read_csv(file_path)
    
    # Initialize the Hybrid Model
    hybrid_model = HybridModel(data)
    
    # Ask the user for a movie name
    movie_name = input("Enter a movie name: ").strip()
    
    # Get recommendations
    recommendations = hybrid_model.recommend(movie_name, top_n=10)
    
    # Display recommendations
    if recommendations:
        print("\nTop 10 movie recommendations for '{}':".format(movie_name))
        for idx, movie in enumerate(recommendations, 1):
            print(f"{idx}. {movie}")
    else:
        print(f"No recommendations found for the movie '{movie_name}'.")

if __name__ == "__main__":
    main()
