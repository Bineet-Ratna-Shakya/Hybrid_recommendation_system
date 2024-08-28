# main.py
from collaborative_filtering import CollaborativeFiltering
from content_based_filtering import ContentBasedFiltering
from hybrid_model import HybridModel

# Initialize models with the given dataset path
data_path = "/Users/soul/Documents/hybrid_recommendation_system/data/Netflix-encoded-Data.csv"
collab_model = CollaborativeFiltering(data_path)
content_model = ContentBasedFiltering(data_path)
hybrid_model = HybridModel(collab_model, content_model)

def display_menu():
    print("\n=== Recommendation System Menu ===")
    print("1. Collaborative Filtering Recommendations")
    print("2. Content-Based Filtering Recommendations")
    print("3. Hybrid Recommendations")
    print("4. Exit")

def get_show_id_by_title(title, data):
    # Case-insensitive search for the show title in the dataset
    matches = data[data['title'].str.lower() == title.lower()]
    if matches.empty:
        print(f"Show titled '{title}' not found in the dataset.")
        return None
    return matches['show_id'].iloc[0]

def get_collaborative_recommendations():
    show_title = input("Enter Show Title: ")
    show_id = get_show_id_by_title(show_title, collab_model.data)
    if show_id:
        num_recs = int(input("Enter number of recommendations: "))
        recommendations = collab_model.recommend_items(show_id, num_recs)
        print(f"\nCollaborative Filtering Recommendations for Show '{show_title}':")
        print(recommendations)

def get_content_based_recommendations():
    show_title = input("Enter Show Title: ")
    show_id = get_show_id_by_title(show_title, content_model.data)
    if show_id:
        num_recs = int(input("Enter number of recommendations: "))
        recommendations = content_model.recommend_similar_items(show_id, num_recs)
        print(f"\nContent-Based Recommendations for Show '{show_title}':")
        print(recommendations)

def get_hybrid_recommendations():
    show_title = input("Enter Show Title: ")
    show_id = get_show_id_by_title(show_title, content_model.data)
    if show_id:
        num_recs = int(input("Enter number of recommendations: "))
        recommendations = hybrid_model.recommend(show_id, num_recs)
        print(f"\nHybrid Recommendations for Show '{show_title}':")
        print(recommendations)

def main():
    while True:
        display_menu()
        choice = input("Select an option (1-4): ")
        
        if choice == '1':
            get_collaborative_recommendations()
        elif choice == '2':
            get_content_based_recommendations()
        elif choice == '3':
            get_hybrid_recommendations()
        elif choice == '4':
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
