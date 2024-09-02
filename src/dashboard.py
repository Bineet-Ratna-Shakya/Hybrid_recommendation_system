import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from collaborative_filtering import CollaborativeFiltering, generate_synthetic_user_item_matrix
from content_based_filtering import ContentBasedFiltering
from deep_learning_model import NeuralCollaborativeFiltering
from hybrid_model import HybridFiltering
from utils import save_feedback

data_cleaned = pd.read_csv('/Users/soul/Documents/hybrid_recommendation_system/data/Netflix-encoded-Data.csv')

num_synthetic_users = 10  # Number of synthetic users
user_item_matrix, movie_titles = generate_synthetic_user_item_matrix(data_cleaned, num_users=num_synthetic_users)

content_filter = ContentBasedFiltering(data_cleaned)
collab_filter = CollaborativeFiltering(user_item_matrix, movie_titles)
hybrid_filter = HybridFiltering(data_cleaned, user_item_matrix, movie_titles, embedding_dim=20)
hybrid_filter.prepare_models()

ncf = NeuralCollaborativeFiltering(user_item_matrix, movie_titles, embedding_dim=20)
ncf.train(epochs=5)

def generate_recommendations():
    movie_title = movie_entry.get().strip()
    try:
        user_index = int(user_entry.get().strip())
        if user_index < 0 or user_index >= num_synthetic_users:
            raise ValueError("Invalid user index")
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid user index (0 to {})".format(num_synthetic_users - 1))
        return

    if movie_title.lower() not in data_cleaned['title'].str.lower().values:
        messagebox.showerror("Error", "Movie title not found in dataset")
        return

    correct_title = data_cleaned[data_cleaned['title'].str.lower() == movie_title.lower()]['title'].values[0]

    content_recs = content_filter.recommend(correct_title)
    collab_recs = collab_filter.recommend(user_idx=user_index)
    hybrid_recs = hybrid_filter.recommend(correct_title, user_idx=user_index)

    for tree in [content_tree, collab_tree, hybrid_tree]:
        for item in tree.get_children():
            tree.delete(item)

    for i, rec in enumerate(content_recs):
        content_tree.insert("", "end", values=(i+1, rec))
    for i, rec in enumerate(collab_recs):
        collab_tree.insert("", "end", values=(i+1, rec))
    for i, rec in enumerate(hybrid_recs):
        hybrid_tree.insert("", "end", values=(i+1, rec))

def incorporate_feedback():
    feedback = {}
    for item in hybrid_tree.get_children():
        movie_title = hybrid_tree.item(item, 'values')[1]
        rating = feedback_entry.get()
        try:
            rating = float(rating)
            if 0 <= rating <= 5:
                feedback[movie_title] = rating
            else:
                raise ValueError("Invalid rating")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid rating between 0 and 5.")
            return
    
    ncf.incorporate_feedback(feedback)  
    messagebox.showinfo("Success", "Feedback incorporated and model retrained!")

app = tk.Tk()
app.title("Hybrid Recommendation System")
app.state('zoomed')  
input_frame = ttk.Frame(app, padding=10)
input_frame.pack(pady=20, fill='x')

output_frame = ttk.Frame(app, padding=10)
output_frame.pack(pady=10, fill='x')

feedback_frame = ttk.Frame(app, padding=10)
feedback_frame.pack(pady=10, fill='x')

ttk.Label(input_frame, text="Movie Title:").grid(row=0, column=0, padx=10, sticky='e')
movie_entry = ttk.Entry(input_frame, width=30)
movie_entry.grid(row=0, column=1, padx=10)

ttk.Label(input_frame, text="Synthetic User Index (0 to {}):".format(num_synthetic_users - 1)).grid(row=1, column=0, padx=10, sticky='e')
user_entry = ttk.Entry(input_frame, width=30)
user_entry.grid(row=1, column=1, padx=10)

recommend_button = ttk.Button(input_frame, text="Generate Recommendations", command=generate_recommendations)
recommend_button.grid(row=2, column=0, columnspan=2, pady=10)

columns = ("#", "Movie Title")

content_tree = ttk.Treeview(output_frame, columns=columns, show='headings', height=6)
content_tree.heading("#", text="#")
content_tree.heading("Movie Title", text="Content-Based Recommendations")
content_tree.pack(side="left", padx=10)

collab_tree = ttk.Treeview(output_frame, columns=columns, show='headings', height=6)
collab_tree.heading("#", text="#")
collab_tree.heading("Movie Title", text="Collaborative Recommendations")
collab_tree.pack(side="left", padx=10)

hybrid_tree = ttk.Treeview(output_frame, columns=columns, show='headings', height=6)
hybrid_tree.heading("#", text="#")
hybrid_tree.heading("Movie Title", text="Hybrid Recommendations")
hybrid_tree.pack(side="left", padx=10)

ttk.Label(feedback_frame, text="Enter your rating (0-5) for the recommended movies:").pack(pady=5)
feedback_entry = ttk.Entry(feedback_frame, width=10)
feedback_entry.pack()

feedback_button = ttk.Button(feedback_frame, text="Submit Feedback", command=incorporate_feedback)
feedback_button.pack(pady=10)

app.mainloop()
