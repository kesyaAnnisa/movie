import pandas as pd

# Load data
ratings = pd.read_csv("dataraw/ratings.csv")
movies = pd.read_csv("dataraw/movies.csv")

# --- Step 1: Ambil 50.000 user ---
sample_users = ratings['userId'].drop_duplicates().sample(50000, random_state=42)
ratings_subset = ratings[ratings['userId'].isin(sample_users)]

# --- Step 2a: Ambil 1000 movie (Random) ---
sample_movies_random = ratings_subset['movieId'].drop_duplicates().sample(1000, random_state=42)
ratings_random = ratings_subset[ratings_subset['movieId'].isin(sample_movies_random)]
movies_random = movies[movies['movieId'].isin(sample_movies_random)]

# Simpan versi random
ratings_random.to_csv("ratings_subset_random.csv", index=False)
movies_random.to_csv("movies_subset_random.csv", index=False)

print("=== Random ===")
print("Jumlah user unik:", ratings_random['userId'].nunique())
print("Jumlah movie unik:", ratings_random['movieId'].nunique())
print("Jumlah rating:", len(ratings_random))
print()

# --- Step 2b: Ambil 1000 movie (Top populer) ---
movie_counts = ratings_subset['movieId'].value_counts().head(1000).index
ratings_top = ratings_subset[ratings_subset['movieId'].isin(movie_counts)]
movies_top = movies[movies['movieId'].isin(movie_counts)]

# Simpan versi top populer
ratings_top.to_csv("ratings_movies_merged.csv", index=False)

print("=== Top Populer ===")
print("Jumlah user unik:", ratings_top['userId'].nunique())
print("Jumlah movie unik:", ratings_top['movieId'].nunique())
print("Jumlah rating:", len(ratings_top))
