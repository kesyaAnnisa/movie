import pandas as pd
import numpy as np
import random
import re

# seed biar hasil tetap sama
random.seed(42)
np.random.seed(42)

# === Load data ===
ratings = pd.read_csv("dataraw/ratings.csv", usecols=['userId','movieId','rating'])
movies  = pd.read_csv("dataraw/movies.csv", usecols=['movieId','title','genres'])

# === Step 1: Hitung popularitas film ===
movie_counts = ratings['movieId'].value_counts()

# 500 film terpopuler
top_500 = movie_counts.head(500).index.tolist()

# 500 film random dari sisa yang masih punya cukup rating
min_ratings = 20
candidates = movie_counts.drop(top_500, errors='ignore')
eligible = candidates[candidates >= min_ratings].index.tolist()

if len(eligible) >= 500:
    random_500 = random.sample(eligible, 500)
else:
    random_500 = candidates.sample(500, random_state=42).index.tolist()

# gabung jadi 1000 film
final_movies = set(top_500) | set(random_500)

# === Step 2: Ambil semua rating untuk 1000 film itu ===
ratings_movies = ratings[ratings['movieId'].isin(final_movies)]

# === Step 3: Pool user yang nge-rate film itu ===
users_pool = ratings_movies['userId'].unique()
print("Jumlah user yang pernah nge-rate film terpilih:", len(users_pool))

# target 50k user
target_users = 50000
if len(users_pool) < target_users:
    print(f"WARNING: hanya ada {len(users_pool)} user. Target disesuaikan.")
    target_users = len(users_pool)

sample_users = np.random.choice(users_pool, size=target_users, replace=False)

# filter ratings final
ratings_final = ratings_movies[ratings_movies['userId'].isin(sample_users)]

# === Step 4: Rapikan movies (extract year + pisah genres) ===

def extract_year(title):
    m = re.search(r"\((\d{4})\)", title)
    if m:
        return int(m.group(1))
    return np.nan

movies_final = movies[movies['movieId'].isin(final_movies)].copy()
movies_final['year'] = movies_final['title'].apply(extract_year)

# hapus tahun dari title
movies_final['title'] = movies_final['title'].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

# pisah genres ke kolom genre1, genre2, ...
max_genres = movies_final['genres'].str.split('|').map(len).max()
genre_split = movies_final['genres'].str.split('|', expand=True)
genre_split.columns = [f'genre{i+1}' for i in range(genre_split.shape[1])]

movies_final = pd.concat([movies_final[['movieId','title','year']], genre_split], axis=1)

# === Step 5: Simpan ===
ratings_final.to_csv("ratings_subset_mix.csv", index=False)
movies_final.to_csv("movies_subset_mix.csv", index=False)

# === Ringkasan ===
print("=== Hasil Akhir ===")
print("Jumlah user unik:", ratings_final['userId'].nunique())
print("Jumlah movie unik:", ratings_final['movieId'].nunique())
print("Jumlah rating:", len(ratings_final))
print("movies_subset_mix.csv preview:")
print(movies_final.head())