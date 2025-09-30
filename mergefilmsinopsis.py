import pandas as pd

# File paths
synopsis_csv = r"output_preprocessing_synopsis/movies_cleansynopsis.csv"
movies_csv = r"output_semua_film/all_movies_combined_all_genres.csv"
output_csv = r"output_semua_film/all_movies_with_synopsis.csv"

# Load CSV
df_synopsis = pd.read_csv(synopsis_csv, encoding='utf-8-sig', on_bad_lines='skip')
df_movies = pd.read_csv(movies_csv, encoding='utf-8-sig', on_bad_lines='skip')

# Samakan tipe data movieId
df_synopsis['movieId'] = df_synopsis['movieId'].astype(str)
df_movies['movieId'] = df_movies['movieId'].astype(str)

# Merge
df_combined = pd.merge(df_movies, df_synopsis, on='movieId', how='left')

# Urutkan kolom sesuai permintaan
desired_order = ['movieId', 'clean_title', 'year', 'genres', 'cleansynopsis', 'clean_review', 'rating']

# Ambil kolom yang ada di dataframe dan sesuai urutan
cols = [col for col in desired_order if col in df_combined.columns]

df_combined = df_combined[cols]

# Simpan hasil
df_combined.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"✔ Gabungan selesai! File disimpan di: {output_csv}")
print(f"Kolom sekarang: {df_combined.columns.tolist()}")
