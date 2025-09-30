import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Load Data
# ==============================
df = pd.read_csv("dataset/ratings_movies_merged.csv")

# ==============================
# Perbaikan Judul Film
# ==============================
def fix_title(title):
    if ", The" in title:
        return "The " + title.replace(", The", "")
    elif ", An" in title:
        return "An " + title.replace(", An", "")
    elif ", A" in title:
        return "A " + title.replace(", A", "")
    return title

df["title"] = df["title"].apply(fix_title)

# ==============================
# Perbaikan Tahun Rilis
# ==============================
df = df[(df["year"] >= 1900) & (df["year"] <= 2025)]

# ==============================
# Informasi Awal
# ==============================
print("Jumlah data: \n", len(df))
print(df.dtypes)
print("\n\n Jumlah missing values per kolom:\n", df.isnull().sum())
print("\n\n Jumlah duplikat baris:", df.duplicated().sum())
print("\n\n Jumlah data unik:\n", df.nunique())
print(df.info)
# ==============================
# 1. Distribusi Rating
# ==============================
plt.figure(figsize=(8,5))
sns.countplot(x="rating", data=df, color="skyblue")
plt.title("Distribusi Rating")
plt.xlabel("Rating")
plt.ylabel("Jumlah")
plt.show()

# ==============================
# 2. Jumlah Rating per User
# ==============================
ratings_per_user = df["userId"].value_counts()
plt.figure(figsize=(10,5))
sns.histplot(ratings_per_user, bins=50, kde=False, color="skyblue")
plt.title("Distribusi Jumlah Rating per User")
plt.xlabel("Jumlah Rating per User")
plt.ylabel("Jumlah User")
plt.show()

# ==============================
# 3. Jumlah Rating per Movie
# ==============================
ratings_per_movie = df["title"].value_counts()
plt.figure(figsize=(10,5))
sns.histplot(ratings_per_movie, bins=50, kde=False, color="orange")
plt.title("Distribusi Jumlah Rating per Movie")
plt.xlabel("Jumlah Rating per Movie")
plt.ylabel("Jumlah Film")
plt.show()

# ==============================
# 4. Top 10 Film dengan Rating Terbanyak
# ==============================
top_movies = df["title"].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_movies.values, y=top_movies.index, palette=sns.color_palette("magma", 10))
plt.title("Top 10 Movies by Number of Ratings")
plt.xlabel("Jumlah Rating")
plt.ylabel("Film")
plt.show()

# ==============================
# 5. Distribusi Tahun Rilis Film
# ==============================
plt.figure(figsize=(12,6))
sns.histplot(df["year"], bins=30, kde=False, color="green")
plt.title("Distribusi Tahun Rilis Film (1900–2025)")
plt.xlabel("Tahun Rilis")
plt.ylabel("Jumlah Film")
plt.show()

# ==============================
# 6. Distribusi Genre Kombinasi
# ==============================
top_genres_combo = df["genres"].value_counts().head(15)
plt.figure(figsize=(12,6))
sns.barplot(x=top_genres_combo.values, y=top_genres_combo.index, 
            palette=sns.color_palette("coolwarm", 15))
plt.title("Top 15 Kombinasi Genre Terpopuler")
plt.xlabel("Jumlah Film")
plt.ylabel("Kombinasi Genre")
plt.show()

# ==============================
# 7. Distribusi Genre Tunggal (Explode)
# ==============================
df_exploded = df.assign(genres=df["genres"].str.split("|")).explode("genres")
plt.figure(figsize=(12,6))
sns.countplot(y="genres", data=df_exploded,
              order=df_exploded["genres"].value_counts().index,
              color="skyblue")
plt.title("Distribusi Genre (Single Genre)")
plt.xlabel("Jumlah Film")
plt.ylabel("Genre")
plt.show()

# ==============================
# 8. Rata-rata Rating per Genre
# ==============================
plt.figure(figsize=(12,6))
sns.barplot(x="genres", y="rating", data=df_exploded,
            estimator=np.mean, palette=sns.color_palette("Set1", n_colors=len(df_exploded["genres"].unique())))
plt.xticks(rotation=90)
plt.title("Rata-rata Rating per Genre")
plt.xlabel("Genre")
plt.ylabel("Rata-rata Rating")
plt.show()

# ==============================
# 9. Hubungan Tahun Rilis dan Rating
# ==============================
plt.figure(figsize=(12,6))
sns.boxplot(x="year", y="rating", data=df[df["year"] >= 1950],
            color="lightgreen", showfliers=False)
plt.xticks(rotation=90)
plt.title("Distribusi Rating Berdasarkan Tahun Rilis (>=1950)")
plt.xlabel("Tahun Rilis")
plt.ylabel("Rating")
plt.show()

# ==============================
# 10. Heatmap Korelasi (Numerik)
# ==============================
plt.figure(figsize=(8,6))
corr = df[["userId", "movieId", "year", "rating"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi Antar Variabel Numerik")
plt.show()
