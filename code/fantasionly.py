import pandas as pd
import os
import re
import numpy as np

# Path utama
path_movies = 'dataraw/movies.csv'   # metadata film asli
path_fantasi_folder = 'dataraw/fantasi/'
output_folder = 'output_per_film_fantasi'

# Buat folder output kalau belum ada
os.makedirs(output_folder, exist_ok=True)

# Load metadata film
df_movies = pd.read_csv(path_movies)

# Mapping movieId, title, dan file review
mapping_files = {
    8368: {"title": "Harry Potter and the Prisoner of Azkaban (2004)", "file": "hpaz.csv"},
    81834: {"title": "Harry Potter and the Deathly Hallows: Part 1 (2010)", "file": "hpdh1.csv"},
    72998: {"title": "Avatar (2009)", "file": "avatar.csv"},
    78772: {"title": "The Twilight Saga: Eclipse (2010)", "file": "twilighteclipse.csv"},
    595: {"title": "Beauty and the Beast (1991)", "file": "beautyandthebeast.csv"},
    2628: {"title": "Star Wars: Episode I - The Phantom Menace (1999)", "file": "starwarsepisode1.csv"},
    5952: {"title": "The Lord of the Rings: The Two Towers (2002)", "file": "thelordoftheringsthetwotowers.csv"},
    7153: {"title": "The Lord of the Rings: The Return of the King  (2003)", "file": "thelordoftheringsthereturnoftheking.csv"},
    98809: {"title": "The Hobbit: An Unexpected Journey (2012)", "file": "thehobbitanunexpectedjourney.csv"},
    118696: {"title": "The Hobbit: The Battle of the Five Armies (2014)", "file": "thehobbitthebattleofthefivearmies.csv"}
}

def split_title_year(title):
    """
    Pisahkan judul dan tahun dari string, contoh:
    'Toy Story (1995)' -> ('Toy Story', 1995)
    """
    match = re.match(r"^(.*)\s\((\d{4})\)$", title)
    if match:
        return match.group(1), int(match.group(2))
    else:
        return title, None

# Loop setiap film
for movie_id, info in mapping_files.items():
    try:
        title_full = info["title"]
        review_file = info["file"]

        # Ambil metadata film dari movies.csv
        row = df_movies[df_movies['movieId'] == movie_id]
        if row.empty:
            print(f"⚠ Metadata untuk {title_full} (movieId={movie_id}) tidak ditemukan di movies.csv")
            continue
        row = row.iloc[0]

        # Bersihkan judul dan ambil tahun
        clean_title, year = split_title_year(title_full)

        # Load review file
        file_path = os.path.join(path_fantasi_folder, review_file)
        df_review = pd.read_csv(file_path)

        # --- Bersihkan nilai 'null', 'Null', 'NULL' jadi NaN ---
        df_review = df_review.replace(r'(?i)^null$', np.nan, regex=True)

        # Drop baris yang rating kosong
        before_drop = len(df_review)
        df_review = df_review.dropna(subset=["rating"])
        after_drop = len(df_review)

        # Ambil maksimal 300 review
        df_review = df_review.head(300)

        # Tambahkan metadata film ke setiap review
        for col in row.index:
            df_review[col] = row[col]

        # Tambahkan kolom "clean_title" dan "year"
        df_review["clean_title"] = clean_title
        df_review["year"] = year

        # Output filename pakai judul bersih
        safe_title = clean_title.lower().replace(" ", "").replace(":", "").replace(",", "")
        output_csv = os.path.join(output_folder, f"final_{safe_title}.csv")
        output_xlsx = os.path.join(output_folder, f"final_{safe_title}.xlsx")

        # Simpan hasil
        df_review.to_csv(output_csv, index=False, encoding="utf-8")
        df_review.to_excel(output_xlsx, index=False, engine="openpyxl")

        print(f"✅ {clean_title} (year={year}, movieId={movie_id})")
        print(f"   - Total review awal : {before_drop}")
        print(f"   - Setelah drop 'Null' : {after_drop}")
        print(f"   - Disimpan (max 300): {len(df_review)} → {output_csv}")

    except Exception as e:
        print(f"⚠ Gagal memproses {title_full}: {e}")

print("\nProses selesai. Semua hasil ada di folder:", output_folder)