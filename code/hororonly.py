import pandas as pd
import os
import re

# Path utama
path_movies = 'dataraw/movies.csv'   # metadata film asli
path_horor_folder = 'dataraw/horor/'  # folder review horor
output_folder = 'output_horor_film'

# Buat folder output kalau belum ada
os.makedirs(output_folder, exist_ok=True)

# Load metadata film
df_movies = pd.read_csv(path_movies)

# Mapping movieId dan title dari daftar 10 film horor/thriller
mapping_files = {
    103688: {"title": "The Conjuring (2013)", "file": "theconjuring1.csv"},
    159858: {"title": "The Conjuring 2 (2016)", "file": "theconjuring2.csv"},
    85788: {"title": "Insidious (2010)", "file": "insidious1.csv"},
    288357: {"title": "Split (2016)", "file": "split.csv"},
    201646: {"title": "Midsommar (2019)", "file": "midsommar.csv"},
    183869: {"title": "Hereditary (2018)", "file": "hereditary.csv"},
    97188: {"title": "Sinister (2012)", "file": "sinister.csv"},
    168834: {"title": "Orphan (2017)", "file": "orphan.csv"},
    3409: {"title": "Final Destination (2000)", "file": "finaldestination.csv"},
    185029: {"title": "A Quiet Place (2018)", "file": "aquietplace.csv"}
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
        file_path = os.path.join(path_horor_folder, review_file)
        df_review = pd.read_csv(file_path)

        # Drop review dengan rating "null", "Null", "NULL"
        before_drop = len(df_review)
        df_review = df_review.replace(r'(?i)^null$', pd.NA, regex=True)  # ganti string "null"/"Null" jadi NaN
        df_review = df_review.dropna(subset=["rating"])  # drop baris yang rating-nya NaN
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