import pandas as pd
import os
import re
import numpy as np

# Path utama
path_movies = 'dataraw/movies.csv'   # metadata film asli
path_animasi_folder = 'dataraw/animasi/'
output_folder = 'output_per_film_animasi'

# Buat folder output kalau belum ada
os.makedirs(output_folder, exist_ok=True)

# Load metadata film
df_movies = pd.read_csv(path_movies)

# Mapping movieId, title, dan file review
mapping_files = {
    1: {"title": "Toy Story (1995)", "file": "toystory1.csv"},
    3114: {"title": "Toy Story 2 (1999)", "file": "toystory2.csv"},
    78499: {"title": "Toy Story 3 (2010)", "file": "toystory3.csv"},
    6377: {"title": "Finding Nemo (2003)", "file": "findingnemo1.csv"},
    122470: {"title": "Inside Out (2011)", "file": "insideout.csv"},
    138616: {"title": "Tangled (2001)", "file": "tangled 2010.csv"},
    103141: {"title": "Monsters University (2013)", "file": "monstersuniversity.csv"},
    203222: {"title": "The Lion King (2019)", "file": "thelionking.csv"},
    152081: {"title": "Zootopia (2016)", "file": "zootopia.csv"},
    4886: {"title": "Monster Inc (2001)", "file": "monstersinc.csv"}
}

def split_title_year(title):
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

        clean_title, year = split_title_year(title_full)

        # Load review file
        file_path = os.path.join(path_animasi_folder, review_file)
        if not os.path.exists(file_path):
            print(f"⚠ File review tidak ditemukan: {file_path}")
            continue
        df_review = pd.read_csv(file_path)

        # ----- Debug: tampilkan ringkasan rating awal -----
        print("\n-----")
        print(f"Memproses: {clean_title} (movieId={movie_id})")
        print("Total baris awal:", len(df_review))
        if "rating" in df_review.columns:
            print("Contoh unique rating (sample 50):", pd.Series(df_review['rating'].dropna().unique()).head(50).tolist())
            print("Top value_counts (rating):\n", df_review['rating'].astype(str).value_counts(dropna=False).head(20))
        else:
            print("⚠ Kolom 'rating' tidak ada di file review!")

        # ----- Standardisasi & bersihkan 'null' varian -----
        # 1) strip whitespace di rating (hindari ' Null ' atau 'null ')
        if "rating" in df_review.columns:
            df_review['rating'] = df_review['rating'].astype(str).str.strip()

        # 2) ganti semua literal 'null' / 'Null' / 'NULL' (case-insensitive, dengan/spasi) jadi NaN
        #    juga tangani 'none' atau 'nan' literal
        df_review = df_review.replace(r'(?i)^\s*null\s*$', np.nan, regex=True)
        df_review = df_review.replace(r'(?i)^\s*(none|nan)\s*$', np.nan, regex=True)

        # 3) coba convert rating ke numeric (jika format numeric seperti '5' atau '4.0')
        #    non-numeric -> NaN (akan di-drop karena rating harus ada)
        if "rating" in df_review.columns:
            df_review['rating'] = pd.to_numeric(df_review['rating'], errors='coerce')

        # ----- Drop hanya baris yang rating-nya kosong (jangan drop semua kolom) -----
        before_drop = len(df_review)
        if "rating" in df_review.columns:
            df_review = df_review.dropna(subset=["rating"])
        after_drop = len(df_review)
        print(f"Baris sebelum drop rating-null: {before_drop}  → setelah drop: {after_drop} (dihapus {before_drop-after_drop})")

        # Ambil maksimal 300 review
        df_review = df_review.head(300)

        # Tambahkan metadata film ke setiap review (copy semua kolom metadata)
        for col in row.index:
            df_review[col] = row[col]

        # Kolom tambahan
        df_review["clean_title"] = clean_title
        df_review["year"] = year

        # Output filename pakai judul bersih
        safe_title = re.sub(r'[^0-9a-zA-Z]+', '', clean_title.lower()).strip('')
        output_csv = os.path.join(output_folder, f"final_{safe_title}.csv")
        output_xlsx = os.path.join(output_folder, f"final_{safe_title}.xlsx")

        # Simpan hasil
        df_review.to_csv(output_csv, index=False, encoding="utf-8-sig")
        try:
            df_review.to_excel(output_xlsx, index=False, engine="openpyxl")
        except Exception:
            pass

        print(f"✅ Disimpan: {output_csv} | Baris akhir: {len(df_review)}")

    except Exception as e:
        print(f"⚠ Gagal memproses {title_full}: {e}")

print("\nProses selesai. Semua hasil ada di folder:", output_folder)