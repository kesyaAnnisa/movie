import pandas as pd
import os

# Path utama
path_final_movies = 'dataraw/10_selected_movies_final.csv'
path_horror_folder = 'dataraw/horor/'
output_folder = 'output_per_film'

# Buat folder output kalau belum ada
os.makedirs(output_folder, exist_ok=True)

# Load dataset utama
df_final_movies = pd.read_csv(path_final_movies)

# Daftar mapping film → nama file review
mapping_files = {
    "Conjuring, The (2013)": "the conjuring1.csv",
    "The Conjuring 2 (2016)": "theconjuring2.csv",
    "Insidious (2010)": "insidious1.csv",
    "Insidious: Chapter 2 (2013)": "insidious2.csv",
    "Midsommar (2019)": "midsommar.csv",
    "Hereditary (2018)": "hereditary.csv",
    "Sinister (2012)": "sinister.csv",
    "Orphan (2017)": "orphan.csv",
    "Final Destination (2000)": "finaldestination.csv",
    "A Quiet Place (2018)": "aquietplace.csv"
}

# Loop setiap film
for title, review_file in mapping_files.items():
    try:
        row = df_final_movies[df_final_movies['title'] == title].iloc[0]

        # Load review file
        file_path = os.path.join(path_horror_folder, review_file)
        df_review = pd.read_csv(file_path)

        # Rename kolom kalau ada bentrok
        col_rename = {}
        seen = set()
        for col in df_review.columns:
            new_col = col
            i = 1
            while new_col in seen:  # kalau sudah ada kolom dengan nama sama → kasih suffix
                new_col = f"{col}_{i}"
                i += 1
            col_rename[col] = new_col
            seen.add(new_col)
        df_review.rename(columns=col_rename, inplace=True)

        # Tambahkan metadata film ke setiap review
        for col in df_final_movies.columns:
            if col == "rating" and "rating" in df_review.columns:
                # Jangan timpa rating kalau review sudah ada rating
                continue
            df_review[col] = row[col]

        # Output filename
        safe_title = title.lower().replace(" ", "").replace(":", "").replace(",", "").replace("(", "").replace(")", "")
        output_csv = os.path.join(output_folder, f"final_{safe_title}.csv")
        output_xlsx = os.path.join(output_folder, f"final_{safe_title}.xlsx")

        # Save ke CSV dan Excel
        df_review.to_csv(output_csv, index=False, encoding="utf-8")
        df_review.to_excel(output_xlsx, index=False, engine="openpyxl")

        print(f"✅ Berhasil gabung: {title} → {output_csv}")

    except Exception as e:
        print(f"⚠️ Gagal memproses {title}: {e}")

print("\nProses selesai. Semua hasil ada di folder:", output_folder)
