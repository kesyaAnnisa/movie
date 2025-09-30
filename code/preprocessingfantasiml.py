import os
import re
import pandas as pd
import html
import contractions

def clean_text(text):
    """Bersihkan teks review agar siap untuk ML"""
    if pd.isna(text):
        return ""

    text = html.unescape(text)                      # unescape HTML entities
    text = re.sub(r"<.*?>", " ", str(text))         # buang html tags
    text = contractions.fix(text)                   # expand contractions
    text = re.sub(r"[^a-zA-Z]", " ", text)          # buang simbol non-alfabet
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()        # strip spasi berlebih
    return text


def clean_title(title):
    """Bersihkan judul film, buang tahun dalam kurung"""
    if pd.isna(title):
        return title
    return re.sub(r"\s*\(\d{4}\)", "", str(title)).strip()


def extract_year(title):
    """Ekstrak tahun dari judul film → integer"""
    if pd.isna(title):
        return pd.NA
    match = re.search(r"\((\d{4})\)", str(title))
    if match:
        return int(match.group(1))   # langsung integer
    return pd.NA


def preprocess_and_combine_all(input_folder, output_filepath):
    all_processed_dfs = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_folder, filename)
            df = pd.read_csv(filepath)

            # --- buat clean_review dulu sebelum review di-drop ---
            if "review" in df.columns:
                df["clean_review"] = df["review"].apply(clean_text)

            # --- buat clean_title & year ---
            if "title" in df.columns:
                if "clean_title" not in df.columns:
                    df["clean_title"] = df["title"].apply(clean_title)
                if "year" not in df.columns:
                    df["year"] = df["title"].apply(extract_year)

            # pastikan year jadi integer (kalau ada NaN, dtype bisa 'Int64')
            if "year" in df.columns:
                df["year"] = df["year"].astype("Int64")

            # --- drop hanya kolom tertentu ---
            drop_cols = ["helpful", "total", "date", "title", "review", "genre2", "genre3", "username"]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            # --- urutan kolom: movieId, clean_title, year dulu ---
            cols = df.columns.tolist()
            priority_cols = [c for c in ["movieId", "clean_title", "year"] if c in cols]
            other_cols = [c for c in cols if c not in priority_cols]
            df = df[priority_cols + other_cols]

            all_processed_dfs.append(df)
            print(f"✔ Selesai memproses: {filename} | Jumlah baris: {len(df)}")

    # --- gabungkan semua DataFrame ---
    if all_processed_dfs:
        combined_df = pd.concat(all_processed_dfs, ignore_index=True)
        combined_df.to_csv(output_filepath, index=False)
        print(f"\n✔ Semua file berhasil digabungkan ke: {output_filepath}")
        print(f"Total baris akhir: {len(combined_df)}")
        print(f"Kolom akhir: {combined_df.columns.tolist()}")
        print(f"Tipe data kolom year: {combined_df['year'].dtype}")
    else:
        print("\nTidak ada file CSV yang ditemukan untuk diproses.")


# --- jalankan ---
output_path = os.path.join("output_fantasi_film_clean_ml", "all_movies_combined.csv")
preprocess_and_combine_all("output_per_film_fantasi", output_path)
