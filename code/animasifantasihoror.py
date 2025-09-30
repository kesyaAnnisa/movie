import pandas as pd
import os

# Daftar file yang akan digabungkan
file_paths = [
    r"output_animasi_film_clean_ml/all_movies_combined.csv",
    r"output_fantasi_film_clean_ml/all_movies_combined.csv",
    r"output_horor_film_clean_ml/all_movies_combined.csv",
    r"output_horor_film_clean_ml/all_movies_combined.csv"

]

all_dfs = []

for file in file_paths:
    if os.path.exists(file):
        df = pd.read_csv(file)
        all_dfs.append(df)
        print(f"✔ File dimuat: {file} | Jumlah baris: {len(df)}")
    else:
        print(f"⚠️ File tidak ditemukan: {file}")

# Gabungkan semua DataFrame
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    total_rows = len(combined_df)
    
    # Simpan hasil gabungan
    output_file = "output_semua_film/all_movies_combined_all_genres.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Semua file berhasil digabungkan ke: {output_file}")
    print(f"📊 Total baris akhir: {total_rows}")
    print(f"Kolom akhir: {combined_df.columns.tolist()}")
else:
    print("\n⚠️ Tidak ada file yang berhasil digabungkan.")
