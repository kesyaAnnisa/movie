import pandas as pd
import os
import re
import html
import contractions

input_csv = r"dataraw\sinopsis.csv"
output_folder = "output_preprocessing_synopsis"
os.makedirs(output_folder, exist_ok=True)
output_csv = os.path.join(output_folder, "movies_cleansynopsis.csv")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<.*?>", " ", str(text))
    text = contractions.fix(text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Step 1: Load CSV dulu
try:
    df = pd.read_csv(input_csv, on_bad_lines='skip', encoding='utf-8-sig')
except Exception as e:
    print(f"❌ Gagal membaca file CSV: {e}")
    exit()

# Step 2: Tampilkan kolom asli (debug)
print("\nKolom yang terbaca oleh pandas:")
for i, col in enumerate(df.columns):
    print(f"Kolom {i+1}: '{col}' (panjang: {len(col)}) - kode karakter: {[ord(c) for c in col]}")

# Step 3: Bersihkan nama kolom dari karakter tersembunyi (BOM, spasi)
df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

# Step 4: Tampilkan kolom setelah dibersihkan
print("\nKolom setelah dibersihkan:")
print(df.columns.tolist())

# Step 5: Cek apakah kolom yang diminta ada
if 'movieId' not in df.columns or 'synopsis' not in df.columns:
    raise ValueError("CSV masih belum punya kolom 'movieId' dan 'synopsis' setelah dibersihkan")

# Step 6: Bersihkan isi teks
df['cleansynopsis'] = df['synopsis'].apply(clean_text)

# Step 7: Simpan hasilnya
df_clean = df[['movieId', 'cleansynopsis']]
df_clean.to_csv(output_csv, index=False)

print(f"\n✔ Preprocessing selesai! File disimpan di: {output_csv}")
print(f"Total movie processed: {len(df_clean)}")
