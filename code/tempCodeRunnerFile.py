import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("output_semua_film/all_movies_with_synopsis.csv")

print("Sebelum pembersihan rating:")
print(df['rating'].value_counts(dropna=False))

# ===============================
# 2. Bersihkan rating & buat label sentimen
# ===============================
df['rating'] = df['rating'].replace('Null', np.nan)
df = df[df['rating'].notna()]
df['rating'] = df['rating'].astype(float)

# Konversi rating jadi label sentimen biner
def rating_to_sentiment(r):
    if r >= 7:
        return 'positif'
    elif r <= 4:
        return 'negatif'
    else:
        return None  # rating 5-6 dianggap netral dan dibuang

df['sentiment'] = df['rating'].apply(rating_to_sentiment)
df = df[df['sentiment'].notna()]

print("\nSetelah konversi ke sentimen:")
print(df['sentiment'].value_counts())

# ===============================
# 3. Siapkan fitur & label
# Gabungkan sinopsis + review
# ===============================
df["text"] = df["cleansynopsis"].fillna("") + " " + df["clean_review"].fillna("")
X = df["text"]
y = df["sentiment"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 4. TF-IDF Vectorization
# ===============================
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("✔ TF-IDF selesai")
print("Shape train:", X_train_tfidf.shape, " | test:", X_test_tfidf.shape)

# ===============================
# 5. Train Logistic Regression
# ===============================
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

print("✔ Training selesai")

# ===============================
# 6. Evaluasi Model
# ===============================
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
print("\n✅ Akurasi:", acc)
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['negatif', 'positif'])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=['negatif', 'positif'],
            yticklabels=['negatif', 'positif'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Sentiment Classification")
plt.tight_layout()
plt.show()

# ===============================
# 7. Simpan Model, Vectorizer, dan Hasil Prediksi
# ===============================
os.makedirs("saved_models", exist_ok=True)

joblib.dump(model, "saved_models/logistic_sentiment_model.pkl")
joblib.dump(vectorizer, "saved_models/tfidf_sentiment_vectorizer.pkl")
print("✔ Model & Vectorizer disimpan ke folder 'saved_models'")

# Simpan hasil prediksi
results = pd.DataFrame({
    "movieId": df.loc[X_test.index, "movieId"].values,
    "clean_title": df.loc[X_test.index, "clean_title"].values,
    "year": df.loc[X_test.index, "year"].values,
    "genres": df.loc[X_test.index, "genres"].values,
    "actual_sentiment": y_test.values,
    "predicted_sentiment": y_pred
})
results.to_csv("saved_models/predictions_sentiment_logistic.csv", index=False, encoding="utf-8-sig")
print("✔ Hasil prediksi disimpan ke 'saved_models/predictions_sentiment_logistic.csv'")
print("Sebelum pembersihan rating:")
print(df['rating'].value_counts(dropna=False))