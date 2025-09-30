import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer

# ========================
# Configuration
# ========================
st.set_page_config(
    page_title="Analisis Sentimen Review Film",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Analisis Sentimen Review Film")
st.write("Masukkan review film dan sistem akan menganalisis apakah review tersebut **Positif** atau **Negatif**")

# ========================
# Load Dataset Function
# ========================
@st.cache_data
def load_dataset():
    current_dir = os.getcwd()
    
    # Possible paths for dataset
    possible_paths = [
        "../output_horor_film_clean_ml/moviesfinalcombination.csv",
        "output_horor_film_clean_ml/moviesfinalcombination.csv",
        "../output_horror_film_clean_ml/moviesfinalcombination.csv",
        "output_horror_film_clean_ml/moviesfinalcombination.csv",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Convert labels to sentiment (assuming Horror = Negative, Non-Horror = Positive)
                if 'label' in df.columns:
                    df['sentiment'] = df['label'].apply(
                        lambda x: 'Negatif' if 'horror' in str(x).lower() else 'Positif'
                    )
                    return df, path
            except Exception as e:
                continue
    
    return None, None

# ========================
# Generate Natural Language Explanation
# ========================
def generate_explanation(user_input, prediction, probability, top_words):
    """Generate human-readable explanation"""
    
    confidence_level = "sangat yakin" if max(probability) > 0.8 else "cukup yakin" if max(probability) > 0.6 else "ragu-ragu"
    percentage = max(probability) * 100
    
    # Analyze sentiment based on words
    positive_words = [word for word, score in top_words if score > 0]
    negative_words = [word for word, score in top_words if score < 0]
    
    explanation = f"Saya {confidence_level} ({percentage:.1f}%) bahwa review ini bersifat **{prediction.lower()}**.\n\n"
    
    if prediction == "Positif":
        explanation += "**Mengapa review ini positif?**\n"
        if positive_words:
            explanation += f"• Review mengandung kata-kata positif seperti: *{', '.join(positive_words[:3])}*\n"
        explanation += "• Nada keseluruhan review menunjukkan kepuasan atau apresiasi\n"
        explanation += "• Tidak ada indikasi kekecewaan yang signifikan\n"
        
        if negative_words:
            explanation += f"\n**Catatan:** Meskipun ada beberapa kata yang cenderung negatif seperti *{', '.join(negative_words[:2])}*, konteks keseluruhan tetap menunjukkan review yang positif."
    
    else:  # Negative
        explanation += "**Mengapa review ini negatif?**\n"
        if negative_words:
            explanation += f"• Review mengandung kata-kata negatif seperti: *{', '.join(negative_words[:3])}*\n"
        explanation += "• Nada keseluruhan review menunjukkan ketidakpuasan atau kritik\n"
        explanation += "• Terdapat indikasi kekecewaan terhadap film\n"
        
        if positive_words:
            explanation += f"\n**Catatan:** Meskipun ada beberapa kata yang cenderung positif seperti *{', '.join(positive_words[:2])}*, konteks keseluruhan tetap menunjukkan review yang negatif."
    
    return explanation

# ========================
# Main App
# ========================
# Try to load dataset
df, dataset_path = load_dataset()

if df is None:
    st.error("❌ Dataset tidak ditemukan!")
    st.write("**Solusi:**")
    st.write("1. Pastikan file `moviesfinalcombination.csv` ada di folder `output_horor_film_clean_ml/`")
    st.write("2. Atau upload file secara manual:")
    
    uploaded_file = st.file_uploader("Upload Dataset CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'label' in df.columns:
                df['sentiment'] = df['label'].apply(
                    lambda x: 'Negatif' if 'horror' in str(x).lower() else 'Positif'
                )
                st.success("Dataset berhasil diupload!")
            else:
                st.error("Dataset harus memiliki kolom 'label'")
                st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
    else:
        st.stop()

# Clean and prepare data
if 'overview' not in df.columns or 'sentiment' not in df.columns:
    st.error("Dataset harus memiliki kolom 'overview' dan dapat dikonversi ke sentiment")
    st.stop()

df = df.dropna(subset=['overview', 'sentiment'])
df['overview'] = df['overview'].astype(str)

# Show dataset info
with st.expander("📊 Informasi Dataset"):
    st.success(f"Dataset berhasil dimuat dari: `{dataset_path if dataset_path else 'Upload manual'}`")
    st.write(f"**Jumlah data:** {len(df):,} reviews")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribusi Sentimen:**")
        sentiment_counts = df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
    
    with col2:
        st.write("**Contoh Data:**")
        for sentiment in df['sentiment'].unique():
            sample = df[df['sentiment'] == sentiment]['overview'].iloc[0]
            st.write(f"**{sentiment}:** {sample[:100]}...")

# ========================
# Train Model
# ========================
@st.cache_resource
def train_sentiment_model(df):
    X = df['overview']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=500, class_weight='balanced')
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    X_test_vec = vectorizer.transform(X_test)
    accuracy = model.score(X_test_vec, y_test)
    
    return model, vectorizer, accuracy

with st.spinner("🤖 Melatih model AI..."):
    model, vectorizer, accuracy = train_sentiment_model(df)

st.success(f"✅ Model siap digunakan! Akurasi: {accuracy:.1%}")

# ========================
# Prediction Interface
# ========================
st.subheader("💬 Analisis Review Anda")

# Example reviews
examples = {
    "Review Positif": "Film ini benar-benar luar biasa! Ceritanya sangat menarik dan aktingnya sangat bagus. Visual effectsnya juga memukau. Sangat recommended untuk ditonton!",
    "Review Negatif": "Film ini sangat mengecewakan. Plotnya membosankan dan tidak masuk akal. Aktingnya juga sangat buruk. Buang-buang waktu saja menontonnya."
}

col1, col2 = st.columns(2)
with col1:
    if st.button("📝 Contoh Review Positif", use_container_width=True):
        st.session_state.review_text = examples["Review Positif"]

with col2:
    if st.button("📝 Contoh Review Negatif", use_container_width=True):
        st.session_state.review_text = examples["Review Negatif"]

user_input = st.text_area(
    "Tulis review film di sini:",
    value=st.session_state.get("review_text", ""),
    placeholder="Contoh: Film ini sangat bagus! Ceritanya menarik dan aktingnya luar biasa...",
    height=120,
    help="Tulis pendapat Anda tentang sebuah film"
)

if st.button("🔍 Analisis Review", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Silakan tulis review terlebih dahulu!")
    else:
        try:
            # Make prediction
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]
            probabilities = model.predict_proba(input_vec)[0]
            prob_dict = dict(zip(model.classes_, probabilities))
            
            # Get LIME explanation for word importance
            def predict_proba_wrapper(texts):
                return model.predict_proba(vectorizer.transform(texts))
            
            explainer = LimeTextExplainer(class_names=model.classes_)
            exp = explainer.explain_instance(user_input, predict_proba_wrapper, num_features=10)
            
            # Extract important words
            word_importance = exp.as_list()
            
            # ========================
            # Display Results
            # ========================
            st.markdown("---")
            st.subheader("📊 Hasil Analisis")
            
            # Main prediction with styling
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if prediction == "Positif":
                    st.success("### 😊 Review POSITIF")
                    st.balloons()
                else:
                    st.error("### 😞 Review NEGATIF")
            
            # Confidence scores
            st.subheader("📈 Tingkat Keyakinan")
            
            for sentiment, prob in prob_dict.items():
                if sentiment == prediction:
                    st.metric(
                        label=f"🎯 {sentiment}",
                        value=f"{prob:.1%}",
                        delta=f"Prediksi utama"
                    )
                else:
                    st.metric(
                        label=f"⚪ {sentiment}",
                        value=f"{prob:.1%}",
                        delta=None
                    )
            
            # Natural language explanation
            st.subheader("💡 Penjelasan")
            explanation = generate_explanation(user_input, prediction, probabilities, word_importance)
            st.markdown(explanation)
            
            # Word importance visualization
            st.subheader("🔍 Kata-kata Penting")
            st.write("Kata-kata yang paling berpengaruh dalam analisis:")
            
            # Show top positive and negative words
            positive_words = [(word, score) for word, score in word_importance if score > 0]
            negative_words = [(word, score) for word, score in word_importance if score < 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if positive_words:
                    st.write("**Kata-kata yang mendukung sentimen positif:**")
                    for word, score in positive_words[:5]:
                        st.write(f"• **{word}** (bobot: +{score:.3f})")
            
            with col2:
                if negative_words:
                    st.write("**Kata-kata yang mendukung sentimen negatif:**")
                    for word, score in negative_words[:5]:
                        st.write(f"• **{word}** (bobot: {score:.3f})")
            
            # Interactive LIME visualization
            with st.expander("🔬 Analisis Detail (LIME)"):
                st.write("Visualisasi interaktif yang menunjukkan kontribusi setiap kata:")
                st.components.v1.html(exp.as_html(), height=400, scrolling=True)
                
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {str(e)}")

# ========================
# Tips and Info
# ========================
with st.expander("💡 Tips Penggunaan"):
    st.markdown("""
    **Cara terbaik menggunakan tool ini:**
    
    1. **Tulis review yang jujur** - Ekspresikan pendapat Anda tentang film dengan natural
    2. **Gunakan kalimat lengkap** - Review yang lebih panjang akan dianalisis lebih akurat
    3. **Gunakan bahasa campuran** - Tool ini bisa menganalisis review dalam bahasa Indonesia maupun Inggris
    
    **Contoh review yang baik:**
    - ✅ "Film Avengers terbaru benar-benar spectacular! Action scenesnya amazing dan plotnya sangat engaging."
    - ✅ "Sangat disappointed dengan film ini. Boring banget dan waste of time."
    - ❌ "Bagus" (terlalu singkat)
    - ❌ "Film" (tidak informatif)
    
    **Catatan:** Sistem ini dilatih untuk mengenali sentimen umum dalam review film, bukan genre spesifik.
    """)

st.markdown("---")
st.markdown("*Dibuat dengan ❤️ menggunakan Streamlit dan Machine Learning*")