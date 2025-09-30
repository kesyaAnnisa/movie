import streamlit as st
import pandas as pd
from transformers import pipeline
import re
import warnings
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
warnings.filterwarnings("ignore")

# Konfigurasi halaman
st.set_page_config(
    page_title="🎬 Movie Review Pages", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Styling
st.markdown("""
<style>
.main { 
    color: black;
}
.stTextInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border-radius: 10px;
    border: 2px solid #4CAF50;
    padding: 10px;
    font-size: 16px;
}
.movie-card {
    background: rgba(255, 255, 255, 0.95);
    padding: 12px;
    border-radius: 12px;
    border: 1px solid rgba(0, 0, 0, 0.1);
    margin: 2px 0;
    color: black;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, span, div {
    color: black !important;
}
.metric-container {
    background: rgba(255, 255, 255, 0.95);
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #ddd;
    text-align: center;
    color: black !important;
    min-height: 70px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin: 2px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-label {
    font-size: 13px;
    color: #666 !important;
    margin-bottom: 3px;
    font-weight: bold;
}
.metric-value {
    font-size: 16px;
    color: black !important;
    font-weight: bold;
    word-wrap: break-word;
    overflow-wrap: break-word;
    white-space: normal;
    line-height: 1.2;
}
.recommendation-card {
    background: rgba(33, 150, 243, 0.15);
    padding: 10px;
    border-radius: 8px;
    border-left: 4px solid #2196F3;
    margin: 5px 0;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Header dengan background bg.jpg
st.markdown("""
<div style='text-align: center; margin-bottom: 1rem; background-image: url("bg.jpg"); background-size: cover; background-position: center; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); position: relative;'>
    <div style='position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255,255,255,0.85); border-radius: 15px; z-index: 1;'></div>
    <div style='position: relative; z-index: 2;'>
        <h1 style='color: #4CAF50; font-size: 2.5rem; margin-bottom: 0.3rem;'>
            🎬 Movie Review 
        </h1>
        <p style='font-size: 1.1rem; color: black; margin-bottom: 0; line-height: 1.3;'>
            Movie reviews based on user reviews!
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv("output_semua_film/all_movies_with_synopsis.csv")
        df = df.copy()
        df['clean_title'] = df['clean_title'].astype(str).fillna("")
        df['clean_review'] = df['clean_review'].astype(str).fillna("")
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['year'] = df['year'].astype(str).fillna("Unknown")
        df['genres'] = df['genres'].astype(str).fillna("Unknown")
        if 'cleansynopsis' in df.columns:
            df['cleansynopsis'] = df['cleansynopsis'].astype(str).fillna("")
        df = df[df['clean_review'].str.strip() != ""]
        df = df[df['clean_review'] != "nan"]
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def load_summarizer():
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
        return summarizer
    except Exception as e:
        st.error(f"❌ Error loading summarizer: {str(e)}")
        return None

def get_top_rated_movies(df, top_n=10):
    movie_stats = df.groupby('clean_title').agg({'rating': ['mean', 'count'], 'movieId': 'first', 'year': 'first', 'genres': 'first'}).reset_index()
    movie_stats.columns = ['title', 'avg_rating', 'review_count', 'movieId', 'year', 'genres']
    movie_stats = movie_stats[movie_stats['review_count'] >= 10]
    movie_stats = movie_stats.sort_values('avg_rating', ascending=False).head(top_n)
    return movie_stats

def get_most_reviewed_movies(df, top_n=10):
    movie_stats = df.groupby('clean_title').agg({'rating': ['mean', 'count'], 'movieId': 'first', 'year': 'first', 'genres': 'first'}).reset_index()
    movie_stats.columns = ['title', 'avg_rating', 'review_count', 'movieId', 'year', 'genres']
    movie_stats = movie_stats.sort_values('review_count', ascending=False).head(top_n)
    return movie_stats

def generate_wordcloud(reviews_text, max_words=50):
    try:
        text = " ".join(reviews_text)
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had', 'this', 'that', 'these', 'those', 'it', 'its', 'as', 'by', 'from'])
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, max_words=max_words, colormap='viridis', relative_scaling=0.5, min_font_size=10).generate(text)
        return wordcloud
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        return None

def get_top_keywords(reviews_text, top_n=10):
    try:
        text = " ".join(reviews_text).lower()
        words = re.findall(r'\b[a-z]{4,}\b', text)
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had', 'this', 'that', 'these', 'those', 'it', 'its', 'as', 'by', 'from', 'very', 'just', 'about', 'more', 'some', 'than', 'into', 'only', 'also', 'even', 'much', 'such', 'what', 'when', 'where', 'which'])
        filtered_words = [word for word in words if word not in stopwords]
        word_counts = Counter(filtered_words)
        return word_counts.most_common(top_n)
    except Exception as e:
        return []

def recommend_similar_movies(df, current_movie_title, current_genres, top_n=5):
    try:
        genre_list = current_genres.split('|')
        similar_movies = []
        for _, row in df.drop_duplicates('clean_title').iterrows():
            if row['clean_title'] == current_movie_title:
                continue
            movie_genres = str(row['genres']).split('|')
            common_genres = set(genre_list) & set(movie_genres)
            if common_genres:
                movie_data = df[df['clean_title'] == row['clean_title']]
                avg_rating = movie_data['rating'].mean()
                review_count = len(movie_data)
                similar_movies.append({'title': row['clean_title'], 'genres': row['genres'], 'year': row['year'], 'avg_rating': round(avg_rating, 1), 'review_count': review_count, 'common_genres': len(common_genres)})
        similar_movies = sorted(similar_movies, key=lambda x: (x['common_genres'], x['avg_rating']), reverse=True)
        return similar_movies[:top_n]
    except Exception as e:
        return []

def clean_review_text(text):
    text = str(text).strip()
    text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.(?=[A-Z])', '. ', text)
    text = re.sub(r',(?=[A-Za-z])', ', ', text)
    return text

def clean_text_for_summary(text_list):
    cleaned = []
    for text in text_list:
        if pd.notna(text) and str(text).strip() not in ["", "nan"]:
            clean = re.sub(r'[^\w\s.,!?()-]', ' ', str(text))
            clean = re.sub(r'\s+', ' ', clean).strip()
            if len(clean.split()) >= 5:
                cleaned.append(clean)
    return cleaned

def extract_objective_content(reviews):
    objective_sentences = []
    for review in reviews:
        sentences = re.split(r'[.!?]+', review)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 4:
                continue
            sentence_lower = sentence.lower()
            personal_pronouns = ['i ', ' i ', 'my ', 'me ', 'you ', 'your ', 'we ', 'us ', 'our']
            if not any(pronoun in sentence_lower for pronoun in personal_pronouns):
                if len(sentence.split()) >= 4:
                    objective_sentences.append(sentence)
    return objective_sentences

def create_objective_summary(reviews, summarizer, total_reviews):
    if not reviews or not summarizer:
        return "Unable to generate summary from available reviews."
    try:
        objective_content = extract_objective_content(reviews)
        if objective_content and len(" ".join(objective_content).split()) >= 20:
            base_text = " ".join(objective_content[:50])
        else:
            sample_reviews = reviews[:min(50, len(reviews))]
            cleaned_reviews = []
            for review in sample_reviews:
                review_text = str(review).strip()
                if len(review_text.split()) >= 3:
                    cleaned_reviews.append(review_text)
            base_text = " ".join(cleaned_reviews)
        words = base_text.split()
        if len(words) < 20:
            base_text = " ".join(reviews[:min(30, len(reviews))])
            words = base_text.split()
        if len(words) > 800:
            base_text = " ".join(words[:800])
        if len(words) < 10:
            return "Unable to generate summary due to insufficient review content."
        input_length = len(words)
        max_length = min(180, max(60, input_length // 3))
        min_length = min(40, max_length - 20)
        result = summarizer(base_text, max_length=max_length, min_length=min_length, do_sample=False, early_stopping=True, no_repeat_ngram_size=3, truncation=True)
        summary = result[0]['summary_text']
        summary = clean_personal_pronouns(summary)
        summary = make_summary_general(summary, total_reviews)
        summary = format_two_paragraphs(summary)
        return summary
    except Exception as e:
        try:
            if len(reviews) > 0:
                first_reviews = " ".join(reviews[:5])
                return f"Based on {total_reviews} viewer reviews, {first_reviews[:200]}..."
            return f"Summary could not be generated. Error: {str(e)}"
        except:
            return "Unable to process reviews at this time."

def clean_personal_pronouns(text):
    text = re.sub(r'\bI think\b', 'The film appears to be', text, flags=re.IGNORECASE)
    text = re.sub(r'\bI believe\b', 'The movie seems', text, flags=re.IGNORECASE)
    text = re.sub(r'\bI feel\b', 'The film creates an impression that', text, flags=re.IGNORECASE)
    text = re.sub(r'\bI found\b', 'The movie presents', text, flags=re.IGNORECASE)
    text = re.sub(r'\bI was\b', 'Viewers are', text, flags=re.IGNORECASE)
    text = re.sub(r'\bI would\b', 'One would', text, flags=re.IGNORECASE)
    text = re.sub(r'\bMy\b', 'The', text, flags=re.IGNORECASE)
    text = re.sub(r'\bme\b', 'viewers', text, flags=re.IGNORECASE)
    text = re.sub(r'\byou\b', 'viewers', text, flags=re.IGNORECASE)
    text = re.sub(r'\byour\b', 'the audience\'s', text, flags=re.IGNORECASE)
    text = re.sub(r'\byou\'re\b', 'viewers are', text, flags=re.IGNORECASE)
    text = re.sub(r'\byou\'ll\b', 'viewers will', text, flags=re.IGNORECASE)
    return text

def make_summary_general(summary, total_reviews):
    if total_reviews > 1:
        if total_reviews >= 100:
            prefix = f"{total_reviews}"
        elif total_reviews >= 50:
            prefix = f"According to {total_reviews} reviews analyzed, "
        else:
            prefix = f"From {total_reviews} viewer reviews, "
        summary_lower = summary.lower()
        if not any(summary_lower.startswith(start) for start in ['based on', 'according to', 'from', 'the film', 'the movie']):
            summary = prefix + summary.lower()
    summary = re.sub(r'\bthis film\b', 'the film', summary, flags=re.IGNORECASE)
    summary = re.sub(r'\bthis movie\b', 'the movie', summary, flags=re.IGNORECASE)
    return summary

def format_two_paragraphs(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 2:
        return text
    mid_point = len(sentences) // 2
    paragraph1 = " ".join(sentences[:mid_point]).strip()
    paragraph2 = " ".join(sentences[mid_point:]).strip()
    if not paragraph1:
        paragraph1 = sentences[0] if sentences else ""
    if not paragraph2 and len(sentences) > 1:
        paragraph2 = " ".join(sentences[1:])
    return f"{paragraph1}\n\n{paragraph2}" if paragraph2 else paragraph1

def calculate_movie_stats(df_movie):
    stats = {}
    stats['total_reviews'] = len(df_movie)
    valid_ratings = df_movie['rating'].dropna()
    if len(valid_ratings) > 0:
        stats['avg_rating'] = round(valid_ratings.mean(), 1)
        stats['max_rating'] = valid_ratings.max()
        stats['min_rating'] = valid_ratings.min()
    else:
        stats['avg_rating'] = stats['max_rating'] = stats['min_rating'] = 0
    if len(df_movie) > 0:
        first_row = df_movie.iloc[0]
        stats['year'] = first_row.get('year', 'Unknown')
        stats['genre'] = first_row.get('genres', 'Unknown')
        stats['movie_id'] = first_row.get('movieId', 'Unknown')
        if 'cleansynopsis' in df_movie.columns:
            synopsis = first_row.get('cleansynopsis', '')
            if pd.notna(synopsis) and str(synopsis).strip() not in ["", "nan"]:
                clean_synopsis = str(synopsis).strip()
                if len(clean_synopsis) > 500:
                    sentences = re.split(r'[.!?]+', clean_synopsis)
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated + sentence) < 450:
                            truncated += sentence + ". "
                        else:
                            break
                    stats['synopsis'] = truncated.strip() + "..." if len(clean_synopsis) > 450 else clean_synopsis
                else:
                    stats['synopsis'] = clean_synopsis
            else:
                stats['synopsis'] = ""
        else:
            stats['synopsis'] = ""
    return stats

# Load data dan model
df = load_data()
summarizer = load_summarizer()

if df is None:
    st.error("❌ Cannot load data. Please check your CSV file.")
    st.stop()

if summarizer is None:
    st.error("❌ Cannot load summarizer model.")
    st.stop()

if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = ""

# Sidebar
with st.sidebar:
    st.sidebar.markdown("<div style='font-size: 20px; font-weight: 600; margin-bottom: 6px;'>🎬 Movie Browser</div>", unsafe_allow_html=True)
    
    if df is not None:
        st.markdown("### 🔍 Filters")
        all_genres = set()
        for genres in df['genres'].unique():
            if pd.notna(genres) and str(genres) != "Unknown":
                all_genres.update(str(genres).split('|'))
        all_genres = sorted(list(all_genres))
        sidebar_genre = st.selectbox("🎭 Genre:", ["All"] + all_genres, key="sidebar_genre")
        all_years = sorted([y for y in df['year'].unique() if y != "Unknown"], reverse=True)
        sidebar_year = st.selectbox("📅 Year:", ["All"] + all_years, key="sidebar_year")
        sidebar_rating = st.selectbox("⭐ Rating:", ["All", "9+ (Excellent)", "8-9 (Great)", "7-8 (Good)", "6-7 (Average)", "<6 (Below Average)"], key="sidebar_rating")
        search_btn = st.button("🔍 Search Movies", type="primary", use_container_width=True)
        st.markdown("---")
        
        has_filter = sidebar_genre != "All" or sidebar_year != "All" or sidebar_rating != "All"
        
        if search_btn:
            if not has_filter:
                filtered_sidebar_df = df.copy()
            else:
                filtered_sidebar_df = df.copy()
                if sidebar_genre != "All":
                    filtered_sidebar_df = filtered_sidebar_df[filtered_sidebar_df['genres'].str.contains(sidebar_genre, na=False)]
                if sidebar_year != "All":
                    filtered_sidebar_df = filtered_sidebar_df[filtered_sidebar_df['year'] == sidebar_year]
                if sidebar_rating != "All":
                    if sidebar_rating == "9+ (Excellent)":
                        filtered_sidebar_df = filtered_sidebar_df[filtered_sidebar_df['rating'] >= 9]
                    elif sidebar_rating == "8-9 (Great)":
                        filtered_sidebar_df = filtered_sidebar_df[(filtered_sidebar_df['rating'] >= 8) & (filtered_sidebar_df['rating'] < 9)]
                    elif sidebar_rating == "7-8 (Good)":
                        filtered_sidebar_df = filtered_sidebar_df[(filtered_sidebar_df['rating'] >= 7) & (filtered_sidebar_df['rating'] < 8)]
                    elif sidebar_rating == "6-7 (Average)":
                        filtered_sidebar_df = filtered_sidebar_df[(filtered_sidebar_df['rating'] >= 6) & (filtered_sidebar_df['rating'] < 7)]
                    elif sidebar_rating == "<6 (Below Average)":
                        filtered_sidebar_df = filtered_sidebar_df[filtered_sidebar_df['rating'] < 6]
            filtered_movies = sorted(filtered_sidebar_df['clean_title'].unique())
            st.session_state.filtered_movies = filtered_movies
            st.session_state.filtered_df = filtered_sidebar_df
        
        if 'filtered_movies' in st.session_state and 'filtered_df' in st.session_state:
            filtered_movies = st.session_state.filtered_movies
            filtered_sidebar_df = st.session_state.filtered_df
            st.markdown(f"### 📊 Search Results")
            st.info(f"Found {len(filtered_movies)} movies")
            if len(filtered_movies) > 0:
                for idx, movie in enumerate(filtered_movies[:50], 1):
                    movie_data = filtered_sidebar_df[filtered_sidebar_df['clean_title'] == movie]
                    avg_rating = round(movie_data['rating'].mean(), 1)
                    review_count = len(movie_data)
                    if st.button(f"{movie}", key=f"movie_{idx}", use_container_width=True):
                        st.session_state.selected_movie = movie
                        st.rerun()
                    st.markdown(f"<div style='font-size: 11px; color: #666; margin: -8px 0 8px 0; padding-left: 5px;'>⭐ {avg_rating}/10 • {review_count} reviews</div>", unsafe_allow_html=True)
                if len(filtered_movies) > 50:
                    st.caption(f"... and {len(filtered_movies) - 50} more movies")
            else:
                st.warning("No movies found with selected filters.")
        
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📋 All Movies", "🏆 Top Rated", "🔥 Most Reviewed"])
        
        with tab1:
            pass
        
        with tab2:
            st.markdown("### Top 10 Highest Rated")
            top_rated = get_top_rated_movies(df, 10)
            for idx, row in top_rated.iterrows():
                if st.button(row['title'], key=f"top_{idx}", use_container_width=True):
                    st.session_state.selected_movie = row['title']
                    st.rerun()
                st.markdown(f"<div style='font-size: 11px; color: #666; margin: -8px 0 8px 0; padding-left: 5px;'>⭐ {row['avg_rating']:.1f}/10 • {int(row['review_count'])} reviews</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Top 10 Most Reviewed")
            most_reviewed = get_most_reviewed_movies(df, 10)
            for idx, row in most_reviewed.iterrows():
                if st.button(row['title'], key=f"most_{idx}", use_container_width=True):
                    st.session_state.selected_movie = row['title']
                    st.rerun()
                st.markdown(f"<div style='font-size: 11px; color: #666; margin: -8px 0 8px 0; padding-left: 5px;'>📊 {int(row['review_count'])} reviews • ⭐ {row['avg_rating']:.1f}/10</div>", unsafe_allow_html=True)

# Main content
st.markdown("---")
st.markdown("### 🔍 Search Movie")

movie_title = st.text_input("Enter movie title:", value=st.session_state.selected_movie, placeholder="Example: Avengers, Batman, Spider-Man...", help="Type partial or full movie title to search", label_visibility="collapsed")

filtered_df = df.copy()

if movie_title and len(movie_title.strip()) > 0:
    search_results = filtered_df[filtered_df['clean_title'].str.contains(movie_title.strip(), case=False, na=False, regex=False)].copy()
    
    if search_results.empty:
        st.markdown("<div class='movie-card' style='background: rgba(244, 67, 54, 0.15); border-left: 5px solid #f44336;'><h3 style='color: black; margin-bottom: 5px;'>🚫 Movie Not Found</h3><p style='color: black; margin-bottom: 0;'>Please try different keywords or adjust filters.</p></div>", unsafe_allow_html=True)
    else:
        stats = calculate_movie_stats(search_results)
        st.markdown(f"<div class='movie-card' style='margin-top: 2px; margin-bottom: 5px; padding: 12px;'><h2 style='color: #4CAF50; margin-bottom: 0.3rem; margin-top: 0; font-size: 24px;'>🎭 {movie_title.title()}</h2></div>", unsafe_allow_html=True)
        
        if stats['synopsis']:
            st.markdown(f"<div style='background: rgba(255,255,255,0.95); padding: 12px; border-radius: 10px; margin: 2px 0; border-left: 4px solid #4CAF50; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'><h4 style='color: #4CAF50; margin-bottom: 5px; margin-top: 0; font-size: 16px;'>📖 Synopsis</h4><p style='font-style: italic; color: black; line-height: 1.4; text-align: justify; margin-bottom: 0; font-size: 14px;'>{stats['synopsis']}</p></div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 0.7, 1.3])
        
        with col1:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>🎬 Movie ID</div><div class='metric-value'>{stats['movie_id']}</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>📊 Total Reviews</div><div class='metric-value'>{stats['total_reviews']}</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>⭐ Average Rating</div><div class='metric-value'>{stats['avg_rating']}/10</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>📅 Year</div><div class='metric-value'>{stats['year']}</div></div>", unsafe_allow_html=True)
        with col5:
            st.markdown(f"<div class='metric-container'><div class='metric-label'>🎭 Genre</div><div class='metric-value' style='font-size: 13px; line-height: 1.2;'>{stats['genre']}</div></div>", unsafe_allow_html=True)
        
        all_reviews = search_results['clean_review'].tolist()
        cleaned_reviews = clean_text_for_summary(all_reviews)
        
        if not cleaned_reviews:
            st.error("❌ No valid reviews found for processing.")
        else:
            summary = create_objective_summary(cleaned_reviews, summarizer, len(cleaned_reviews))

            # buang awalan "300" kalau ada
            if summary.startswith("300"):
                summary = summary[3:].strip()

            st.markdown("### 🎯 Audience Analysis")
            st.markdown(f"<div style='background: rgba(76, 175, 80, 0.15); padding: 10px 14px; border-radius: 8px; border-left: 4px solid #4CAF50; border: 1px solid #ddd; margin: 5px 0 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'><h3 style='color: #4CAF50; font-size: 18px; margin: 0 0 2px 0;'>📊 Summary from {len(cleaned_reviews)} Reviews</h3><div style='font-size: 14px; line-height: 1.35; text-align: justify; color: black; white-space: pre-line; margin: 0;'>{summary}</div></div>", unsafe_allow_html=True)
        

            st.markdown("### 💬 Review Keywords Analysis")
            col_wc1, col_wc2 = st.columns([2, 1])

            
            with col_wc1:
                wordcloud = generate_wordcloud(all_reviews, max_words=50)
                if wordcloud:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
            
            with col_wc2:
                st.markdown("**Top Keywords:**")
                keywords = get_top_keywords(all_reviews, top_n=10)
                
                for idx, (word, count) in enumerate(keywords, 1):
                    st.markdown(f"""
                    <div style='background: rgba(76, 175, 80, 0.15); padding: 5px 10px; 
                                border-radius: 5px; margin: 3px 0;'>
                        <strong>{idx}. {word}</strong> <span style='color: #666;'>({count}×)</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendation Section
            st.markdown("### 🎯 You Might Also Like")
            recommendations = recommend_similar_movies(df, movie_title.title(), stats['genre'], top_n=5)
            
            if recommendations:
                rec_cols = st.columns(min(len(recommendations), 3))
                
                for idx, rec in enumerate(recommendations):
                    col_idx = idx % 3
                    with rec_cols[col_idx]:
                        st.markdown(f"""
                        <div class='recommendation-card'>
                            <div style='font-weight: bold; color: #2196F3; margin-bottom: 3px;'>
                                🎬 {rec['title']}
                            </div>
                            <div style='font-size: 12px; color: #666;'>
                                ⭐ {rec['avg_rating']}/10 • 📅 {rec['year']}<br>
                                🎭 {rec['genres'][:40]}{'...' if len(rec['genres']) > 40 else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No similar movies found based on genre.")
            
            # Sample reviews dengan View More/Less
            with st.expander("👁️ View sample reviews"):
                sample_reviews = search_results.head(5)
                for idx, (_, row) in enumerate(sample_reviews.iterrows(), 1):
                    rating = row.get('rating', 'N/A')
                    review_text = str(row.get('clean_review', '')).strip()
                    movie_id = row.get('movieId', 'N/A')
                    
                    if review_text and review_text != 'nan':
                        review_text = clean_review_text(review_text)
                        
                        # Create unique key for each review's expander state
                        review_key = f"review_{movie_id}_{idx}"
                        
                        # Check if review is long (more than 400 characters)
                        is_long = len(review_text) > 400
                        
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="color: #4CAF50; font-weight: bold; margin-bottom: 8px;">
                                Review {idx} • Movie ID: {movie_id} • Rating: {rating}/10
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Initialize session state for this review if not exists
                        if review_key not in st.session_state:
                            st.session_state[review_key] = False
                        
                        # Display review text based on state
                        if is_long and not st.session_state[review_key]:
                            st.markdown(f"""
                            <div style="color: black; line-height: 1.6; text-align: justify; margin-top: -10px;">
                                {review_text[:400]}...
                            </div>
                            """, unsafe_allow_html=True)
                            if st.button(f"📖 View More", key=f"btn_more_{review_key}"):
                                st.session_state[review_key] = True
                                st.rerun()
                        else:
                            st.markdown(f"""
                            <div style="color: black; line-height: 1.6; text-align: justify; margin-top: -10px;">
                                {review_text}
                            </div>
                            """, unsafe_allow_html=True)
                            if is_long and st.session_state[review_key]:
                                if st.button(f"📕 View Less", key=f"btn_less_{review_key}"):
                                    st.session_state[review_key] = False
                                    st.rerun()

else:
    st.markdown("""
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p style="color: black;">
        🔧 Powered by Streamlit & HuggingFace Transformers | 
        📊 Database: 9,000+ Movie Reviews
    </p>
</div>
""", unsafe_allow_html=True)