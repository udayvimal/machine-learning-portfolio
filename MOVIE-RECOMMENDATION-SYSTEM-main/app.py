import streamlit as st
import pickle
import pandas as pd
import requests
import os

# ---------- Custom Dark Theme CSS ----------
st.markdown("""
<style>
/* Dark background gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #eee;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Sidebar dark style */
[data-testid="stSidebar"] {
    background: #121212;
    color: #bbb;
    font-weight: 500;
}

/* Title gradient text */
h1 {
    background: linear-gradient(45deg, #ff6a00, #ee0979);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Button style */
.stButton > button {
    background: linear-gradient(45deg, #ff6a00, #ee0979);
    color: #fff;
    font-weight: 700;
    border-radius: 12px;
    height: 45px;
    width: 170px;
    transition: background 0.3s ease;
    box-shadow: 0 0 10px #ff6a00aa;
}

.stButton > button:hover {
    background: linear-gradient(45deg, #ee0979, #ff6a00);
    box-shadow: 0 0 20px #ee0979cc;
}

/* Movie card style */
.movie-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(255, 106, 0, 0.4);
    padding: 12px;
    text-align: center;
    margin-bottom: 20px;
    transition: transform 0.25s ease;
}
.movie-card:hover {
    transform: scale(1.08);
}

/* Poster styling */
img {
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(255, 106, 0, 0.7);
    transition: transform 0.3s ease;
    cursor: pointer;
}
img:hover {
    transform: scale(1.12);
}

/* Footer */
footer {
    text-align: center;
    color: #888;
    font-size: 13px;
    margin-top: 50px;
}

/* Tooltip for selectbox */
[data-testid="stSelectbox"] > div > div {
    color: #ddd;
}
</style>
""", unsafe_allow_html=True)

# --------- Page setup ----------
st.set_page_config(
    page_title="🎬 Movie Recommender - Dark Mode",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🎥 About Movie Recommender")
    st.markdown("""
    Welcome to the **dark mode** Movie Recommender System!  
    Select your favorite movie and get 5 similar picks with stunning posters.  

    Powered by Streamlit and OMDb API.
    """)
    st.markdown("---")
    st.markdown("👨‍💻 Developed by [Uday Vimal](https://github.com/udayvimal)")
    st.markdown("---")
    with st.expander("📁 Show project files"):
        st.write(os.listdir("."))

# -------- Load data --------
@st.cache_data(show_spinner=False)
def load_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    movies_dict = load_pickle_file("movies.pkl")
    similarity = load_pickle_file("similarity.pkl")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

movies = pd.DataFrame(movies_dict)

# -------- Functions ---------
poster_cache = {}

def fetch_poster(movie_title):
    if movie_title in poster_cache:
        return poster_cache[movie_title]

    api_key = "63dfac01"
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        poster_url = data.get("Poster", "https://via.placeholder.com/200")
    except Exception:
        poster_url = "https://via.placeholder.com/200"

    poster_cache[movie_title] = poster_url
    return poster_url

def recommend(movie, movies, similarity):
    try:
        idx = movies[movies['title'] == movie].index[0]
    except IndexError:
        st.warning("⚠️ Movie not found in dataset.")
        return [], []

    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    rec_titles = []
    rec_posters = []

    for i in movie_list:
        title = movies.iloc[i[0]].title
        rec_titles.append(title)
        rec_posters.append(fetch_poster(title))

    return rec_titles, rec_posters

# ---------- Main UI ----------
st.markdown("<h1>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#ddd; font-size:18px;'>Choose a movie below to get top 5 recommendations.</p>", unsafe_allow_html=True)

selected_movie = st.selectbox(
    "🔍 Select a movie:",
    options=movies["title"].values,
    help="Type to search your favorite movie",
)

st.write("")  # spacing

if st.button("Recommend 🎯"):
    with st.spinner("Finding your movie matches..."):
        recs, posters = recommend(selected_movie, movies, similarity)

    if recs:
        st.markdown("---")
        st.subheader("🎥 Recommended for you:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <img src="{posters[i]}" alt="{recs[i]}" style="width:100%; height:auto;" />
                        <h4 style="color:#ff6a00; margin-top:10px;">{recs[i]}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.warning("No recommendations found. Try a different movie.")

# ---------- Footer ----------
st.markdown(
    """
    <footer>
    Developed by <a href="https://github.com/udayvimal" target="_blank" style="color:#ff6a00;">Uday Vimal</a> &bull; Powered by Streamlit
    </footer>
    """,
    unsafe_allow_html=True,
)
