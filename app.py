import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    st.error("TMDB_API_KEY not found. Make sure it is set in the .env file.")
    st.stop()





# TMDB API config
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"




# Set page config
st.set_page_config(
    page_title="Movie Recommendation System App",
    page_icon="",
    layout="wide"
)

# Function to load data
@st.cache_data
def load_movie_data():
    df = pd.read_csv("./Datasets/ml-32m/movies.csv")
    df['genres'] = df['genres'].apply(lambda x: x.split('|'))
    return df

@st.cache_data
def load_ratings_data():
    df = pd.read_csv("./Datasets/ml-32m/ratings.csv")
    return df

@st.cache_data
def load_Q_matrix():
    return np.load("./Saved Models/Q_matrix.npy")

@st.cache_data
def load_movie_mapper():
    with open("./Saved Models/movie_mapper.pkl", "rb") as f:
        movie_mapper = pickle.load(f)
        return movie_mapper
    
@st.cache_data
def load_movie_inv_mapper():
    with open("./Saved Models/movie_inv_mapper.pkl", "rb") as f:
        movie_inv_mapper = pickle.load(f)
        return movie_inv_mapper

@st.cache_data
def load_user_inv_mapper():
    with open("./Saved Models/user_inv_mapper.pkl", "rb") as f:
        user_inv_mapper = pickle.load(f)
        return user_inv_mapper
    
@st.cache_data
def load_movie_stats():
    df = pd.read_csv("./Datasets/ml-32m/movie_stats.csv")
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

@st.cache_data
def load_links():
    df = pd.read_csv("./Datasets/ml-32m/links.csv")
    return df

try:
    # Try to load the data
    movie_data = load_movie_data()
    ratings_data = load_ratings_data()
    movie_stats = load_movie_stats()
    links = load_links()
    movie_mapper = load_movie_mapper()
    movie_inv_mapper = load_movie_inv_mapper()
    user_inv_mapper = load_user_inv_mapper()
    Q_matrix = load_Q_matrix()

    resource_loaded = True
except Exception as e:
    st.error(f"Error loading resource: {e}")
    resource_loaded = False


# Function to edit necessary data
def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else None


# Preparing necessary data
TOP_N_MOVIES = 12
movie_data['year'] = movie_data['title'].apply(extract_year)
all_genres = sorted(set(g for genre_list in movie_data['genres'] for g in genre_list))
movie_data = movie_data.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')

# To avoid a column name conflict on 'title' during the merge,
# we drop it from a temporary version of movie_stats.
# This keeps the original movie_stats DataFrame intact for other uses.
stats_to_merge = movie_stats.drop(columns=['title']) if 'title' in movie_stats.columns else movie_stats
movies_with_rating = movie_data.merge(stats_to_merge, on='movieId', how='left')

title_to_id = dict(zip(movie_data['title'], movie_data['movieId']))
id_to_title = dict(zip(movie_data['movieId'], movie_data['title']))



# --- TMDB Data Fetcher ---
# --- TMDB Fetcher using TMDB ID ---
@st.cache_data(show_spinner=False)
def fetch_tmdb_data_by_id(tmdb_id):
    if np.isnan(tmdb_id):
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
        params = {"api_key": TMDB_API_KEY}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"TMDB fetch error: {e}")
    return None



# --- Movie Poster Fetcher ---
@st.cache_data(show_spinner=False)
def fetch_poster_df(movies_df):
    posters = []
    for _, row in movies_df.iterrows():
        tmdb = fetch_tmdb_data_by_id(row['tmdbId'])
        if tmdb and tmdb.get("poster_path"):
            posters.append({
                "title": row['title'],
                "tmdbId": row['tmdbId'],
                "poster_url": TMDB_IMAGE_BASE + tmdb["poster_path"]
            })
    return pd.DataFrame(posters)



# --- Poster Display with Checkboxes ---
def display_movie_selection(movie_df, max_cols=3):
    selected = []
    rows = len(movie_df) // max_cols + 1
    for row_idx in range(rows):
        cols = st.columns(max_cols)
        for i in range(max_cols):
            idx = row_idx * max_cols + i
            if idx >= len(movie_df):
                break
            movie = movie_df.iloc[idx]
            with cols[i]:
                st.image(movie["poster_url"], width=150)
                st.caption(movie["title"])
                if st.checkbox(f"{movie['title']}", key=f"movie_{idx}"):
                    selected.append(movie["title"])
                # Add synopsis expander
                tmdb_data = fetch_tmdb_data_by_id(movie['tmdbId'])
                if tmdb_data and tmdb_data.get("overview"):
                    with st.expander("Synopsis"):
                        st.write(tmdb_data["overview"])
    return selected



# Show the movie poster based on movies selected by the user
def show_selected_movies(movie_titles, all_poster_df, max_cols=4):
    st.markdown("### 🎯 Your Selected Movies:")
    selected_df = all_poster_df[all_poster_df['title'].isin(movie_titles)]

    rows = len(selected_df) // max_cols + 1
    for row_idx in range(rows):
        cols = st.columns(max_cols)
        for i in range(max_cols):
            idx = row_idx * max_cols + i
            if idx >= len(selected_df):
                break
            row = selected_df.iloc[idx]
            with cols[i]:
                st.image(row['poster_url'], width=120)
                st.caption(row['title'])




@st.cache_data(show_spinner=False)
def fetch_popular_movie():
    url = "https://api.themoviedb.org/3/movie/popular"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": 1}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                return results[0]  # return the most popular one
    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to fetch popular movie: {e}")
    return None


# Functions to get the recommended movies
def filter_movies_by_genre(genres, movies_df, movie_mapper):
    filtered_movies = movies_df[movies_df['genres'].apply(lambda gs: all(g in gs for g in genres))]
    return filtered_movies[filtered_movies['movieId'].isin(set(movie_mapper.keys()))]


def filter_movies_by_genre_and_year(genres, year_range, movies_df, movie_mapper):
    filtered = movies_df[
        (movies_df['genres'].apply(lambda gs: all(g in gs for g in genres))) &
        (movies_df['year'] >= year_range[0]) &
        (movies_df['year'] <= year_range[1])
    ]
    return filtered[filtered['movieId'].isin(set(movie_mapper.keys()))]


def get_top_movies(filtered_df, top_n=20):
    return filtered_df.sort_values(by='bayesian_average', ascending=False).head(top_n)


def get_selected_movies_ids(selected_titles, title_to_id, movie_mapper):
    return [title_to_id[t] for t in selected_titles if title_to_id.get(t) in movie_mapper]


def compute_user_vector(movie_ids, movie_mapper, Q_matrix):
    user_vecs = [Q_matrix[movie_mapper[mid]] for mid in movie_ids]
    return np.mean(user_vecs, axis=0).reshape(1, -1)


def recommend_similar_movies(user_vec, Q_matrix, movie_inv_mapper, id_to_title, exclude_titles, top_k=6):
    sims = cosine_similarity(user_vec, Q_matrix).flatten()
    sorted_idx = sims.argsort()[::-1]

    recommendations = []
    for i in sorted_idx:
        mid = movie_inv_mapper.get(i)
        if not mid:
            st.warning(f"Invalid mapper index: {i}")
            continue
        title = id_to_title.get(mid)
        if not title:
            st.warning(f"Invalid movieId: {mid}")
            continue
        if title not in exclude_titles and title not in recommendations:
            recommendations.append(title)
        if len(recommendations) >= top_k:
            break

    return recommendations




def display_recommendations(titles, movies_with_rating, fetch_tmdb_data_by_id, image_base_url):
    st.markdown("## 🍿 Because you loved those movies, try these:")
    tmdb_cache = {}

    for title in titles:
        row = movies_with_rating[movies_with_rating['title'] == title].dropna(subset=['tmdbId'])
        if row.empty:
            continue

        tmdb_id = int(row.iloc[0]['tmdbId'])
        genres = ", ".join(row.iloc[0]['genres'])

        if tmdb_id not in tmdb_cache:
            tmdb_cache[tmdb_id] = fetch_tmdb_data_by_id(tmdb_id)

        tmdb = tmdb_cache[tmdb_id]
        
        if tmdb and tmdb.get("poster_path"):
            st.image(image_base_url + tmdb["poster_path"], width=150)
        st.subheader(tmdb["title"] if tmdb else title)
        st.markdown(f"**Genres:** {genres}")
        st.write(tmdb.get("overview", "No description available.") if tmdb else "")

        

# Function to check API connectivity
def test_tmdb_connection(api_key: str, verbose: bool = False):
    """Test connectivity to the TMDB API."""
    url = "https://api.themoviedb.org/3/movie/550"  # Known stable endpoint: Fight Club
    params = {"api_key": api_key}
        
    try:
        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:
            st.success("✅ TMDB API is reachable and working correctly.")
            if verbose:
                st.write("Status Code:", response.status_code)
        elif response.status_code == 401:
            st.error("❌ Unauthorized. Check if your TMDB API key is correct.")
        elif response.status_code == 404:
            st.error("❌ Endpoint not found. Check the TMDB base URL or movie ID.")
        else:
            st.warning(f"⚠️ Unexpected status code: {response.status_code}")
    except requests.exceptions.Timeout:
        st.error("⏳ Request timed out. Check your network or TMDB server status.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection error: {e}") 




# Main Title
st.title("🎬 Movie Recommendation App")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("select Page:", ['Home', 'Movie Recommendation System', 'EDA', 'Evaluasi Model'])


# Home Page
if page == 'Home':

    col1, col2 = st.columns(2)

    with col1:
        st.write("---")
        st.image("https://www.themoviedb.org/assets/2/v4/logos/v2/blue_square_2-d537fb228cf3ded904ef09b136fe3fec72548ebc1fea3fbbd1ad9e36364db38b.svg", width=320)
        st.write("This app is powered by TMDB API")


        st.write("""
        ### About the App
        This app recommends movies to the user 
        based on the user preffered movies and genres.
                         
        - Movie Genres
        - Selected Movies user likes
                 
        This app uses Hybrid Recommendation System
        - Cold Start using Bayesian Average + Genre
        - Uses SVD to create matrix factorization
        """)
        st.write("---")


    with col2:
        st.write("---")
        st.markdown("### 🔥 Currently Popular Movie")

        pop_movie = fetch_popular_movie()
        if pop_movie:
            poster_url = TMDB_IMAGE_BASE + pop_movie.get("poster_path", "")
            title = pop_movie.get("title", "Untitled")
            overview = pop_movie.get("overview", "No description available.")
            
            st.image(poster_url, width=200)
            st.subheader(title)
            st.write(overview)
        else:
            st.info("Unable to fetch popular movie right now.")

        st.write("---")
    
    st.markdown("### 🔧 TMDB API Connectivity Status")
    test_tmdb_connection(TMDB_API_KEY, verbose=True)

# Prediction Page
elif page == 'Movie Recommendation System':

    st.write("""
    ### Search Page
    This app recommends the user based on movie genres they like.
    """)
        
    # Step 1: Genre selection and set year filter
    selected_genres = st.multiselect("What genres do you enjoy?", all_genres, default=['Action', 'Comedy'])

    min_year = int(movie_data['year'].min())
    max_year = int(movie_data['year'].max())
    
    st.write("---")
    selected_year_range = st.slider(
        "🎞️ Filter movies released between:",
        min_value=min_year,
        max_value=max_year,
        value=(2000, 2020),
        step=1,
        format="%d"
    )

    st.markdown(f"### 🎯 You selected: **{selected_year_range[0]} – {selected_year_range[1]}**")
    st.write("---")

    if selected_genres:
        filtered_movies = filter_movies_by_genre_and_year(selected_genres, selected_year_range, movies_with_rating, movie_mapper)
        top_movies = get_top_movies(filtered_movies, TOP_N_MOVIES)
        poster_df = fetch_poster_df(top_movies)

        st.markdown("### Pick a few movies you love:")
        st.info("Please select at least one movie to get recommendations.")
        selected_movies = display_movie_selection(poster_df, max_cols=3)

        st.write("---")
        if selected_movies:
            show_selected_movies(selected_movies, poster_df)

            st.write("---")
            selected_ids = get_selected_movies_ids(selected_movies, title_to_id, movie_mapper)
            if not selected_ids:
                st.warning("None of the selected movies are available in the model.")
            else:
                user_vec = compute_user_vector(selected_ids, movie_mapper, Q_matrix)
                recommendations = recommend_similar_movies(user_vec, Q_matrix, movie_inv_mapper, id_to_title, selected_movies)
                display_recommendations(recommendations, movies_with_rating, fetch_tmdb_data_by_id, TMDB_IMAGE_BASE)


elif page == 'EDA':
    st.title("🔍 Exploratory Data Analysis (EDA)")

    st.markdown("This page provides an exploratory data analysis of the MovieLens dataset, similar to the analysis in the training notebook.")
    st.write("---")

    # --- Dataset Overview ---
    st.subheader("📖 Dataset Overview")
    st.info("Key statistics about the dataset, showing its scale and density.")
    
    n_ratings = len(ratings_data)
    n_movies_rated = ratings_data['movieId'].nunique()
    n_users = ratings_data['userId'].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ratings Given", f"{n_ratings:,}")
    col2.metric("Unique Users", f"{n_users:,}")
    col3.metric("Unique Movies Rated", f"{n_movies_rated:,}")
    
    st.write("---")

    # --- Ratings Analysis ---
    st.subheader("⭐ Ratings Analysis")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("#### Distribution of Ratings")
        rating_counts = ratings_data['rating'].value_counts().sort_index()
        st.bar_chart(rating_counts)
        mean_global_rating = ratings_data['rating'].mean()
        st.markdown(f"**Mean Global Rating:** `{mean_global_rating:.2f}`")

    with col2:
        st.write("#### Sample DataFrames")
        with st.expander("Movie Data Sample"):
            st.dataframe(movie_data.head())
        with st.expander("Ratings Data Sample"):
            st.dataframe(ratings_data.head())
        with st.expander("Movie Stats Sample (with Bayesian Avg)"):
            st.dataframe(movies_with_rating.head())

    st.write("---")

    # --- Movie Popularity and Quality ---
    st.subheader("🎬 Movie Popularity & Quality Insights")

    # Most Rated Movies
    st.write("#### 📈 Top 20 Most frequently Rated Movies")
    most_rated = movies_with_rating.sort_values('count', ascending=False).head(20)
    st.dataframe(most_rated[['title', 'count', 'mean', 'bayesian_average', 'year']])

    # Top & Bottom Movies by Bayesian Average
    st.write("#### 🏆 Top vs. Bottom Movies (by Bayesian Average)")
    st.markdown("The Bayesian Average adjusts a movie's rating based on the number of ratings it has received. This prevents movies with few high ratings from dominating the 'top-rated' lists.")
    
    top_bayesian = movies_with_rating.sort_values('bayesian_average', ascending=False).head(10)
    bottom_bayesian = movies_with_rating.sort_values('bayesian_average', ascending=True).head(10)

    col1, col2 = st.columns(2)
    with col1:
        st.write("##### 👍 Top 10 Highest-Rated Movies")
        st.dataframe(top_bayesian[['title', 'bayesian_average', 'count', 'year']])
    with col2:
        st.write("##### 👎 Top 10 Lowest-Rated Movies")
        st.dataframe(bottom_bayesian[['title', 'bayesian_average', 'count', 'year']])
    
    st.write("---")
    
    # --- Distributions ---
    st.subheader("📊 Visual Distributions")
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Genre Distribution")
        genre_counts = pd.Series([genre for sublist in movie_data['genres'] for genre in sublist]).value_counts()
        st.bar_chart(genre_counts)
    with col2:
        st.write("#### Movie Release Year Distribution")
        # Ensure 'year' column is not NaN and is integer
        year_counts = movie_data['year'].dropna().astype(int).value_counts().sort_index()
        st.line_chart(year_counts)

    # --- Bayesian Average Distribution ---
    st.write("#### Bayesian Average Rating Distribution")
    fig, ax = plt.subplots()
    sns.histplot(movies_with_rating['bayesian_average'], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Bayesian Average Ratings")
    ax.set_xlabel("Bayesian Average Rating")
    ax.set_ylabel("Number of Movies")
    st.pyplot(fig)


elif page == 'Evaluasi Model':
    st.title("🧪 Model Evaluation")

    st.write("Evaluasi dilakukan terhadap model rekomendasi berbasis SVD dan Bayesian Average.")
    
    st.subheader("📉 Matrix Q Dimensions")
    st.write(f"Shape of Q matrix: {Q_matrix.shape}")

    st.subheader("🔁 Similarity Score Sample")
    # Ambil 5 movie vector secara acak dan tampilkan cosine similarity
    sample_indices = np.random.choice(Q_matrix.shape[0], size=5, replace=False)
    sample_vectors = Q_matrix[sample_indices]

    # Ambil movieId dan judul film dari movie_inv_mapper
    sample_movie_ids = [movie_inv_mapper[i] for i in sample_indices]
    sample_titles = [id_to_title[mid] for mid in sample_movie_ids]

    # Hitung cosine similarity
    similarities = cosine_similarity(sample_vectors)

    # Buat DataFrame dengan label judul film
    similarity_df = pd.DataFrame(similarities, index=sample_titles, columns=sample_titles)

    st.dataframe(similarity_df)
