# 🎬 CineMatch: Movie Recommender

A content-based movie recommender built on the TMDB 5000 dataset, with a Netflix-style Streamlit UI.

Pick a movie and CineMatch suggests similar ones based on plot, genres, keywords, cast and crew.

## Features
- Similar-movie recommendations with a short "why recommended" reason (same director, shared cast, genres)
- Horizontally scrolling rows: *Because you picked…*, *More from the director*, *Top-rated in genre*, *Trending this week*, *All-time top rated*
- Hero banner with backdrop, poster, rating, runtime, cast, and an in-app trailer
- Sidebar filters: number of results, genres, release year, minimum rating, sort order
- Search by actor or director, 🎲 **Surprise me**, and a watchlist

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Add a [TMDB API read access token](https://www.themoviedb.org/settings/api) (for posters and trailers) in `.streamlit/secrets.toml`:
```toml
TMDB_TOKEN = "your-token-here"
```

Then start the app:
```bash
streamlit run app.py
```

The app works without a token too, but without posters or trailers.

## How it works
`Movie_recomender.ipynb` builds a tag for each movie from its overview, genres, keywords, top cast and director. It vectorizes the tags and computes cosine similarity between every pair of movies.
`build_artifacts.py` then shrinks the full similarity matrix (~185 MB) down to each movie's top 50 neighbours (`neighbors.npz`, under 1 MB). It also builds `movies_meta.pkl` with the details shown in the UI.
