"""One-time preprocessing: shrink the similarity matrix and build movie metadata.

Produces:
  neighbors.npz    - top-K most similar movies (indices + scores) per movie, ~1 MB
                     instead of the 185 MB full similarity matrix.
  movies_meta.pkl  - title, genres, year, rating, runtime, overview, cast, director
                     so the app can filter and display details without API calls.

Run: python build_artifacts.py
"""
import ast
import pickle

import numpy as np
import pandas as pd

TOP_K = 50

movies = pickle.load(open("movie_list.pkl", "rb")).reset_index(drop=True)

# --- Top-K neighbours ---------------------------------------------------------
similarity = np.asarray(pickle.load(open("similarity.pkl", "rb")), dtype=np.float32)
np.fill_diagonal(similarity, -1)  # never recommend the movie itself
top = np.argpartition(-similarity, TOP_K, axis=1)[:, :TOP_K]
top_scores = np.take_along_axis(similarity, top, axis=1)
order = np.argsort(-top_scores, axis=1)
indices = np.take_along_axis(top, order, axis=1).astype(np.int32)
scores = np.take_along_axis(top_scores, order, axis=1).astype(np.float16)
np.savez_compressed("neighbors.npz", indices=indices, scores=scores)
del similarity


# --- Metadata -----------------------------------------------------------------
def names(text, limit=None):
    try:
        items = [item["name"] for item in ast.literal_eval(text)]
    except (ValueError, SyntaxError):
        return []
    return items[:limit] if limit else items


def director(text):
    try:
        return next((c["name"] for c in ast.literal_eval(text) if c.get("job") == "Director"), "")
    except (ValueError, SyntaxError):
        return ""


raw = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
raw = raw.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id")

meta = pd.DataFrame({
    "movie_id": raw["id"],
    "genres": raw["genres"].apply(names),
    "year": pd.to_datetime(raw["release_date"], errors="coerce").dt.year.astype("Int64"),
    "rating": raw["vote_average"].round(1),
    "votes": raw["vote_count"],
    "runtime": raw["runtime"].fillna(0).astype(int),
    "popularity": raw["popularity"],
    "overview": raw["overview"].fillna(""),
    "tagline": raw["tagline"].fillna(""),
    "cast": raw["cast"].apply(lambda t: names(t, 5)),
    "director": raw["crew"].apply(director),
}).drop_duplicates("movie_id")

# Keep row order identical to movie_list.pkl so neighbour indices stay valid.
out = movies[["movie_id", "title"]].merge(meta, on="movie_id", how="left")
assert len(out) == len(movies) and (out["movie_id"].values == movies["movie_id"].values).all()
out["genres"] = out["genres"].apply(lambda g: g if isinstance(g, list) else [])
out["cast"] = out["cast"].apply(lambda c: c if isinstance(c, list) else [])
out.to_pickle("movies_meta.pkl")

print(f"neighbors.npz: {indices.shape}, movies_meta.pkl: {out.shape}")
