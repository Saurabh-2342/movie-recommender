import html
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title="CineMatch · Movie Recommender", page_icon="🎬", layout="wide")

TMDB_API = "https://api.themoviedb.org/3"
POSTER_URL = "https://image.tmdb.org/t/p/w500"
BACKDROP_URL = "https://image.tmdb.org/t/p/w1280"
CARD_WIDTH = 185
ROW_LENGTH = 15


# ---------------------------------------------------------------- data ------
@st.cache_resource
def load_data():
    """Loaded once per server, not on every rerun. Built by build_artifacts.py."""
    movies = pd.read_pickle("movies_meta.pkl")
    neighbors = np.load("neighbors.npz")
    return movies, neighbors["indices"], neighbors["scores"]


def tmdb_token():
    try:
        return st.secrets["TMDB_TOKEN"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("TMDB_TOKEN")


@st.cache_resource
def http_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries, pool_maxsize=16))
    session.headers.update({"accept": "application/json", "Authorization": f"Bearer {tmdb_token()}"})
    return session


@st.cache_resource
def details_cache():
    return {}


EMPTY_DETAILS = {"poster": None, "backdrop": None, "imdb_id": None, "trailer": None}


def _fetch_details(session, movie_id):
    empty = EMPTY_DETAILS
    try:
        response = session.get(
            f"{TMDB_API}/movie/{movie_id}", params={"append_to_response": "videos"}, timeout=10
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException:
        return empty

    videos = [v for v in data.get("videos", {}).get("results", []) if v.get("site") == "YouTube"]
    trailers = sorted(
        (v for v in videos if v.get("type") == "Trailer"), key=lambda v: not v.get("official", False)
    )
    trailer = (trailers or videos or [None])[0]
    return {
        "poster": f"{POSTER_URL}{data['poster_path']}" if data.get("poster_path") else None,
        "backdrop": f"{BACKDROP_URL}{data['backdrop_path']}" if data.get("backdrop_path") else None,
        "imdb_id": data.get("imdb_id"),
        "trailer": f"https://www.youtube.com/watch?v={trailer['key']}" if trailer else None,
    }


def fetch_details(movie_ids):
    """Fetch poster/backdrop/trailer for many movies in parallel, cached per movie."""
    if not tmdb_token():
        return {m: EMPTY_DETAILS for m in movie_ids}
    cache = details_cache()
    session = http_session()  # resolve cached resources on the main thread
    missing = [m for m in dict.fromkeys(movie_ids) if m not in cache]
    if missing:
        with ThreadPoolExecutor(max_workers=8) as pool:
            fetched = pool.map(lambda m: _fetch_details(session, m), missing)
            for movie_id, details in zip(missing, fetched):
                if details["poster"] or details["trailer"]:  # don't cache failed lookups
                    cache[movie_id] = details
                else:
                    cache.setdefault(movie_id, details)
    return {m: cache[m] for m in movie_ids}


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_trending():
    if not tmdb_token():
        return []
    try:
        response = http_session().get(f"{TMDB_API}/trending/movie/week", timeout=10)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.exceptions.RequestException:
        return []


movies, neighbor_idx, neighbor_scores = load_data()
title_to_idx = {t: i for i, t in reversed(list(enumerate(movies["title"])))}
id_to_idx = {m: i for i, m in enumerate(movies["movie_id"])}
all_genres = sorted({g for gs in movies["genres"] for g in gs})
years = movies["year"].dropna()
min_year, max_year = int(years.min()), int(years.max())


# ------------------------------------------------------------- state --------
st.session_state.setdefault("selected_idx", title_to_idx.get("Avatar", 0))
st.session_state.setdefault("watchlist", [])
st.session_state.setdefault("person", "")


def select_movie(idx):
    st.session_state.selected_idx = idx
    st.session_state.person = ""


def surprise_me():
    """Jump to a random well-liked movie, honouring the sidebar genre filter."""
    pool = movies[(movies["votes"] >= 300) & (movies["rating"] >= 7.0)]
    genres = st.session_state.get("genre_filter")
    if genres:
        pool = pool[pool["genres"].apply(lambda g: bool(set(genres) & set(g)))]
    pool = pool.index.difference([st.session_state.selected_idx])
    if len(pool):
        select_movie(int(random.choice(pool)))


def toggle_watchlist(movie_id):
    watchlist = st.session_state.watchlist
    if movie_id in watchlist:
        watchlist.remove(movie_id)
    else:
        watchlist.append(movie_id)


# ------------------------------------------------------------- style --------
st.markdown(
    """
<style>
.block-container { padding-top: 2rem; max-width: 1400px; }
.brand { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; }
.brand span { color: #E50914; }
.subtitle { color: #9AA4B2; margin: 0 0 1.2rem; }

.hero { position: relative; border-radius: 20px; overflow: hidden; padding: 32px;
        display: flex; gap: 32px; align-items: flex-start; margin-bottom: 1rem;
        background-size: cover; background-position: center top;
        border: 1px solid rgba(255,255,255,.06); }
.hero-poster { width: 210px; flex-shrink: 0; border-radius: 12px;
               box-shadow: 0 18px 40px rgba(0,0,0,.6); }
.hero h1 { font-size: 2.4rem; line-height: 1.1; margin: 0 0 .3rem; padding: 0; }
.hero .tagline { color: #C9CED8; font-style: italic; margin-bottom: .8rem; }
.hero .meta { color: #C9CED8; margin-bottom: .8rem; font-size: .95rem; }
.hero .overview { max-width: 760px; line-height: 1.6; color: #E1E4EA; }
.hero .credits { margin-top: .9rem; color: #9AA4B2; font-size: .9rem; }
.hero .credits b { color: #E8EAF0; font-weight: 600; }
.star { color: #F5C518; }

.chip { display: inline-block; padding: 3px 11px; margin: 0 6px 6px 0; border-radius: 999px;
        font-size: .75rem; background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.14); color: #E8EAF0; }

.card { background: #161A23; border-radius: 14px; overflow: hidden; margin-bottom: 8px;
        border: 1px solid rgba(255,255,255,.05);
        transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
.card:hover { transform: translateY(-6px); box-shadow: 0 14px 30px rgba(0,0,0,.55);
              border-color: rgba(229,9,20,.55); }
.card .poster { position: relative; aspect-ratio: 2 / 3; background: #232836; }
.card .poster img { width: 100%; height: 100%; object-fit: cover; display: block; }
.card .poster .placeholder { position: absolute; inset: 0; display: flex; align-items: center;
        justify-content: center; text-align: center; padding: 12px; color: #6B7385;
        font-weight: 600; background: linear-gradient(160deg, #232836, #12151C); }
.badge { position: absolute; top: 8px; right: 8px; padding: 3px 8px; border-radius: 8px;
         background: rgba(0,0,0,.78); font-size: .78rem; font-weight: 700; }
.rank { position: absolute; top: 8px; left: 8px; padding: 3px 8px; border-radius: 8px;
        background: #E50914; font-size: .75rem; font-weight: 700; }
.card .body { padding: 10px 12px 12px; }
.card .title { font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card .title a { color: inherit; text-decoration: none; }
.card .title a:hover { color: #E50914; }
.card .sub { color: #9AA4B2; font-size: .78rem; margin: 2px 0 6px; }
.card .reason { color: #7FD1AE; font-size: .74rem; min-height: 1.1rem; white-space: nowrap;
                overflow: hidden; text-overflow: ellipsis; }

.row-heading { font-size: 1.3rem; font-weight: 700; margin: 1.6rem 0 .2rem; }
.row-heading .accent { color: #E50914; }

/* Netflix-style rows: one horizontal line of cards that scrolls sideways */
[class*="st-key-row-"] { flex-wrap: nowrap !important; overflow-x: auto; overflow-y: hidden;
                         padding: 10px 4px 16px; scroll-snap-type: x proximity;
                         scrollbar-width: thin; scrollbar-color: #3A4050 transparent; }
[class*="st-key-row-"]:hover { scrollbar-color: #E50914 transparent; }
[class*="st-key-row-"]::-webkit-scrollbar { height: 8px; }
[class*="st-key-row-"]::-webkit-scrollbar-thumb { background: #3A4050; border-radius: 8px; }
[class*="st-key-row-"]:hover::-webkit-scrollbar-thumb { background: #E50914; }
[class*="st-key-row-"] > * { flex-shrink: 0; scroll-snap-align: start; }

.st-key-surprise button { height: 100%; }

@media (max-width: 700px) {
  .hero { flex-direction: column; padding: 20px; gap: 18px; }
  .hero-poster { width: 140px; }
  .hero h1 { font-size: 1.7rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------- components -------
def esc(text):
    return html.escape(str(text))


def fmt_runtime(minutes):
    return f"{minutes // 60}h {minutes % 60}m" if minutes else ""


def chips(items):
    return "".join(f'<span class="chip">{esc(i)}</span>' for i in items)


def why_recommended(source, target):
    if source["director"] and source["director"] == target["director"]:
        return f"🎬 Also directed by {source['director']}"
    shared_cast = [c for c in target["cast"] if c in source["cast"]]
    if shared_cast:
        return f"🎭 Also stars {shared_cast[0]}"
    shared_genres = [g for g in target["genres"] if g in source["genres"]]
    if len(shared_genres) >= 2:
        return f"✨ {' · '.join(shared_genres[:2])}"
    return "✨ Similar story & themes"


def render_hero(movie, details):
    backdrop = details["backdrop"]
    background = (
        f"background-image: linear-gradient(90deg, rgba(11,13,18,.97) 0%, rgba(11,13,18,.88) 45%, "
        f"rgba(11,13,18,.55) 100%), url('{backdrop}');"
        if backdrop
        else "background: linear-gradient(135deg, #1B1F2A, #0B0D12);"
    )
    poster = (
        f'<img class="hero-poster" src="{details["poster"]}" alt="">' if details["poster"] else ""
    )
    meta = " · ".join(
        str(p)
        for p in [
            movie["year"] if pd.notna(movie["year"]) else "",
            fmt_runtime(movie["runtime"]),
            f'<span class="star">★</span> {movie["rating"]:.1f} ({movie["votes"]:,} votes)',
        ]
        if p
    )
    credits = ""
    if movie["director"]:
        credits += f"<b>Director</b> &nbsp;{esc(movie['director'])}<br>"
    if movie["cast"]:
        credits += f"<b>Starring</b> &nbsp;{esc(', '.join(movie['cast']))}"
    tagline = f'<div class="tagline">“{esc(movie["tagline"])}”</div>' if movie["tagline"] else ""
    st.markdown(
        f"""
<div class="hero" style="{background}">
  {poster}
  <div>
    <h1>{esc(movie["title"])}</h1>
    {tagline}
    <div class="meta">{meta}</div>
    <div>{chips(movie["genres"])}</div>
    <div class="overview">{esc(movie["overview"])}</div>
    <div class="credits">{credits}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_card(movie, details, key, reason="", rank=None):
    """movie: dict with title, year, rating, genres, movie_id."""
    link = f"https://www.imdb.com/title/{details['imdb_id']}" if details.get("imdb_id") else None
    title = esc(movie["title"])
    poster = (
        f'<img src="{details["poster"]}" alt="{title}" loading="lazy">'
        if details.get("poster")
        else f'<div class="placeholder">{title}</div>'
    )
    if link:
        poster = f'<a href="{link}" target="_blank">{poster}</a>'
        title = f'<a href="{link}" target="_blank">{title}</a>'
    rating = movie.get("rating")
    badge = f'<span class="badge"><span class="star">★</span> {rating:.1f}</span>' if rating else ""
    rank_badge = f'<span class="rank">#{rank}</span>' if rank else ""
    year = movie.get("year")
    genres = ", ".join(movie.get("genres", [])[:2])
    parts = [str(int(year))] if pd.notna(year) and year else []
    if genres:
        parts.append(esc(genres))
    sub = " · ".join(parts) or "&nbsp;"
    st.markdown(
        f"""
<div class="card">
  <div class="poster">{poster}{badge}{rank_badge}</div>
  <div class="body">
    <div class="title">{title}</div>
    <div class="sub">{sub}</div>
    <div class="reason">{esc(reason)}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    movie_id = movie["movie_id"]
    in_watchlist = movie_id in st.session_state.watchlist
    actions = st.columns(3, gap="small")
    if movie_id in id_to_idx:
        actions[0].button(
            "🔁", key=f"{key}-similar", help="More like this", width="stretch",
            on_click=select_movie, args=(id_to_idx[movie_id],),
        )
    actions[1].button(
        "✓" if in_watchlist else "＋", key=f"{key}-watch", width="stretch",
        help="Remove from watchlist" if in_watchlist else "Add to watchlist",
        on_click=toggle_watchlist, args=(movie_id,),
    )
    if details.get("trailer"):
        with actions[2].popover("▶", help="Watch trailer", width="stretch"):
            st.video(details["trailer"])


def render_row(heading, items, details, key_prefix, reasons=None, ranked=False, note=None):
    """A Netflix-style horizontally scrolling row of movie cards."""
    if not items:
        return
    st.markdown(f'<div class="row-heading">{heading}</div>', unsafe_allow_html=True)
    if note:
        st.caption(note)
    with st.container(horizontal=True, key=f"row-{key_prefix}", gap="medium"):
        for i, movie in enumerate(items):
            with st.container(key=f"{key_prefix}-card-{i}", width=CARD_WIDTH):
                render_card(
                    movie, details.get(movie["movie_id"], {}), key=f"{key_prefix}-{i}",
                    reason=reasons[i] if reasons else "", rank=i + 1 if ranked else None,
                )


# ------------------------------------------------------------ sidebar -------
with st.sidebar:
    st.markdown("### 🎛️ Refine")
    count = st.slider("Number of recommendations", 5, 20, 10, step=5)
    genre_filter = st.multiselect("Genres", all_genres, placeholder="Any genre", key="genre_filter")
    year_range = st.slider("Release year", min_year, max_year, (min_year, max_year))
    min_rating = st.slider("Minimum rating", 0.0, 9.0, 0.0, step=0.5)
    sort_by = st.radio("Sort by", ["Best match", "Rating", "Newest"], horizontal=True)
    st.divider()
    st.markdown("### 🔎 Find by person")
    st.text_input("Actor or director", key="person", placeholder="e.g. Christopher Nolan")
    st.divider()
    st.caption(f"📌 {len(st.session_state.watchlist)} in watchlist · {len(movies):,} movies")
    if not tmdb_token():
        st.warning("No TMDB_TOKEN set. Posters and trailers are disabled.")


def passes_filters(row):
    if genre_filter and not set(genre_filter) & set(row["genres"]):
        return False
    if pd.isna(row["year"]) or not year_range[0] <= row["year"] <= year_range[1]:
        return False
    return row["rating"] >= min_rating


# ------------------------------------------------------------- header -------
st.markdown('<p class="brand">🎬 Cine<span>Match</span></p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Pick a movie you love. We\'ll find what to watch next.</p>',
    unsafe_allow_html=True,
)

person = st.session_state.person.strip().lower()
if person:
    mask = movies["director"].str.lower().str.contains(person, regex=False) | movies["cast"].apply(
        lambda cast: any(person in c.lower() for c in cast)
    )
    options = movies.index[mask].tolist()
    if not options:
        st.warning(f"No movies found for “{st.session_state.person}”. Showing all titles.")
        options = movies.index.tolist()
else:
    options = movies.index.tolist()
if st.session_state.selected_idx not in options:
    st.session_state.selected_idx = options[0]


def label(i):
    year = movies.at[i, "year"]
    return f"{movies.at[i, 'title']} ({year})" if pd.notna(year) else movies.at[i, "title"]


search_col, surprise_col = st.columns([6, 1], vertical_alignment="center")
selected_idx = search_col.selectbox(
    "Search a movie", options, key="selected_idx", format_func=label,
    label_visibility="collapsed", placeholder="Search a movie…",
)
surprise_col.button(
    "🎲 Surprise me", key="surprise", width="stretch", on_click=surprise_me,
    help="Jump to a random highly-rated movie (respects your genre filter)",
)
selected = movies.loc[selected_idx]

# --- Recommendations (computed from the precomputed neighbour list) ---------
candidates = [
    (movies.loc[j], float(s))
    for j, s in zip(neighbor_idx[selected_idx], neighbor_scores[selected_idx])
    if passes_filters(movies.loc[j])
]
if sort_by == "Rating":
    candidates = sorted(candidates[: count * 2], key=lambda c: -c[0]["rating"])
elif sort_by == "Newest":
    candidates = sorted(candidates[: count * 2], key=lambda c: -(c[0]["year"] or 0))
recs = [c[0] for c in candidates[:count]]



def pick(frame, sort_col="rating"):
    """Rows of `frame` that pass the sidebar filters, excluding the selected movie."""
    frame = frame[frame.index != selected_idx].sort_values(sort_col, ascending=False)
    return [r for _, r in frame.iterrows() if passes_filters(r)][:ROW_LENGTH]


by_director = (
    pick(movies[movies["director"] == selected["director"]]) if selected["director"] else []
)
main_genre = selected["genres"][0] if selected["genres"] else None
by_genre = (
    pick(movies[(movies["votes"] >= 500) & movies["genres"].apply(lambda g: main_genre in g)])
    if main_genre
    else []
)
top_rated = pick(movies[movies["votes"] >= 1000])
watchlist_rows = [movies.loc[id_to_idx[m]] for m in st.session_state.watchlist if m in id_to_idx]

with st.spinner("Fetching posters…"):
    details = fetch_details(
        [selected["movie_id"]]
        + [r["movie_id"] for r in recs + by_director + by_genre + top_rated + watchlist_rows]
    )

# ------------------------------------------------------------- hero ---------
render_hero(selected, details[selected["movie_id"]])
hero_details = details[selected["movie_id"]]
hero_actions = st.columns([1.3, 1.3, 1.3, 6])
in_list = selected["movie_id"] in st.session_state.watchlist
hero_actions[0].button(
    "✓ In watchlist" if in_list else "＋ Watchlist", key="hero-watch", width="stretch",
    type="secondary" if in_list else "primary",
    on_click=toggle_watchlist, args=(selected["movie_id"],),
)
if hero_details["trailer"]:
    with hero_actions[1].popover("▶ Trailer", width="stretch"):
        st.video(hero_details["trailer"])
if hero_details["imdb_id"]:
    hero_actions[2].link_button(
        "IMDb ↗", f"https://www.imdb.com/title/{hero_details['imdb_id']}", width="stretch"
    )

# ------------------------------------------------------------- rows ---------
title = esc(selected["title"])
if recs:
    render_row(
        f'Because you picked <span class="accent">{title}</span>',
        [r.to_dict() for r in recs], details, "rec",
        reasons=[why_recommended(selected, r) for r in recs],
    )
else:
    st.info("No similar movies match your filters. Try widening them in the sidebar.")

if watchlist_rows:
    render_row("📌 My watchlist", [r.to_dict() for r in watchlist_rows], details, "watch")

render_row(
    f'More from <span class="accent">{esc(selected["director"])}</span>',
    [r.to_dict() for r in by_director], details, "director",
)
render_row(
    f'Top-rated <span class="accent">{esc(main_genre)}</span>',
    [r.to_dict() for r in by_genre], details, "genre",
)

trending = fetch_trending()
if trending:
    items = [
        {
            "movie_id": t["id"],
            "title": t.get("title", ""),
            "year": int(t["release_date"][:4]) if t.get("release_date") else None,
            "rating": t.get("vote_average"),
            "genres": [],
        }
        for t in trending[:ROW_LENGTH]
    ]
    trending_details = {
        t["id"]: {
            "poster": f"{POSTER_URL}{t['poster_path']}" if t.get("poster_path") else None,
            "imdb_id": None,
            "trailer": None,
        }
        for t in trending
    }
    render_row(
        "🔥 Trending this week", items, trending_details, "trend", ranked=True,
        note="🔁 appears on movies that are in the recommendation dataset (up to 2017).",
    )
else:
    popular = movies.sort_values("popularity", ascending=False).head(ROW_LENGTH)
    popular = [r.to_dict() for _, r in popular.iterrows()]
    render_row(
        "🔥 Most popular", popular, fetch_details([p["movie_id"] for p in popular]), "pop",
        ranked=True,
    )

render_row(
    "⭐ All-time top rated", [r.to_dict() for r in top_rated], details, "top", ranked=True
)
