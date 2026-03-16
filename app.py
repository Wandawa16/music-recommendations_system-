"""
app.py — Streamlit Music Recommendation Dashboard
==================================================
Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Dark ambient background */
.stApp {
    background: #0a0a0f;
    color: #e8e6f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111118;
    border-right: 1px solid #2a2a3a;
}

/* Cards */
.music-card {
    background: linear-gradient(135deg, #161622 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.music-card:hover { border-color: #7c5cfc; }

.rank-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #7c5cfc;
    opacity: 0.5;
    line-height: 1;
}

.track-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #f0eeff;
    margin: 0;
}

.track-meta {
    font-size: 0.8rem;
    color: #8888aa;
    margin: 2px 0 0;
}

.similarity-pill {
    background: #7c5cfc22;
    border: 1px solid #7c5cfc55;
    color: #b09dff;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    display: inline-block;
}

.popularity-bar-wrap {
    background: #2a2a3a;
    border-radius: 4px;
    height: 4px;
    margin-top: 6px;
}
.popularity-bar {
    background: linear-gradient(90deg, #7c5cfc, #e05cfc);
    border-radius: 4px;
    height: 4px;
}

.stat-box {
    background: #161622;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #7c5cfc;
}
.stat-label {
    font-size: 0.75rem;
    color: #8888aa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Inputs */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] select {
    background: #161622 !important;
    border-color: #2a2a3a !important;
    color: #e8e6f0 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c5cfc, #e05cfc);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Tab styling */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: #8888aa;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #b09dff;
    border-bottom-color: #7c5cfc;
}

/* Divider */
hr { border-color: #2a2a3a; }

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #7c5cfc;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

MOOD_PRESETS = {
    "😊 Happy":      {"valence": (0.6, 1.0), "energy": (0.5, 1.0)},
    "😢 Sad":        {"valence": (0.0, 0.4), "energy": (0.0, 0.5)},
    "⚡ Energetic":  {"energy": (0.7, 1.0), "danceability": (0.5, 1.0)},
    "🌿 Calm":       {"energy": (0.0, 0.4), "acousticness": (0.5, 1.0)},
    "🎉 Party":      {"danceability": (0.7, 1.0), "energy": (0.6, 1.0)},
    "🎯 Focus":      {"instrumentalness": (0.3, 1.0), "speechiness": (0.0, 0.2)},
    "💕 Romantic":   {"valence": (0.4, 0.8), "acousticness": (0.3, 1.0)},
    "🔥 Aggressive": {"energy": (0.8, 1.0)},
}

# ── Data loading ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str):
    df = pd.read_csv(path)
    df = (df
          .drop_duplicates(subset="track_id")
          .dropna(subset=["track_name", "artists"] + AUDIO_FEATURES)
          .reset_index(drop=True))
    df["track_name_lower"] = df["track_name"].str.lower().str.strip()
    df["artists_lower"]    = df["artists"].str.lower().str.strip()
    return df

@st.cache_data(show_spinner=False)
def build_matrix(df: pd.DataFrame):
    scaler = MinMaxScaler()
    matrix = scaler.fit_transform(df[AUDIO_FEATURES])
    return matrix

# ── Helper: render track card ─────────────────────────────────
def track_card(rank, row, show_similarity=False, show_features=False):
    sim_html = ""
    if show_similarity and "similarity" in row:
        sim_html = f'<span class="similarity-pill">⟳ {row["similarity"]:.0%} match</span>'

    pop = int(row.get("popularity", 0))
    bar_w = pop

    feat_html = ""
    if show_features:
        feats = ["danceability", "energy", "valence"]
        feat_html = "<div style='margin-top:8px;display:flex;gap:12px;'>"
        for f in feats:
            if f in row:
                v = row[f]
                feat_html += (
                    f"<span style='font-size:0.7rem;color:#8888aa;'>"
                    f"{f[:5].title()} <span style='color:#b09dff;font-weight:500;'>{v:.2f}</span></span>"
                )
        feat_html += "</div>"

    st.markdown(f"""
    <div class="music-card">
      <div style="display:flex;align-items:flex-start;gap:14px;">
        <div class="rank-num">{rank:02d}</div>
        <div style="flex:1;min-width:0;">
          <div class="track-title">{row['track_name']}</div>
          <div class="track-meta">{row['artists']} · <span style="color:#7c5cfc55;background:#7c5cfc18;padding:1px 7px;border-radius:20px;font-size:0.7rem;">{row.get('track_genre','—')}</span></div>
          <div class="popularity-bar-wrap"><div class="popularity-bar" style="width:{bar_w}%"></div></div>
          {feat_html}
        </div>
        <div style="text-align:right;flex-shrink:0;">
          {sim_html}
          <div style="font-size:0.75rem;color:#8888aa;margin-top:4px;">★ {pop}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar: file upload + stats ──────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">Dataset</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        with st.spinner("Loading…"):
            df = load_data(uploaded)
            matrix = build_matrix(df)
        st.success(f"Loaded {len(df):,} tracks")

        st.markdown("---")
        st.markdown('<p class="section-header">Stats</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{len(df)//1000}K</div><div class="stat-label">Tracks</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{df["track_genre"].nunique()}</div><div class="stat-label">Genres</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{df["artists"].nunique()//1000}K</div><div class="stat-label">Artists</div></div>', unsafe_allow_html=True)
        with col4:
            avg_pop = int(df["popularity"].mean())
            st.markdown(f'<div class="stat-box"><div class="stat-num">{avg_pop}</div><div class="stat-label">Avg Pop.</div></div>', unsafe_allow_html=True)

    else:
        st.info("Upload your Spotify CSV to get started.")
        df = None
        matrix = None

# ── Main area ─────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1.5rem;">
  <h1 style="font-size:2.4rem;font-weight:800;margin:0;background:linear-gradient(135deg,#b09dff,#e05cfc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    Music Recommender
  </h1>
  <p style="color:#8888aa;margin:6px 0 0;font-size:0.95rem;">
    Content-based recommendations powered by audio features
  </p>
</div>
""", unsafe_allow_html=True)

if df is None:
    st.markdown("""
    <div style="background:#161622;border:1px dashed #2a2a4a;border-radius:16px;padding:3rem;text-align:center;margin-top:2rem;">
      <div style="font-size:3rem;margin-bottom:1rem;">🎵</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.2rem;color:#f0eeff;font-weight:600;">Upload your dataset to begin</div>
      <div style="color:#8888aa;margin-top:0.5rem;font-size:0.9rem;">Use the sidebar to upload your Spotify CSV file</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔀  Similar Songs",
    "🎭  Genre & Mood",
    "🔍  Artist Search",
    "📈  Charts",
])

# ─────────────────────────────────────────────
# TAB 1 — Similar Songs
# ─────────────────────────────────────────────
with tab1:
    st.markdown("### Find similar songs")
    st.caption("Enter a track name and we'll find songs with matching audio DNA.")

    col_a, col_b, col_c = st.columns([3, 2, 1])
    with col_a:
        track_input = st.text_input("Track name", placeholder="e.g. Blinding Lights")
    with col_b:
        artist_input = st.text_input("Artist (optional)", placeholder="e.g. The Weeknd")
    with col_c:
        n_similar = st.number_input("Results", min_value=5, max_value=50, value=10)

    same_genre = st.checkbox("Same genre only")

    if st.button("Find similar songs", key="btn_similar"):
        if not track_input.strip():
            st.warning("Please enter a track name.")
        else:
            name_lower = track_input.lower().strip()
            mask = df["track_name_lower"] == name_lower
            if artist_input.strip():
                mask = mask & df["artists_lower"].str.contains(artist_input.lower().strip(), na=False)
            matches = df[mask]
            if matches.empty:
                mask = df["track_name_lower"].str.contains(name_lower, na=False)
                matches = df[mask]

            if matches.empty:
                st.error(f"Track '{track_input}' not found. Try checking the spelling.")
            else:
                idx = matches["popularity"].idxmax()
                seed = df.loc[idx]
                seed_vec = matrix[idx].reshape(1, -1)
                scores = cosine_similarity(seed_vec, matrix)[0]

                candidates = df.copy()
                candidates["similarity"] = scores
                candidates = candidates[candidates.index != idx]
                if same_genre:
                    candidates = candidates[candidates["track_genre"] == seed["track_genre"]]

                results = candidates.nlargest(n_similar, "similarity")

                # Seed track info
                st.markdown(f"""
                <div style="background:#1a1a2e;border:1px solid #7c5cfc44;border-radius:12px;padding:1rem 1.4rem;margin:1rem 0;">
                  <div style="font-size:0.65rem;color:#7c5cfc;text-transform:uppercase;letter-spacing:0.12em;font-weight:600;">Seed track</div>
                  <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#f0eeff;margin-top:4px;">{seed['track_name']}</div>
                  <div style="color:#8888aa;font-size:0.85rem;">{seed['artists']} · {seed['track_genre']} · ★ {int(seed['popularity'])}</div>
                </div>
                """, unsafe_allow_html=True)

                # Radar chart of seed track
                feat_labels = ["Dance", "Energy", "Speech", "Acoustic", "Instrum.", "Live", "Valence"]
                feat_cols   = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence"]
                vals = [seed[f] for f in feat_cols]
                vals_norm = vals + [vals[0]]
                labels_norm = feat_labels + [feat_labels[0]]

                fig = go.Figure(go.Scatterpolar(
                    r=vals_norm, theta=labels_norm, fill='toself',
                    fillcolor='rgba(124,92,252,0.15)',
                    line=dict(color='#7c5cfc', width=2),
                ))
                fig.update_layout(
                    polar=dict(
                        bgcolor='#161622',
                        radialaxis=dict(visible=True, range=[0,1], gridcolor='#2a2a3a', tickfont=dict(color='#8888aa', size=9)),
                        angularaxis=dict(gridcolor='#2a2a3a', tickfont=dict(color='#b09dff', size=10)),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    margin=dict(l=40,r=40,t=20,b=20),
                    height=260,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('<p class="section-header">Recommendations</p>', unsafe_allow_html=True)
                for rank, (_, row) in enumerate(results.iterrows(), 1):
                    track_card(rank, row, show_similarity=True, show_features=True)

# ─────────────────────────────────────────────
# TAB 2 — Genre & Mood
# ─────────────────────────────────────────────
with tab2:
    st.markdown("### Browse by genre & mood")

    col_g, col_m, col_n2 = st.columns([2, 2, 1])
    with col_g:
        genres = ["All genres"] + sorted(df["track_genre"].unique())
        genre_sel = st.selectbox("Genre", genres)
    with col_m:
        mood_sel = st.selectbox("Mood", ["Any mood"] + list(MOOD_PRESETS.keys()))
    with col_n2:
        n_filter = st.number_input("Results", min_value=5, max_value=100, value=20, key="n2")

    sort_opts = ["popularity", "danceability", "energy", "valence", "acousticness", "tempo"]
    sort_sel = st.selectbox("Sort by", sort_opts)

    if st.button("Apply filters", key="btn_filter"):
        results = df.copy()
        if genre_sel != "All genres":
            results = results[results["track_genre"] == genre_sel]
        if mood_sel != "Any mood":
            for feat, (lo, hi) in MOOD_PRESETS[mood_sel].items():
                results = results[results[feat].between(lo, hi)]

        if results.empty:
            st.warning("No tracks match these filters. Try loosening the mood or genre.")
        else:
            st.caption(f"{len(results):,} tracks matched · showing top {min(n_filter, len(results))}")
            results = results.sort_values(sort_sel, ascending=False).head(n_filter)

            # Genre distribution pie
            genre_counts = results["track_genre"].value_counts().head(8)
            fig2 = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                color_discrete_sequence=px.colors.sequential.Purples_r,
                hole=0.55,
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='#8888aa', size=10)),
                margin=dict(l=0,r=0,t=10,b=0),
                height=220,
            )
            st.plotly_chart(fig2, use_container_width=True)

            for rank, (_, row) in enumerate(results.iterrows(), 1):
                track_card(rank, row, show_features=True)

# ─────────────────────────────────────────────
# TAB 3 — Artist Search
# ─────────────────────────────────────────────
with tab3:
    st.markdown("### Search by artist")

    col_ar, col_nar = st.columns([4, 1])
    with col_ar:
        artist_search = st.text_input("Artist name", placeholder="e.g. Taylor Swift, Drake, Bad Bunny")
    with col_nar:
        n_artist = st.number_input("Results", min_value=5, max_value=50, value=15, key="n3")

    sort_artist = st.selectbox("Sort by", sort_opts, key="sort_artist")

    if st.button("Search artist", key="btn_artist"):
        if not artist_search.strip():
            st.warning("Please enter an artist name.")
        else:
            mask = df["artists_lower"].str.contains(artist_search.lower().strip(), na=False)
            results = df[mask]
            total = len(results)

            if results.empty:
                st.error(f"No tracks found for '{artist_search}'.")
            else:
                results = results.sort_values(sort_artist, ascending=False).head(n_artist)

                st.markdown(f"""
                <div style="background:#161622;border:1px solid #2a2a4a;border-radius:10px;padding:0.8rem 1.2rem;margin-bottom:1rem;display:inline-block;">
                  <span style="font-family:'Syne',sans-serif;font-weight:700;color:#f0eeff;">{artist_search}</span>
                  <span style="color:#8888aa;font-size:0.85rem;margin-left:10px;">{total:,} tracks in dataset</span>
                </div>
                """, unsafe_allow_html=True)

                # Scatter: energy vs valence
                fig3 = px.scatter(
                    results, x="valence", y="energy",
                    size="popularity", color="track_genre",
                    hover_data=["track_name"],
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    labels={"valence": "Valence (happiness)", "energy": "Energy"},
                )
                fig3.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#161622',
                    font=dict(color='#8888aa'),
                    xaxis=dict(gridcolor='#2a2a3a', range=[0,1]),
                    yaxis=dict(gridcolor='#2a2a3a', range=[0,1]),
                    legend=dict(font=dict(color='#8888aa', size=10)),
                    margin=dict(l=0,r=0,t=10,b=0),
                    height=280,
                )
                st.plotly_chart(fig3, use_container_width=True)

                for rank, (_, row) in enumerate(results.iterrows(), 1):
                    track_card(rank, row, show_features=True)

# ─────────────────────────────────────────────
# TAB 4 — Popularity Charts
# ─────────────────────────────────────────────
with tab4:
    st.markdown("### Popularity charts")

    col_cg, col_cn, col_cm = st.columns([2, 1, 1])
    with col_cg:
        chart_genre = st.selectbox("Genre", ["All genres"] + sorted(df["track_genre"].unique()), key="cg")
    with col_cn:
        n_chart = st.number_input("Results", min_value=5, max_value=100, value=20, key="n4")
    with col_cm:
        min_pop = st.slider("Min popularity", 0, 100, 0)

    if st.button("Show charts", key="btn_charts"):
        results = df.copy()
        if chart_genre != "All genres":
            results = results[results["track_genre"] == chart_genre]
        results = results[results["popularity"] >= min_pop]
        results = (results
                   .sort_values("popularity", ascending=False)
                   .drop_duplicates(subset=["track_name","artists"])
                   .head(n_chart))

        if results.empty:
            st.warning("No tracks match the filters.")
        else:
            # Horizontal bar chart
            fig4 = go.Figure(go.Bar(
                x=results["popularity"].values[::-1],
                y=(results["track_name"] + " – " + results["artists"]).values[::-1],
                orientation='h',
                marker=dict(
                    color=results["popularity"].values[::-1],
                    colorscale=[[0,'#2a1a5e'],[0.5,'#7c5cfc'],[1,'#e05cfc']],
                ),
                text=results["popularity"].values[::-1],
                textposition='outside',
                textfont=dict(color='#8888aa', size=10),
            ))
            fig4.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#161622',
                font=dict(color='#8888aa', size=10),
                xaxis=dict(gridcolor='#2a2a3a', range=[0,105]),
                yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10, color='#c0bedc')),
                margin=dict(l=0, r=50, t=10, b=0),
                height=max(350, n_chart * 22),
            )
            st.plotly_chart(fig4, use_container_width=True)

            st.markdown('<p class="section-header">Track list</p>', unsafe_allow_html=True)
            for rank, (_, row) in enumerate(results.iterrows(), 1):
                track_card(rank, row, show_features=True)