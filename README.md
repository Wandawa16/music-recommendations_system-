# 🎵 Music Recommendation System

A content-based music recommendation engine built with Python and Streamlit, powered by Spotify audio features.

---

## 📸 Features

- **Similar Song Finder** — Discover songs that match the audio DNA of any track using cosine similarity
- **Genre & Mood Filter** — Browse music by genre with 8 built-in mood presets (Happy, Calm, Party, Focus, and more)
- **Artist Search** — Find all tracks by any artist with audio feature visualizations
- **Popularity Charts** — Ranked leaderboards globally or by genre
- **Interactive Dashboard** — Clean Streamlit UI with Plotly charts and radar graphs

---

## 🗂️ Project Structure

```
music-recommender/
├── app.py                  # Streamlit dashboard (main entry point)
├── music_recommender.py    # Core recommendation engine (Python class)
├── demo.py                 # Script-based usage examples
├── requirements.txt        # Python dependencies
└── spotify_tracks.csv      # Your dataset (not included)
```

---

## 🚀 Getting Started

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/music-recommender.git
cd music-recommender
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your dataset

Place your Spotify tracks CSV file in the project root. The file should have these columns:

| Column | Type | Description |
|---|---|---|
| `track_id` | string | Unique track identifier |
| `track_name` | string | Name of the song |
| `artists` | string | Artist name(s) |
| `album_name` | string | Album name |
| `track_genre` | string | Genre label |
| `popularity` | int | Popularity score (0–100) |
| `danceability` | float | How suitable for dancing (0–1) |
| `energy` | float | Intensity and activity (0–1) |
| `valence` | float | Musical positivity (0–1) |
| `acousticness` | float | Acoustic confidence (0–1) |
| `instrumentalness` | float | Predicts no vocals (0–1) |
| `speechiness` | float | Spoken words presence (0–1) |
| `liveness` | float | Audience presence detection (0–1) |
| `loudness` | float | Overall loudness in dB |
| `tempo` | float | Estimated tempo in BPM |

> **Dataset source:** [Spotify Tracks Dataset on Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — 114,000 tracks across 114 genres.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Use the sidebar to upload your CSV.

---

## 🧠 How It Works

The system uses **content-based filtering** — it recommends songs based on how similar their audio features are to a seed track.

### Recommendation Algorithm

1. All 9 audio features are scaled to `[0, 1]` using `MinMaxScaler`
2. Each track becomes a feature vector in 9-dimensional space
3. **Cosine similarity** is computed between the seed track and every other track
4. The top-N closest tracks are returned as recommendations

```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```

A similarity of `1.0` = identical audio profile. A similarity of `0.0` = completely different.

### Mood Presets

Moods map to audio feature ranges:

| Mood | Key Features |
|---|---|
| 😊 Happy | valence > 0.6, energy > 0.5 |
| 😢 Sad | valence < 0.4, energy < 0.5 |
| ⚡ Energetic | energy > 0.7, danceability > 0.5 |
| 🌿 Calm | energy < 0.4, acousticness > 0.5 |
| 🎉 Party | danceability > 0.7, energy > 0.6 |
| 🎯 Focus | instrumentalness > 0.3, speechiness < 0.2 |
| 💕 Romantic | valence 0.4–0.8, acousticness > 0.3 |
| 🔥 Aggressive | energy > 0.8 |

---

## 🐍 Using the Python Class Directly

You can use `music_recommender.py` without the Streamlit UI:

```python
from music_recommender import MusicRecommender

rec = MusicRecommender("spotify_tracks.csv")

# Find similar songs
rec.recommend_similar("Blinding Lights", artist="The Weeknd", n=10)

# Filter by genre and mood
rec.filter_by_genre(genre="hip-hop", mood="energetic", n=15)

# Search an artist
rec.search_by_artist("Taylor Swift")

# Popularity charts
rec.popularity_charts(genre="pop", n=10)

# Dataset overview
rec.stats()
```

See `demo.py` for more usage examples.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web dashboard UI |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Feature scaling + cosine similarity |
| `numpy` | Numerical operations |
| `plotly` | Interactive charts and radar graphs |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🔧 Customisation

### Add a new mood preset

In `music_recommender.py`, add an entry to `MOOD_PRESETS`:

```python
MOOD_PRESETS = {
    ...
    "workout": {"energy": (0.75, 1.0), "tempo": (120, 200)},
}
```

### Change the audio features used for similarity

Edit the `AUDIO_FEATURES` list in both `music_recommender.py` and `app.py`:

```python
AUDIO_FEATURES = [
    "danceability", "energy", "valence", "tempo"  # use fewer features
]
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Acknowledgements

- Dataset: [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) by Maharshi Pandya on Kaggle
- Audio features powered by the [Spotify Web API](https://developer.spotify.com/documentation/web-api)
