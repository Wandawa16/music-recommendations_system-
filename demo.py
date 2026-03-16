"""
demo.py — Quick-start examples for the Music Recommender
=========================================================
Run this file to see all 4 features in action.

  python demo.py

Make sure your CSV file path is correct below.
"""

from music_recommender import MusicRecommender

# ── Load your dataset ──────────────────────────────────────────
# Change this path to wherever your CSV is saved
rec = MusicRecommender("spotify_tracks.csv")

# Print a summary of what's loaded
rec.stats()

# ── 1. Recommend similar songs ────────────────────────────────
# Basic: find 10 songs that sound like this track
rec.recommend_similar("Shape of You")

# With artist to avoid wrong-track matches
rec.recommend_similar("Blinding Lights", artist="The Weeknd", n=5)

# Restrict recommendations to the same genre
rec.recommend_similar("God's Plan", same_genre=True, n=8)

# ── 2. Filter by genre / mood ─────────────────────────────────
# Browse all available genres
rec.list_genres()

# Browse all mood presets
rec.list_moods()

# Top 15 hip-hop tracks with an energetic vibe
rec.filter_by_genre(genre="hip-hop", mood="energetic", n=15)

# Calm acoustic tracks (great for studying)
rec.filter_by_genre(mood="calm", n=10, sort_by="acousticness")

# Just a genre, sorted by danceability
rec.filter_by_genre(genre="latin", n=10, sort_by="danceability")

# ── 3. Search by artist ───────────────────────────────────────
# All Drake songs, sorted by popularity
rec.search_by_artist("Drake")

# Partial name works too
rec.search_by_artist("Billie", n=10)

# Sort by danceability instead
rec.search_by_artist("Bad Bunny", sort_by="danceability", n=10)

# ── 4. Popularity charts ──────────────────────────────────────
# Global top 20
rec.popularity_charts()

# Top 10 pop songs
rec.popularity_charts(genre="pop", n=10)

# Only highly popular tracks (score >= 80)
rec.popularity_charts(min_popularity=80, n=30)