"""
Music Recommendation System
=============================
Features:
  1. recommend_similar()   - Content-based similarity using audio features
  2. filter_by_genre()     - Filter & rank songs by genre/mood
  3. search_by_artist()    - Find all songs by an artist
  4. popularity_charts()   - Top songs overall or by genre

Usage:
  from music_recommender import MusicRecommender
  rec = MusicRecommender("spotify_tracks.csv")
  rec.recommend_similar("Shape of You")
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# Audio features used for similarity scoring
# ─────────────────────────────────────────────
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

# Mood presets: map a vibe to a filter on audio features
MOOD_PRESETS = {
    "happy":       {"valence": (0.6, 1.0), "energy": (0.5, 1.0)},
    "sad":         {"valence": (0.0, 0.4), "energy": (0.0, 0.5)},
    "energetic":   {"energy": (0.7, 1.0), "danceability": (0.5, 1.0)},
    "calm":        {"energy": (0.0, 0.4), "acousticness": (0.5, 1.0)},
    "party":       {"danceability": (0.7, 1.0), "energy": (0.6, 1.0)},
    "focus":       {"instrumentalness": (0.3, 1.0), "speechiness": (0.0, 0.2)},
    "romantic":    {"valence": (0.4, 0.8), "acousticness": (0.3, 1.0)},
    "aggressive":  {"energy": (0.8, 1.0), "loudness": (-5, 0)},
}


class MusicRecommender:
    """Content-based music recommendation engine."""

    def __init__(self, csv_path: str):
        """
        Load the dataset and precompute the scaled feature matrix.

        Args:
            csv_path: Path to the Spotify tracks CSV file.
        """
        print("Loading dataset...")
        self.df = pd.read_csv(csv_path)
        self._clean()
        self._build_feature_matrix()
        print(f"Ready! Loaded {len(self.df):,} tracks across "
              f"{self.df['track_genre'].nunique()} genres.\n")

    # ──────────────────────────────────────────
    # Internal setup
    # ──────────────────────────────────────────

    def _clean(self):
        """Drop duplicates and null rows, reset index."""
        self.df = (
            self.df
            .drop_duplicates(subset="track_id")
            .dropna(subset=["track_name", "artists"] + AUDIO_FEATURES)
            .reset_index(drop=True)
        )
        # Normalise text columns for reliable lookups
        self.df["track_name_lower"] = self.df["track_name"].str.lower().str.strip()
        self.df["artists_lower"]    = self.df["artists"].str.lower().str.strip()

    def _build_feature_matrix(self):
        """Scale audio features to [0,1] and store as a numpy matrix."""
        scaler = MinMaxScaler()
        self._feature_matrix = scaler.fit_transform(self.df[AUDIO_FEATURES])
        self._scaler = scaler

    def _find_track_index(self, track_name: str, artist: str = None) -> int:
        """
        Return the DataFrame index for the best matching track.
        Raises ValueError if not found.
        """
        name_lower = track_name.lower().strip()
        mask = self.df["track_name_lower"] == name_lower

        if artist:
            artist_lower = artist.lower().strip()
            mask = mask & self.df["artists_lower"].str.contains(artist_lower, na=False)

        matches = self.df[mask]
        if matches.empty:
            # Fuzzy fallback: partial name match
            mask = self.df["track_name_lower"].str.contains(name_lower, na=False)
            matches = self.df[mask]

        if matches.empty:
            raise ValueError(
                f"Track '{track_name}' not found. "
                "Try checking the spelling or use search_by_artist() first."
            )

        # If multiple hits, pick the most popular one
        return matches["popularity"].idxmax()

    @staticmethod
    def _print_table(results: pd.DataFrame, title: str):
        """Pretty-print a results table to the terminal."""
        cols = ["track_name", "artists", "track_genre", "popularity"]
        cols = [c for c in cols if c in results.columns]
        display = results[cols].copy()
        display.columns = [c.replace("_", " ").title() for c in cols]
        display.index = range(1, len(display) + 1)

        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
        print(display.to_string())
        print(f"{'─'*60}\n")

    # ──────────────────────────────────────────
    # 1. Recommend similar songs
    # ──────────────────────────────────────────

    def recommend_similar(
        self,
        track_name: str,
        artist: str = None,
        n: int = 10,
        same_genre: bool = False,
    ) -> pd.DataFrame:
        """
        Recommend songs that sound similar to the given track.

        Uses cosine similarity across all 9 scaled audio features.

        Args:
            track_name:  Name of the seed track.
            artist:      Optional artist to narrow the lookup.
            n:           Number of recommendations to return (default 10).
            same_genre:  If True, restrict results to the same genre.

        Returns:
            DataFrame with top-n similar tracks + a 'similarity' score (0–1).

        Example:
            rec.recommend_similar("Blinding Lights", n=5)
            rec.recommend_similar("Bohemian Rhapsody", same_genre=True)
        """
        idx = self._find_track_index(track_name, artist)
        seed = self.df.loc[idx]
        seed_vec = self._feature_matrix[idx].reshape(1, -1)

        # Compute cosine similarity against every track
        scores = cosine_similarity(seed_vec, self._feature_matrix)[0]

        # Build results frame (exclude the seed track itself)
        candidates = self.df.copy()
        candidates["similarity"] = scores
        candidates = candidates[candidates.index != idx]

        if same_genre:
            candidates = candidates[candidates["track_genre"] == seed["track_genre"]]

        results = (
            candidates
            .nlargest(n, "similarity")
            [["track_name", "artists", "track_genre", "popularity", "similarity"]]
            .reset_index(drop=True)
        )
        results["similarity"] = results["similarity"].round(3)
        results.index = range(1, len(results) + 1)

        self._print_table(
            results,
            f"Top {n} songs similar to '{seed['track_name']}' by {seed['artists']}"
        )
        return results

    # ──────────────────────────────────────────
    # 2. Filter by genre / mood
    # ──────────────────────────────────────────

    def filter_by_genre(
        self,
        genre: str = None,
        mood: str = None,
        n: int = 20,
        sort_by: str = "popularity",
    ) -> pd.DataFrame:
        """
        Filter songs by genre and/or mood preset, sorted by a chosen column.

        Args:
            genre:    Genre string (case-insensitive partial match, e.g. 'pop', 'jazz').
            mood:     One of: happy, sad, energetic, calm, party, focus, romantic, aggressive.
            n:        Number of results (default 20).
            sort_by:  Column to sort by — 'popularity', 'danceability', 'energy', etc.

        Returns:
            DataFrame of matching tracks.

        Example:
            rec.filter_by_genre(genre="hip-hop", mood="energetic", n=15)
            rec.filter_by_genre(mood="calm", n=10, sort_by="acousticness")
        """
        results = self.df.copy()

        if genre:
            mask = results["track_genre"].str.lower().str.contains(
                genre.lower(), na=False
            )
            results = results[mask]
            if results.empty:
                available = sorted(self.df["track_genre"].unique())
                raise ValueError(
                    f"Genre '{genre}' not found.\nAvailable genres:\n{available}"
                )

        if mood:
            mood_lower = mood.lower()
            if mood_lower not in MOOD_PRESETS:
                raise ValueError(
                    f"Mood '{mood}' not recognised.\n"
                    f"Available moods: {list(MOOD_PRESETS.keys())}"
                )
            for feature, (lo, hi) in MOOD_PRESETS[mood_lower].items():
                results = results[
                    results[feature].between(lo, hi)
                ]

        if results.empty:
            print("No tracks match the given filters.")
            return pd.DataFrame()

        if sort_by not in results.columns:
            sort_by = "popularity"

        results = (
            results
            .sort_values(sort_by, ascending=False)
            .head(n)
            [["track_name", "artists", "track_genre", "popularity",
              "danceability", "energy", "valence"]]
            .reset_index(drop=True)
        )
        results.index = range(1, len(results) + 1)

        label = " | ".join(filter(None, [
            f"genre='{genre}'" if genre else None,
            f"mood='{mood}'" if mood else None,
        ])) or "all genres"
        self._print_table(results, f"Top {n} tracks — {label}")
        return results

    def list_genres(self) -> list:
        """Print and return all unique genres in the dataset."""
        genres = sorted(self.df["track_genre"].unique())
        print(f"\n{len(genres)} genres available:\n")
        for i, g in enumerate(genres, 1):
            print(f"  {i:>3}. {g}")
        return genres

    def list_moods(self) -> dict:
        """Print available mood presets and their feature ranges."""
        print("\nAvailable mood presets:")
        print(f"{'─'*40}")
        for mood, filters in MOOD_PRESETS.items():
            parts = [f"{k}: {v[0]}–{v[1]}" for k, v in filters.items()]
            print(f"  {mood:<12} {', '.join(parts)}")
        return MOOD_PRESETS

    # ──────────────────────────────────────────
    # 3. Search by artist
    # ──────────────────────────────────────────

    def search_by_artist(
        self,
        artist_name: str,
        n: int = 20,
        sort_by: str = "popularity",
    ) -> pd.DataFrame:
        """
        Find all tracks by an artist (partial name match).

        Args:
            artist_name: Artist name or partial name (case-insensitive).
            n:           Maximum number of results (default 20).
            sort_by:     Column to sort results by (default 'popularity').

        Returns:
            DataFrame of the artist's tracks.

        Example:
            rec.search_by_artist("Taylor Swift")
            rec.search_by_artist("Drake", sort_by="danceability")
        """
        mask = self.df["artists_lower"].str.contains(
            artist_name.lower().strip(), na=False
        )
        results = self.df[mask]

        if results.empty:
            raise ValueError(
                f"No tracks found for artist '{artist_name}'. "
                "Check the spelling or try a partial name."
            )

        if sort_by not in results.columns:
            sort_by = "popularity"

        results = (
            results
            .sort_values(sort_by, ascending=False)
            .head(n)
            [["track_name", "artists", "track_genre", "popularity",
              "danceability", "energy", "valence"]]
            .reset_index(drop=True)
        )
        results.index = range(1, len(results) + 1)

        total = mask.sum()
        self._print_table(
            results,
            f"Tracks by '{artist_name}' ({total} found, showing {len(results)})"
        )
        return results

    # ──────────────────────────────────────────
    # 4. Popularity charts
    # ──────────────────────────────────────────

    def popularity_charts(
        self,
        genre: str = None,
        n: int = 20,
        min_popularity: int = 0,
    ) -> pd.DataFrame:
        """
        Get the top songs ranked by popularity score.

        Args:
            genre:          Optionally restrict to a genre (partial match).
            n:              Number of top songs to return (default 20).
            min_popularity: Minimum popularity score filter (0–100).

        Returns:
            DataFrame of top tracks sorted by popularity.

        Example:
            rec.popularity_charts()                          # global top 20
            rec.popularity_charts(genre="pop", n=10)         # top pop songs
            rec.popularity_charts(min_popularity=80, n=50)   # only very popular songs
        """
        results = self.df.copy()

        if genre:
            results = results[
                results["track_genre"].str.lower().str.contains(genre.lower(), na=False)
            ]
            if results.empty:
                raise ValueError(f"Genre '{genre}' not found.")

        results = results[results["popularity"] >= min_popularity]

        results = (
            results
            .sort_values("popularity", ascending=False)
            .drop_duplicates(subset=["track_name", "artists"])
            .head(n)
            [["track_name", "artists", "track_genre", "popularity",
              "danceability", "energy", "valence"]]
            .reset_index(drop=True)
        )
        results.index = range(1, len(results) + 1)

        label = f"genre='{genre}'" if genre else "all genres"
        self._print_table(
            results,
            f"Top {n} most popular tracks — {label}"
        )
        return results

    # ──────────────────────────────────────────
    # 5. Dataset stats
    # ──────────────────────────────────────────

    def stats(self):
        """Print a summary of the loaded dataset."""
        print(f"\n{'═'*60}")
        print(f"  Dataset Summary")
        print(f"{'═'*60}")
        print(f"  Total tracks    : {len(self.df):,}")
        print(f"  Unique artists  : {self.df['artists'].nunique():,}")
        print(f"  Unique genres   : {self.df['track_genre'].nunique()}")
        print(f"  Avg popularity  : {self.df['popularity'].mean():.1f} / 100")
        print(f"  Explicit tracks : {self.df['explicit'].sum():,} "
              f"({100*self.df['explicit'].mean():.1f}%)")
        print(f"\n  Audio feature averages:")
        for feat in AUDIO_FEATURES:
            val = self.df[feat].mean()
            bar_len = int(val * 20) if val <= 1 else int((val + 60) / 8)
            bar = "█" * bar_len + "░" * (20 - min(bar_len, 20))
            print(f"    {feat:<18} {bar}  {val:.3f}")
        print(f"{'═'*60}\n")