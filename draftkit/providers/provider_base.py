from abc import ABC, abstractmethod
from datetime import datetime, timezone

import pandas as pd


CANONICAL_PROVIDER_COLUMNS = [
    "player_name",
    "sportsbook_projection",
    "sportsbook_projection_rank",
    "passing_yards_projection",
    "passing_tds_projection",
    "rushing_yards_projection",
    "rushing_tds_projection",
    "receptions_projection",
    "receiving_yards_projection",
    "receiving_tds_projection",
    "provider_name",
    "last_updated",
]


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_provider_df(provider_name=None):
    df = pd.DataFrame(columns=CANONICAL_PROVIDER_COLUMNS)
    df.attrs["provider_name"] = provider_name
    df.attrs["last_updated"] = utc_timestamp()
    return df


def normalize_provider_output(df, provider_name):
    if df is None or df.empty:
        return empty_provider_df(provider_name)

    out = df.copy()
    for column in CANONICAL_PROVIDER_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA

    out["provider_name"] = out["provider_name"].fillna(provider_name)
    if out["last_updated"].isna().all():
        out["last_updated"] = utc_timestamp()
    else:
        out["last_updated"] = out["last_updated"].fillna(utc_timestamp())

    numeric_columns = [
        "sportsbook_projection",
        "sportsbook_projection_rank",
        "passing_yards_projection",
        "passing_tds_projection",
        "rushing_yards_projection",
        "rushing_tds_projection",
        "receptions_projection",
        "receiving_yards_projection",
        "receiving_tds_projection",
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if out["sportsbook_projection_rank"].isna().all() and out["sportsbook_projection"].notna().any():
        out["sportsbook_projection_rank"] = (
            out["sportsbook_projection"]
            .rank(method="first", ascending=False)
            .astype("Int64")
        )

    out = out[CANONICAL_PROVIDER_COLUMNS].copy()
    out.attrs["provider_name"] = provider_name
    out.attrs["last_updated"] = (
        str(out["last_updated"].dropna().max())
        if out["last_updated"].notna().any()
        else utc_timestamp()
    )
    return out


class SportsbookProvider(ABC):
    @abstractmethod
    def get_provider_name(self):
        raise NotImplementedError

    @abstractmethod
    def health_check(self):
        raise NotImplementedError

    @abstractmethod
    def fetch_player_props(self):
        raise NotImplementedError

    @abstractmethod
    def fetch_projection_data(self):
        raise NotImplementedError

    @abstractmethod
    def normalize_data(self, raw_data):
        raise NotImplementedError

    @abstractmethod
    def get_debug_info(self):
        raise NotImplementedError


class MockSportsbookProvider(SportsbookProvider):
    def __init__(self, projections_df=None):
        self._last_updated = utc_timestamp()
        self._projections_df = projections_df

    def get_provider_name(self):
        return "mock"

    def health_check(self):
        projections = self.fetch_projection_data()
        return {
            "provider_name": self.get_provider_name(),
            "status": "healthy",
            "row_count": int(len(projections)),
            "last_updated": self._last_updated,
        }

    def fetch_player_props(self):
        return pd.DataFrame([
            {
                "player_name": "Josh Allen",
                "market_type": "passing_yards",
                "line_value": 255.5,
                "sportsbook": self.get_provider_name(),
            },
            {
                "player_name": "Christian McCaffrey",
                "market_type": "rushing_yards",
                "line_value": 82.5,
                "sportsbook": self.get_provider_name(),
            },
        ])

    def fetch_projection_data(self):
        if self._projections_df is not None:
            return self.normalize_data(self._projections_df)

        raw = pd.DataFrame([
            {
                "player_name": "Josh Allen",
                "sportsbook_projection": 24.8,
                "passing_yards_projection": 255.5,
                "passing_tds_projection": 1.9,
                "rushing_yards_projection": 39.5,
                "rushing_tds_projection": 0.35,
            },
            {
                "player_name": "Christian McCaffrey",
                "sportsbook_projection": 22.6,
                "rushing_yards_projection": 82.5,
                "rushing_tds_projection": 0.65,
                "receptions_projection": 4.6,
                "receiving_yards_projection": 38.5,
                "receiving_tds_projection": 0.25,
            },
            {
                "player_name": "Justin Jefferson",
                "sportsbook_projection": 20.9,
                "receptions_projection": 6.4,
                "receiving_yards_projection": 88.5,
                "receiving_tds_projection": 0.45,
            },
        ])
        return self.normalize_data(raw)

    def normalize_data(self, raw_data):
        return normalize_provider_output(raw_data, self.get_provider_name())

    def get_debug_info(self):
        projections = self.fetch_projection_data()
        return {
            "provider_name": self.get_provider_name(),
            "status": "healthy",
            "row_count": int(len(projections)),
            "last_updated": projections.attrs.get("last_updated", self._last_updated),
            "columns": list(projections.columns),
            "sample_rows": projections.head(5).to_dict("records"),
        }
