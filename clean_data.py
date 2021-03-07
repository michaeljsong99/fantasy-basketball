"""
The purpose of this module is to clean the data so we can use it for training.
"""


import pandas as pd

# Logging
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s: \n %(message)s \n"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

from config import Config


class DataCleaner:
    def __init__(self):
        self.season_stats = pd.read_csv("raw_data/season_stats.csv")
        config = Config()
        self.earliest_season = config.start_year
        self.players = set()
        self.stats_by_season = {}
        self.training_data = {}
        self.normalized_training_data = {}

    def _clean_season_stats(self):
        """
        Clean the player data file. We want to select only the seasons we care about, and we also want
        per game stats. We also want to sort by minutes played.
        :return: None
        """
        logger.info("Reading season stats data...")
        # Select the stats we want.
        stats = [
            "Year",
            "Player",
            "Age",
            "G",
            "GS",
            "MP",
            "FG",
            "FGA",
            "3P",
            "FT",
            "FTA",
            "TRB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PTS",
        ]
        self.season_stats = self.season_stats[stats]
        self.season_stats = self.season_stats.loc[
            self.season_stats["Year"] >= self.earliest_season
        ]
        # Sort dataframe by minutes played. This is in case of duplicate names, we take the more "popular" player.
        self.season_stats = self.season_stats.sort_values(["MP"], ascending=[False])
        # Drop duplicate players.
        self.season_stats = self.season_stats.drop_duplicates(
            ["Player", "Year"], ignore_index=True
        )

        for col in self.season_stats.columns:
            if col != "Player":
                self.season_stats[col] = pd.to_numeric(
                    self.season_stats[col], downcast="integer"
                )

        # Now, we create the 'per-game' stats. This is useful, because total stats will be influenced by factors
        # like injuries, rest, or shortened, seasons.
        per_game_stats = [
            "MP",
            "FG",
            "FGA",
            "3P",
            "FT",
            "FTA",
            "TRB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PTS",
        ]
        for col in per_game_stats:
            self.season_stats[f"{col}/game"] = (
                self.season_stats[col] / self.season_stats["G"]
            )

        logger.info(
            f"Found {len(self.season_stats)} player season records from {self.earliest_season}-{max(self.season_stats['Year'])}."
        )

    def _select_training_data(self):
        """
        The idea is for each year, we select the top 300 players by minutes played.
        This is to avoid the problem of injuries messing up model training.
        We will not include rookies in each year, since we do not have historical data for them.
        :return: None
        """

        self.stats_by_season = {}

        latest_season = max(self.season_stats["Year"])
        for year in range(self.earliest_season, latest_season + 1):
            stats_in_year = self.season_stats.loc[self.season_stats["Year"] == year]
            # Only take top 300 players by minutes played each season.
            stats_in_year = stats_in_year.head(300)
            stats_in_year.set_index("Player", inplace=True)
            stats_in_year.drop("Year", axis=1, inplace=True)
            player_stats_for_year = stats_in_year.to_dict(orient="index")
            self.stats_by_season[year] = player_stats_for_year

            # Historically, there have been a few seasons that have had less games played.
            # Additionally, certain metrics will change over time (i.e. points, 3-pointers made).
            # We want to normalize the stats for each season as well for training.
            normalized_stats = (stats_in_year - stats_in_year.min()) / (
                stats_in_year.max() - stats_in_year.min()
            )
            # For normalized stats, they will only be used as features, not evaluation.
            # So we only need the per-game stats.
            total_stat_cols = [
                "FG",
                "FGA",
                "MP",
                "3P",
                "FT",
                "FTA",
                "TRB",
                "AST",
                "STL",
                "BLK",
                "TOV",
                "PTS",
            ]
            normalized_stats.drop(total_stat_cols, axis=1, inplace=True)
            normalized_player_stats_for_year = normalized_stats.to_dict(orient="index")
            self.normalized_training_data[year] = normalized_player_stats_for_year

        # Now remove rookies and players that did not play in the previous year from each year.
        final_stats_by_season = {
            self.earliest_season: self.stats_by_season[self.earliest_season]
        }

        for year in range(self.earliest_season + 1, latest_season + 1):
            players = list(self.stats_by_season[year].keys())
            final_players_for_season = {}
            for player in players:
                if player in self.stats_by_season[year - 1]:
                    final_players_for_season[player] = self.stats_by_season[year][
                        player
                    ]

            final_stats_by_season[year] = final_players_for_season
            logger.info(
                f"{len(final_players_for_season.keys())} players selected for year {year}."
            )
        self.training_data = final_stats_by_season

    def _validate_data(self):
        # Check to make sure that for each year, if there is a player, we have their previous season data.
        for year in range(self.earliest_season + 1, max(self.training_data.keys()) + 1):
            players = list(self.training_data[year].keys())
            for player in players:
                if player not in self.normalized_training_data[year - 1]:
                    raise RuntimeError(
                        f"Player in year {year} but not found for year {year-1}: {player}"
                    )
        logger.info("Player data validated.")

    def data_cleaning_pipeline(self):
        """
        Run the data cleaning pipeline to get a training dataset.
        :return: A dictionary with years as keys, and the values are dictionaries with players as keys
                    and their stats as values.
        """
        self._clean_season_stats()
        self._select_training_data()
        self._validate_data()
        return self.training_data, self.normalized_training_data


cleaner = DataCleaner()
data = cleaner.data_cleaning_pipeline()
