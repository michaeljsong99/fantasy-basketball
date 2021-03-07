"""
This module is designed to generate training data for the Neural Network.
The basic process is as follows:
Select certain seasons to use for training, some for test, and some for validation.
For each season, we will run 50,000 simulations, with 8 teams and 13 players per team.
Then for each team, we will take the players' previous year's stats as features.
We accumulate the team's stats to get a training vector of length 15, which has format:
[Age G GS MP/game FG/game FGA/game 3P/game FT/game FTA/game TRB/game AST/game STL/game BLK/game TOV/game PTS/game]
The final label will be the total amount of fantasy points that the team scored.
"""
import os
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

from roto_calculator import RotoCalculator
from config import Config
from utils import calculate_team_prev_season_data


class TrainingDataGenerator:
    def __init__(self, season_stats, normalized_season_stats):
        self.season_stats = season_stats
        self.normalized_season_stats = normalized_season_stats
        self.config = Config()

    def generate_training_data_for_season(self, season):
        season_data = self.season_stats[season]
        roto_calculator = RotoCalculator(season_data=season_data)
        prev_season_stats = self.normalized_season_stats[season - 1]

        training_dir = (
            f"training_data/{self.config.num_teams}teams_{self.config.team_size}players"
        )
        if not os.path.exists(training_dir):
            os.mkdir(training_dir)

        training_data = []

        for i in range(
            self.config.simulations_per_season
        ):  # By default, 10k simulations per season.
            if (i + 1) % 500 == 0:
                logger.info(
                    f"Simulation {i+1}/{self.config.simulations_per_season} in year {season}."
                )
            sim_results = roto_calculator.run_simulation()
            for team in sim_results:
                # The label.
                total_fantasy_pts = team["total_fantasy_pts"]

                # Extract the feature vector per team.
                players = team["team"]

                features = calculate_team_prev_season_data(
                    team=players, prev_season_data=prev_season_stats
                )
                features.append(total_fantasy_pts)
                training_data.append(features)

            # Create CSV files in batches of 10,000 simulations (10k x num_teams rows for each file).
        df = pd.DataFrame.from_records(
            training_data, columns=self.config.vector_columns
        )
        file_name = f"{season}_{self.config.num_teams}T_{self.config.team_size}P.csv"
        df.to_csv(os.path.join(training_dir, file_name))
        logger.info(f"Saved CSV file {file_name}.")

    def generate_training_data(self):
        for year in self.config.training_years:
            logger.info(f"Generating training data for year {year} ...")
            self.generate_training_data_for_season(season=year)
