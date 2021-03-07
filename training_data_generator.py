'''
This module is designed to generate training data for the Neural Network.
The basic process is as follows:
Select certain seasons to use for training, some for test, and some for validation.
For each season, we will run 50,000 simulations, with 8 teams and 13 players per team.
Then for each team, we will take the players' previous year's stats as features.
We accumulate the team's stats to get a training vector of length 15, which has format:
[Age G GS MP/game FG/game FGA/game 3P/game FT/game FTA/game TRB/game AST/game STL/game BLK/game TOV/game PTS/game]
The final label will be the total amount of fantasy points that the team scored.
'''

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

class TrainingDataGenerator:
    def __init__(self, season_stats, normalized_season_stats):
        self.season_stats = season_stats
        self.normalized_season_stats = normalized_season_stats
        self.config = Config()

    def generate_training_data_for_season(self, season):
        season_stats = self.season_stats[season]
        prev_season_stats = self.normalized_season_stats[season-1]
        for i in range(10000):
            pass

    def generate_training_data(self):
        for year in self.config.training_years:
            logger.info(f"Generating training data for year {year} ...")
            self.generate_training_data_for_season(season=year)
