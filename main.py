# Library imports
import random

# Local module imports
from clean_data import DataCleaner
from roto_calculator import RotoCalculator

if __name__ == "__main__":
    data_cleaner = DataCleaner()
    training_data, normalized_training_data = data_cleaner.data_cleaning_pipeline()
    players_2017 = training_data[2017]

    roto = RotoCalculator(season_data=players_2017)
    results = roto.run_simulation()
    print(results)
