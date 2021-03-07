# Local module imports
from clean_data import DataCleaner
from training_data_generator import TrainingDataGenerator


def generate_training_data():
    data_cleaner = DataCleaner()
    training_data, normalized_training_data = data_cleaner.data_cleaning_pipeline()
    training_data_generator = TrainingDataGenerator(season_stats=training_data,
                                                    normalized_season_stats=normalized_training_data)


if __name__ == "__main__":
    generate_training_data()
    # players_2017 = training_data[2017]

    # roto = RotoCalculator(season_data=players_2017)
    # results = roto.run_simulation()
    # print(results)
