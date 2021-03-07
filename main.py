# Local module imports
from clean_data import DataCleaner
from ml_workflow import MLWorkflow
from training_data_generator import TrainingDataGenerator


def generate_training_data():
    """
    Call this method in __main__ to generate historical training data.
    :return: None
    """
    data_cleaner = DataCleaner()
    training_data, normalized_training_data = data_cleaner.data_cleaning_pipeline()
    training_data_generator = TrainingDataGenerator(
        season_stats=training_data, normalized_season_stats=normalized_training_data
    )
    training_data_generator.generate_training_data()


def train_model():
    """
    Call this method in __main__ to perform model training.
    :return:
    """
    ml_workflow = MLWorkflow()
    ml_workflow.train_model()
    pass


if __name__ == "__main__":
    pass
    train_model()
