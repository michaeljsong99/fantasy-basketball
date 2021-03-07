"""
This module is designed to run the entire ML workflow for the project.
"""

from glob import glob
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

from config import Config


class MLWorkflow:
    def __init__(self):
        self.config = Config()
        self.training_data = None
        self.test_data = None
        self.validation_data = None

    def read_in_data(self):
        training_dir = "training_data"
        test_dir = "test_data"
        validation_data = "validation_data"

        training_files = [
            y for x in os.walk(training_dir) for y in glob(os.path.join(x[0], "*.csv"))
        ]
        test_files = [
            y for x in os.walk(test_dir) for y in glob(os.path.join(x[0], "*.csv"))
        ]
        validation_files = [
            y
            for x in os.walk(validation_data)
            for y in glob(os.path.join(x[0], "*.csv"))
        ]

        training_dfs = [pd.read_csv(file, index_col=0) for file in training_files]
        test_dfs = [pd.read_csv(file, index_col=0) for file in test_files]
        validation_dfs = [pd.read_csv(file, index_col=0) for file in validation_files]

        logger.info("Reading training data...")
        training_df = pd.concat(training_dfs)
        logger.info("Reading test data...")
        test_df = pd.concat(test_dfs)
        logger.info("Reading validation data...")
        validation_df = pd.concat(validation_dfs)

        # Clear memory.
        training_dfs = None
        test_dfs = None
        validation_dfs = None

        if self.config.combine_data:
            combined_df = pd.concat([training_df, test_df, validation_df])
            # Shuffle the dataframe.
            length = len(combined_df)
            combined_df = combined_df.sample(frac=1).reset_index(drop=True)
            # Set 70% for training, 20% for test, 10% for validation.
            first_split = round(0.7 * length)
            second_split = round(0.9 * length)
            training_df = combined_df.iloc[:first_split]
            test_df = combined_df.iloc[first_split:second_split]
            validation_df = combined_df.iloc[second_split:]
