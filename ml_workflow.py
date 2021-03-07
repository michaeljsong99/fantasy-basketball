"""
This module is designed to run the entire ML workflow for the project.
"""

from glob import glob
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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
from nnmodeldataset import NNModelDataSet
from model import FeedforwardNeuralNetModel


class MLWorkflow:
    def __init__(self):
        self.config = Config()
        self.training_data = None
        self.test_data = None
        self.validation_data = None
        self.model = None
        self.optimizer = None
        self.criterion = None

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

        self.training_data = training_df
        self.test_data = test_df
        self.validation_data = validation_df

    def select_loss_function(self, loss_func):
        if loss_func == "mae":
            return nn.L1Loss()
        elif loss_func == "smooth_l1":
            return nn.SmoothL1Loss()
        else:
            # By default, use Mean Squared Error loss function.
            return nn.MSELoss()

    def select_optimizer(self, optimizer, model_params, learning_rate, weight_decay):
        if optimizer == "adam":
            return optim.Adam(model_params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "sgd" or optimizer == "sgd_m":
            momentum = 0 if optimizer == "sgd" else 0.9
            return optim.SGD(
                model_params,
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer == "rmsprop":
            return optim.RMSprop(
                model_params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay
            )
        else:
            raise RuntimeError(
                f"The following optimizer is not recognized: {optimizer}"
            )

    def train_model(self):
        self.read_in_data()
        label = "total_fantasy_pts"
        X_train = self.training_data.drop(columns=[label]).to_numpy()
        y_train = self.training_data[label].to_numpy()
        X_test = self.test_data.drop(columns=[label]).to_numpy()
        y_test = self.test_data[label].to_numpy()
        X_val = self.validation_data.drop(columns=[label]).to_numpy()
        y_val = self.validation_data[label].to_numpy()

        """
        NOTE: we don't want to normalize the data again! 
        It has already been normalized per season in data generation.
        If we normalized the features again, we would be comparing different seasons to each other.
        """

        input_dim = len(X_test[0])  # Number of features.

        train_data = NNModelDataSet(X=X_train, y=y_train)
        test_data = NNModelDataSet(X=X_test, y=y_test)
        validation_data = NNModelDataSet(X=X_val, y=y_val)

        train_data_loader = DataLoader(
            train_data, batch_size=self.config.batch_size, shuffle=True
        )

        self.model = FeedforwardNeuralNetModel(input_dim=input_dim)
        model_parameters = self.model.parameters()

        # Set the loss function.
        self.criterion = self.select_loss_function(loss_func=self.config.loss_function)

        # Select the optimizer.
        self.optimizer = self.select_optimizer(
            optimizer=self.config.optimizer,
            model_params=model_parameters,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Train the model.
        logger.info("Beginning Model Training...")
        for epoch in range(self.config.epochs):

            epoch_loss = None

            for i, (data, labels) in enumerate(train_data_loader):

                # Clear gradients with respect to parameters.
                self.optimizer.zero_grad()

                # Forward Feed
                outputs = self.model(data)

                # Calculate Loss
                loss = self.criterion(outputs, labels)
                epoch_loss = loss

                # Get gradients with respect to parameters.
                loss.backward()

                # Update parameters.
                self.optimizer.step()

            loss_value = epoch_loss.item()
            logger.info(
                f"Training Loss on Epoch {epoch+1}/{self.config.epochs}: {loss_value}"
            )
