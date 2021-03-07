"""
A class to encapsulate a dataset.
"""
import torch


class NNModelDataSet:
    def __init__(self, X, y):
        if y is None:
            # This occurs during prediction.
            self.data = torch.tensor(X).float()

        else:
            self.data = X.tolist()
            self.labels = y.tolist()

            self.data = torch.tensor(self.data).float()
            self.labels = torch.tensor(self.labels).float().view(-1, 1)

        self.input_size = self.data.shape[-1]

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.labels[index]
        return data, labels

    def __len__(self):
        return len(self.labels)
