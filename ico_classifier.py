import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tf
from data_generators import ADNNDataTransformer
# from torch.utils.data import dataset
# import torchvision
# import torchvision.transforms as transforms


class ADNN(nn.Module):
    def __init__(self, input_size, hidden_size, criterion,
                 optimizer, num_classes=2):
        super(ADNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)

        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

    def fit(self, dataset, target, num_epochs, lr):
        dataset = tf.normalize(torch.tensor(dataset, dtype=torch.float32))
        # dataset = torch.tensor(dataset, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        criterion = self.criterion()
        optimizer = self.optimizer(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            outputs = self(dataset)

            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f'Epoch [{epoch + 1}/{num_epochs}], '
            #       f'Loss: {loss.item():.4f}')


class ElementClassifier:
    def __init__(self, model, h, M):
        self.ADNN = model
        self.h = h
        self.M = M

    def fit(self, *args, **kwargs):
        self.ADNN.fit(*args, **kwargs)

    def predict(self, data):
        with torch.no_grad():
            log_const_proba = 0.
            log_probabilities = np.full(shape=(data.shape[0], 2 * self.M + 2),
                                        fill_value=log_const_proba)
            for i, m in enumerate(range(self.M, -self.M - 1, -1)):
                outputs = self.ADNN(tf.normalize(torch.tensor(data + 2 * m * self.h, dtype=torch.float32)))
                # outputs = self.ADNN(torch.tensor(data + 2 * m * self.h, dtype=torch.float32))
                outputs = tf.log_softmax(outputs, dim=1)
                outputs = outputs[:, 1] - outputs[:, 0]
                outputs = outputs.numpy()
                log_probabilities[:, i+1:] += outputs[:, np.newaxis]
            return np.arange(-2 * self.M - 1, 2 * self.M + 2, 2)[log_probabilities.argmax(axis=1)]


class ICOClassifier:
    def __init__(self, H, M, hidden_size=500, criterion='cross-entropy', optimizer='sgd'):
        self.H = H
        self.dim = H.shape[0]
        self.M = M

        if criterion == 'cross-entropy':
            self.criterion = nn.CrossEntropyLoss
        else:
            raise ValueError(f"Criterion '{criterion}' not implemented")

        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam
        else:
            raise ValueError(f"Optimizer '{optimizer}' not implemented")

        self.ElementClassifiers = [ElementClassifier(ADNN(input_size=self.dim,
                                                          hidden_size=hidden_size,
                                                          criterion=self.criterion,
                                                          optimizer=self.optimizer),
                                                     H[:, i], M) for i in range(self.dim)]
        self.data_transformer = ADNNDataTransformer(H).transform

    def fit(self, data, target, num_epochs=10, lr=0.001):
        datasets, targets = self.data_transformer(data, target)
        for i, (classifier, dataset, target) in enumerate(zip(self.ElementClassifiers, datasets, targets)):
            classifier.fit(dataset, target, num_epochs=num_epochs, lr=lr)
            print(f'ADNN {1+i}/{self.dim} trained')

    def predict(self, data):
        return np.column_stack([model.predict(data) for model in self.ElementClassifiers])
