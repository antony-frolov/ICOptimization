import torch
import torch.nn as nn
import torch.nn.functional as F
from data_generators import ADNNDataTransformer
from metrics import mse
from timeit import default_timer as timer


class ADNN(nn.Module):
    def __init__(self, input_size, hidden_size, M, optimizer, optimizer_params):
        super(ADNN, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()

        self.M = M

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer(self.parameters(), **optimizer_params)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

    def train_epoch(self, dataloader):

        total_loss = 0.
        for batch in dataloader:
            data, target = batch
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.forward(data)

            loss = self.criterion(output, target)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss / len(dataloader)

    def predict(self, inputs):
        with torch.no_grad():
            log_ratios = torch.zeros(len(inputs[0]), 2*self.M+2)
            for i, data in enumerate(inputs):
                data = data.to(self.device)
                outputs = self.forward(data)
                outputs = F.log_softmax(outputs, dim=1)
                log_ratio = outputs[:, 1] - outputs[:, 0]
                log_ratios[:, i+1] = log_ratio

            log_probabilities = torch.cumsum(log_ratios, dim=1)

            return torch.arange(-2*self.M-1, 2*self.M+2, 2, dtype=torch.float32)[torch.argmax(log_probabilities, dim=1)]


class ICOClassifier:
    def __init__(self, dim, M, hidden_size=500, optimizer='adam', optimizer_params=None, logging_level='verbose'):
        self.dim = dim
        self.M = M
        self.logging_level = logging_level

        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam
        else:
            raise NotImplementedError

        if optimizer_params is None:
            optimizer_params = {'lr': 0.01}
        self.optimizer_params = optimizer_params

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ADNNs = []
        for _ in range(self.dim[1]):
            adnn = ADNN(input_size=self.dim[0], hidden_size=hidden_size, M=self.M,
                        optimizer=self.optimizer, optimizer_params=self.optimizer_params)
            self.ADNNs.append(adnn.to(self.device))

        self.data_transformer = ADNNDataTransformer()

    def train(self, train_data, num_epochs=10, batch_size=32, test_data=None, logging_freq=10):
        history = {'epoch': [], 'train_loss': [], 'train_mse': [], 'time': []}
        if test_data is not None:
            history['test_mse'] = []

        dataloaders = self.data_transformer.train_transform(**train_data, batch_size=batch_size)
        start_time = timer()
        for epoch in range(num_epochs):
            total_loss = 0.
            for classifier, dataloader in zip(self.ADNNs, dataloaders):
                loss = classifier.train_epoch(dataloader)
                total_loss += loss
            total_loss /= self.dim[1]

            if self.logging_level == 'verbose':
                print(f"\rEpoch [{epoch + 1}/{num_epochs}]" +
                      f" Train CE Loss: {total_loss:.4f}", end='')

            if (epoch+1) % logging_freq == 0:
                x_pred_train = self.predict(train_data).to('cpu')
                y_pred_train = (train_data['Hs'] @ x_pred_train.unsqueeze(2)).squeeze(2)
                train_mse = mse(train_data['ys'], y_pred_train).mean()

                if test_data is not None:
                    x_pred_test = self.predict(test_data).to('cpu')
                    y_pred_test = (test_data['Hs'] @ x_pred_test.unsqueeze(2)).squeeze(2)
                    test_mse = mse(test_data['ys'], y_pred_test).mean()

                if self.logging_level == 'verbose':
                    print(f"\rEpoch [{epoch + 1}/{num_epochs}]" +
                          f" Train CE Loss: {total_loss:.4f}" +
                          f" Train MSE Loss: {train_mse:.4f}" +
                          f" Test MSE Loss: {test_mse:.4f}" if test_data is not None else '')

                history['epoch'].append(epoch+1)
                history['time'].append(timer() - start_time)
                history['train_loss'].append(total_loss)
                history['train_mse'].append(train_mse)
                if test_data is not None:
                    history['test_mse'].append(test_mse)

        return history

    def predict(self, data):
        inputs = self.data_transformer.test_transform(M=self.M, **data)
        predictions = []
        for input_i, model in zip(inputs, self.ADNNs):
            predictions.append(model.predict(input_i))
        return torch.stack(predictions, axis=1)
