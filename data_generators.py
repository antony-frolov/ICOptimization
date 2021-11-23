import numpy as np


class DataGenerator:
    def __init__(self, H, M, res_dist='normal', scale=0.01):
        self.H = H
        self.dim = H.shape[0]
        if res_dist == 'normal':
            self.res_gen = lambda: np.random.normal(scale=scale, size=self.dim)
        else:
            raise ValueError(f"Distribution '{res_dist}' not implemented")
        self.x_range = np.arange(-2*M - 1, (2*M + 1) + 1, 2)
        self.x_gen = lambda: np.random.choice(self.x_range, size=self.dim)

    def generate(self, size=10):
        target = np.row_stack([self.x_gen() for _ in range(size)])
        data = np.row_stack([self.H @ x - self.res_gen() for x in target])
        return data, target


class ADNNDataTransformer:
    def __init__(self, H):
        self.H = H

    def transform(self, data, target):
        datasets, targets = [], []
        for i in range(data.shape[1]):
            data_p = data - (target[:, i] - 1)[:, np.newaxis] * self.H[:, i]
            data_n = data - (target[:, i] + 1)[:, np.newaxis] * self.H[:, i]
            dataset = np.hstack((data_p, data_n)).reshape(-1, data.shape[1])
            datasets.append(dataset)
            bin_target = np.zeros((len(target), 2))
            bin_target[:, 0] = 1
            bin_target = bin_target.ravel()
            targets.append(bin_target)
        return datasets, targets


if __name__ == '__main__':
    H = np.array([[1, 0],
                  [0, 1]])
    train_generator = DataGenerator(H, M=1)
    data, target = train_generator.generate(2)
    print(data, target, sep='\n')

    adnn_transformer = ADNNDataTransformer(H)
    datasets, targets = adnn_transformer.transform(data, target)
    print(datasets, targets, sep='\n')
