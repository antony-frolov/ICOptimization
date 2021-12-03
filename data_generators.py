import numpy as np


class DataGenerator:
    def __init__(self, H, M, res_dist='normal', scale=0.01, dim=None, H_dist='uniform'):
        self.H = H
        if isinstance(H, str):
            if isinstance(dim, int) and dim > 0:
                self.dim = dim
            else:
                raise ValueError()
        else:    
            self.dim = H.shape[0]
        
        if res_dist == 'normal':
            self.res_gen = lambda: np.random.normal(scale=scale, size=self.dim)
        else:
            raise ValueError(f"Residue distribution '{res_dist}' not implemented")
        
        self.x_range = np.arange(-2*M - 1, (2*M + 1) + 1, 2)
        self.x_gen = lambda: np.random.choice(self.x_range, size=self.dim)
        
        self.H_dist = None
        self.H_gen = None
        
        if isinstance(H, str):
            if H == 'random':
                if H_dist == 'uniform':
                    self.H_dist = H_dist
                    self.H_gen = lambda: np.random.random((self.dim, self.dim))
                else:
                    raise ValueError(f"H distribution '{H_dist}' not implemented")
            elif H == 'fixed':
                self.H = np.random.random(size=(dim, dim)) * 2 - 1
                

    def generate(self, size=10):
        target = np.row_stack([self.x_gen() for _ in range(size)])
        if isinstance(self.H, str) and self.H == 'random':
            data = {'H': np.array([self.H_gen() for _ in range(len(target))])}
            data['y'] = np.row_stack([H @ x - self.res_gen() for H, x in zip(data['H'], target)])
        else:
            data = {
                'H': self.H,
                'y': np.row_stack([self.H @ x - self.res_gen() for x in target]),
            }

        return data, target


class ADNNDataTransformer:

    def transform(self, data, target):
        datasets, targets = [], []
        for i in range(target.shape[1]):
            h_i = data['H'][..., i]
            y = data['y']
            data_p = y - (target[:, i] - 1)[:, np.newaxis] * h_i
            data_n = y - (target[:, i] + 1)[:, np.newaxis] * h_i
            dataset = np.hstack((data_p, data_n)).reshape(-1, y.shape[1])
            datasets.append(dataset)
            bin_target = np.zeros((len(target), 2))
            bin_target[:, 0] = 1
            bin_target = bin_target.ravel()
            targets.append(bin_target)
        return datasets, targets
