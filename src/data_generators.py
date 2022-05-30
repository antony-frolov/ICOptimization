import torch
import torch.nn.functional as F


class DataGenerator():
    def __init__(self, mode, M, dim=None, H_dist=None, res_dist=None):
        self.mode = mode
        self.H = None
        if self.mode == 'fixed':
            self.H = torch.empty(*dim).uniform_(H_dist['low'], H_dist['high'])

        self.M = M
        self.dim = dim

        self.res_dist = {'dist': 'normal', 'scale': 0.01} if res_dist is None else res_dist

        self.H_dist = {'dist': 'uniform', 'low': -self.M, 'high': self.M} if H_dist is None else H_dist

    def generate(self, size=100):
        xs = torch.randint(0, 2*(self.M+1), size=(size, self.dim[1]), dtype=torch.float32) * 2 - (2*self.M + 1)
        if self.mode == 'random' and self.H_dist['dist'] == 'uniform':
            Hs = torch.empty(size, *self.dim).uniform_(self.H_dist['low'], self.H_dist['high'])
        elif self.mode == 'fixed':
            Hs = self.H.unsqueeze(0)

        if self.res_dist['dist'] == 'uniform':
            res = torch.empty(size, self.dim[0]).uniform_(self.res_dist['low'], self.res_dist['high'])
        elif self.res_dist['dist'] == 'normal':
            res = torch.empty(size, self.dim[0]).normal_(0, self.res_dist['scale'])
        else:
            raise NotImplementedError

        ys = (Hs @ xs.unsqueeze(2)).squeeze(2) - res
        return {'Hs': Hs, 'xs': xs, 'ys': ys}


class ADNNDataTransformer:
    def train_transform(self, Hs, xs, ys, batch_size=32):
        size = ys.shape[0]
        dim = Hs.shape[1:]
        dataloaders = []
        for i in range(dim[1]):
            h_i = Hs[..., i]
            pos_shifted = ys - xs[:, i][:, None] * h_i + h_i
            neg_shifted = ys - xs[:, i][:, None] * h_i - h_i
            input_i = torch.hstack([pos_shifted, neg_shifted]).reshape(-1, dim[0])
            input_i = F.normalize(input_i)

            output_i = torch.zeros((size, 2), dtype=int)
            output_i[:, 0] = 1
            output_i = output_i.ravel()

            dataset_i = torch.utils.data.TensorDataset(input_i, output_i)
            dataloader_i = torch.utils.data.DataLoader(dataset_i, batch_size=batch_size, shuffle=True)
            dataloaders.append(dataloader_i)

        return dataloaders

    def test_transform(self, Hs, ys, M, **kwargs):
        dim = Hs.shape[1:]
        inputs = []

        for i in range(dim[1]):
            inputs.append([])
            h_i = Hs[..., i]
            for m in range(M, -M-1, -1):
                input_i = ys + 2 * m * h_i
                inputs[i].append(F.normalize(input_i))

        return inputs
