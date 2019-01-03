from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
import numpy as np


class MatchingCifarLoader:
    """
    DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    """
    def __init__(self, root):
        self.trainset = self.DataSet(CIFAR10(root=root, train=True, download=True), 50000)
        self.testset = self.DataSet(CIFAR10(root=root, train=False, download=True), 10000)

    def get_trainset(self, batch_size, num_worker, shuffle=True):
        return DataLoader(self.trainset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_worker)

    def get_testset(self, batch_size, num_worker):
        return DataLoader(self.testset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_worker)

    class DataSet:
        def __init__(self, dset, set_length):
            self.trainset = dset
            self.set_length = set_length

        def __len__(self):
            return self.set_length

        def __getitem__(self, idx):
            x_support = np.zeros((10, 32, 32, 3), dtype='uint8')
            y_support = np.zeros((10, 10), dtype='uint8')
            selected_idxs = []
            for i in range(10):
                randomed_idx = np.random.randint(0, self.set_length-1)
                while randomed_idx in selected_idxs or self.trainset[randomed_idx][1] != i:
                    randomed_idx = np.random.randint(0, self.set_length-1)
                selected_idxs.append(randomed_idx)
                x, y = self.trainset[randomed_idx]
                x_support[i] = np.array(x, dtype='uint8')
                y_support[i, y] = 1
            while idx in selected_idxs:
                idx = np.random.randint(0, self.set_length-1)
            x, y = self.trainset[idx]
            x_target = np.array(x, dtype='uint8')
            y_target = np.array(y, dtype='uint8')
            return x_support, y_support, x_target, y_target
