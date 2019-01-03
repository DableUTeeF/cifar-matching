from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
import numpy as np


# todo: try train in first 5 cls and validate in last 5 cls
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
        def __init__(self, dset, set_length, use_all=False):
            self.dset = dset
            self.set_length = set_length
            self.use_all = use_all

        def __len__(self):
            return self.set_length // 2

        def __getitem__(self, idx):
            if not self.use_all:
                """
                Use all classes or not
                """
                c = 5
            else:
                c = 10
            x_support = np.zeros((c, 32, 32, 3), dtype='uint8')
            y_support = np.zeros((c, c), dtype='uint8')
            selected_idxs = []
            for i in range(5):
                if self.set_length < 20000:
                    """
                    Separate between train and test set.
                    Train set will return only first 5 classes, else for test set.
                    """
                    j = i + 5
                else:
                    j = i
                randomed_idx = np.random.randint(0, self.set_length-1)
                while randomed_idx in selected_idxs or self.dset[randomed_idx][1] != j:
                    randomed_idx = np.random.randint(0, self.set_length-1)
                selected_idxs.append(randomed_idx)
                x, y = self.dset[randomed_idx]
                if not self.use_all and y > 4:
                    y -= 5
                x_support[i] = np.array(x, dtype='uint8')
                y_support[i, y] = 1

            if self.use_all:
                r = range(10)
                s = 0  # use for subtract y target
            else:
                if self.set_length < 20000:
                    r = range(5, 10)
                    s = 5
                else:
                    r = range(5)
                    s = 0
            while idx in selected_idxs or self.dset[idx][1] not in r:
                idx = np.random.randint(0, self.set_length-1)
            x, y = self.dset[idx]
            x_target = np.array(x, dtype='uint8')
            y_target = np.array(y, dtype='uint8')
            return x_support, y_support, x_target, y_target - s
