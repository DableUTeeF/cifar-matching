import matching_networks
from torch.nn import functional as F
import torch
import tqdm
from math import ceil
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


class Builder:
    def __init__(self, data, optim, lr):
        """
        Initializes the experiment
        :param data:
        """
        self.data = data
        self.classes_per_set = 10
        self.lr = lr
        self.image_size = (32, 32)
        self.optim = optim
        self.wd = 1e-6
        self.g = matching_networks.Classifier(num_channels=3)
        self.dn = matching_networks.DistanceNetwork()
        self.classify = matching_networks.AttentionalClassify()
        # self.g_lstm = matching_networks.BidirectionalLSTM(1, batch_size, 256, True)
        # self.matchNet = MatchingNetwork(keep_prob, batch_size, num_channels, self.lr, fce, classes_per_set,
        #                                 samples_per_class, image_size, self.isCuadAvailable & self.use_cuda)
        self.total_iter = 0
        self.g.cuda()
        self.total_train_iter = 0
        self.optimizer = self._create_optimizer(self.g, self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', verbose=True)

    def run_training_epoch(self, batch_size, num_worker):
        total_c_loss = 0.0
        total_accuracy = 0.0
        # optimizer = self._create_optimizer(self.matchNet, self.lr)
        traindata = self.data.get_trainset(batch_size, num_worker, shuffle=True)
        total_train_batches = len(traindata)
        with tqdm.tqdm(total=total_train_batches) as pbar:
            for i, (x_support_set, y_support_set, x_target, y_target) in enumerate(traindata):
                x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
                x_target = x_target.permute(0, 3, 1, 2)
                y_support_set = y_support_set.float()
                y_target = y_target.long()
                x_support_set = x_support_set.float()
                x_target = x_target.float()
                acc, c_loss = self.matchNet(x_support_set, y_support_set, x_target, y_target)

                # optimize process
                self.optimizer.zero_grad()
                c_loss.backward()
                self.optimizer.step()

                total_c_loss += c_loss.data[0]
                total_accuracy += acc.data[0]
                iter_out = f"loss: {total_c_loss / i:.{3}}, acc: {total_accuracy / i:.{3}}"
                pbar.set_description(iter_out)
                pbar.update(1)
                # self.total_train_iter+=1

            self.scheduler.step(total_accuracy)
            total_c_loss = total_c_loss / total_train_batches
            total_accuracy = total_accuracy / total_train_batches
            return total_c_loss, total_accuracy

    def matchNet(self, x_support_set, y_support_set_one_hot, x_target, y_target):
        # previously in matchnet.forward()
        encoded_images = []
        with torch.no_grad():
            for j in np.arange(x_support_set.size(1)):
                try:
                    gen_encode = self.g(x_support_set[:, j, :, :].cuda()).cpu()
                except RuntimeError as e:
                    raise RuntimeError(f'j={j}: {e}')
                encoded_images.append(gen_encode.cpu())
        gen_encode = self.g(x_target.cuda())
        encoded_images.append(gen_encode.cpu())
        output = torch.stack(encoded_images)
        similarites = self.dn(support_set=output[:-1], input_image=output[-1])
        preds = self.classify(similarites, support_set_y=y_support_set_one_hot)
        values, indices = preds.max(1)
        acc = torch.mean((indices.squeeze() == y_target).float())
        c_loss = F.cross_entropy(preds, y_target.long())
        return acc, c_loss

    def _create_optimizer(self, model, lr):
        # setup optimizer
        if self.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.wd)
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.9, weight_decay=self.wd)
        else:
            raise Exception("Not a valid optimizer offered: {0}".format(self.optim))
        return optimizer

    def run_val_epoch(self, batch_size, num_worker):
        total_c_loss = 0.0
        total_accuracy = 0.0
        valdata = self.data.get_testset(batch_size, num_worker)
        total_val_batches = len(valdata)

        with tqdm.tqdm(total=total_val_batches) as pbar:
            for i, (x_support_set, y_support_set, x_target, y_target) in enumerate(valdata):
                x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
                x_target = x_target.permute(0, 3, 1, 2)
                y_support_set = y_support_set.float()
                y_target = y_target.long()
                x_support_set = x_support_set.float()
                x_target = x_target.float()
                with torch.no_grad():
                    acc, c_loss = self.matchNet(x_support_set, y_support_set, x_target, y_target)
                total_c_loss += c_loss.data[0]
                total_accuracy += acc.data[0]
                iter_out = f"v_loss: {total_c_loss / i:.{3}}, v_acc: {total_accuracy / i:.{3}}"
                pbar.set_description(iter_out)
                pbar.update(1)
                # self.total_train_iter+=1

            total_c_loss = total_c_loss / total_val_batches
            total_accuracy = total_accuracy / total_val_batches
            self.scheduler.step(total_c_loss)
            return total_c_loss, total_accuracy