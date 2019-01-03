from data_loader import MatchingCifarLoader
from networks import Builder
import utils
import os
import warnings
warnings.simplefilter("ignore")
# Experiment setup
batch_size = 32
# Training setup
total_epochs = 1000
total_train_batches = 1000
total_val_batches = 250
total_test_batches = 500
best_val_acc = 0.0
try:
    os.listdir('/root')
    rootpath = '/root/palm/DATA/'
except PermissionError:
    rootpath = '/home/palm/PycharmProjects/DATA/'
name = 'cifar10'

data = MatchingCifarLoader(os.path.join(rootpath, name))
obj_oneShotBuilder = Builder(data, 'sgd', 1e-2)
try:
    length = len(os.listdir(f'log/{name}/'))
except FileNotFoundError:
    length = 0
logger = utils.Logger(f'log/{name}/{length}')
logger.text_summary('Describe', '10/5 cls, ResNet20', 0)
for e in range(total_epochs):
    total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch(batch_size, 2)
    print("Epoch {}: train_loss:{} train_accuracy:{}".format(e, total_c_loss, total_accuracy))
    total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_val_epoch(10, 2)
    print("Epoch {}: val_loss:{} val_accuracy:{}".format(e, total_val_c_loss, total_val_accuracy))
    logger.scalar_summary('acc', total_accuracy, e)
    logger.scalar_summary('loss', total_c_loss, e)
    logger.scalar_summary('val_acc', total_val_accuracy, e)
    logger.scalar_summary('val_loss', total_val_c_loss, e)
    if total_val_accuracy > best_val_acc:
        best_val_acc = total_val_accuracy
