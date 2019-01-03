from data_loader import MatchingCifarLoader

a = MatchingCifarLoader('/home/palm/PycharmProjects/DATA/cifar10')
test = a.testset
t = test[1]
