#!/usr/bin/env python
from copy import copy
from glob import glob
from scipy.ndimage import imread

import torch
from mnist import * # import MNIST class
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from torch.autograd import Variable
from torch.nn.init import calculate_gain
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2


CUDA = torch.cuda.is_available()
DC_TRAIN = 'datasets/train_64x64'
DC_VALID = 'datasets/valid_64x64'


class DeepBoi(torch.nn.Module):
    """
    Convnet Classifier taken from "Very deep convolutional networks for
    large-scale image recognition"
    """
    def __init__(self, conv_blocks, in_channels=3, n_out=2):
        super(DeepBoi, self).__init__()

        # builds the convolutional layers
        conv_arch = []

        for block in conv_blocks:
            for convlayers in range(block[1]):

                # conv2d --> batchnorm --> relu, done n=convlayer times / block
                conv_arch.extend([
                    torch.nn.Conv2d(in_channels=in_channels,
                        out_channels=block[0], kernel_size=(3, 3), padding=1),
                    torch.nn.BatchNorm2d(block[0]),
                    torch.nn.ReLU()])

                in_channels = block[0]

            # add a maxpool layer between blocks
            conv_arch.append(torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        # builds the fully connected layers
        self.conv = torch.nn.Sequential(*conv_arch)
        test_data = Variable(torch.rand(1, 3, 64, 64))
        test_data = self.conv(test_data)
        init_layer = test_data.data.view(1, -1).size(1)

        fc_layers = [init_layer, 1024, 1024]
        fc_arch = []
        for i in range(1, len(fc_layers)):
            fc_arch.extend([
                torch.nn.Linear(fc_layers[i-1], fc_layers[i]),
                torch.nn.BatchNorm2d(fc_layers[i]),
                torch.nn.ReLU()])
        fc_arch.append(torch.nn.Linear(fc_layers[-1], n_out))

        # assemble the model
        self.fc = torch.nn.Sequential(*fc_arch)
        self.clf = torch.nn.LogSoftmax(dim=0)


    def forward(self, x):
        # input.view(self.size(0), -1)
        x = self.conv(x)
        dims = x.shape
        x = x.view(dims[1]*dims[2]*dims[3], -1).transpose(0, 1)
        return(self.fc(x))

    def predict(self, X):
        return(self.clf(X))

    # init_type here for compatibility with previous models but is ugly af
    def initalizer(self, init_type=None):
        for k, v in self.conv.named_parameters():
            if k.endswith('weight') and len(v.shape) > 1:
                torch.nn.init.xavier_uniform(v, gain=calculate_gain('relu'))


def unique(tensor1d):
    t = np.unique(tensor1d.numpy())
    return(torch.from_numpy(t))


def run_experiment(clf, epochs, loaders, init_type='glorot'):

    train, valid, test = loaders
    clf.initalizer(init_type=init_type)

    if CUDA:
        clf = clf.cuda()

    optimizer = torch.optim.Adam(clf.parameters())
    lossfn = torch.nn.NLLLoss()

    epoch_loss, valid_acc, train_acc, test_acc = [], [], [], []
    best_valid_acc, gen_gap = 0, 0

    all_norms = []
    all_losses = []
    for ep in range(epochs):

        epoch_losses = []

        # training data
        for batch_idx, (X_train, y_train) in enumerate(train):

            if CUDA:
                X_train, y_train = X_train.cuda(), y_train.cuda()

            # initalize batch
            optimizer.zero_grad()
            X_train, y_train = Variable(X_train), Variable(y_train)

            # make predictions: 4d for conv, 2d for mlp
            train_fwd = clf.forward(X_train)
            #train_fwd = clf.forward(X_train.unsqueeze(1)) # because no channel dimention for mnist
            train_pred = clf.predict(train_fwd)

            # calculate loss (cross entropy)
            loss = lossfn(train_pred, y_train)
            epoch_losses.append(loss.data[0])
            all_losses.append(loss.data[0])

            # calculate dloss/dx for all parameters that have requires_grad
            loss.backward()

            # update paramater values
            optimizer.step()

        # average loss for epoch
        epoch_loss.append(np.mean(epoch_losses))

        # training accuracy for this epoch
        this_train_acc = evaluate(clf, train)
        train_acc.append(this_train_acc)
        this_valid_acc = evaluate(clf, valid)
        valid_acc.append(this_valid_acc)
        this_test_acc = evaluate(clf, test)
        test_acc.append(this_test_acc)

        # keep track of the best validation accuracy, generalization gap
        if this_valid_acc > best_valid_acc:
            best_valid_acc = this_valid_acc
            gen_gap = this_train_acc - this_test_acc

        # update every n epochs
        if (ep+1) % 5 == 0:
            print('+ [{:03d}] loss={:0.6f} acc={:0.2f}/{:0.2f}'.format(
                ep+1, epoch_loss[-1], train_acc[-1], valid_acc[-1]))

    results = {'clf': clf,
               'epoch_loss': epoch_loss,
               'all_loss': all_losses,
               'train_acc': train_acc,
               'valid_acc': valid_acc,
               'test_acc': test_acc,
               'best_valid_acc': best_valid_acc,
               'gen_gap': gen_gap,
               'norms': np.array(all_norms)}

    return(results)


def evaluate(clf, dataset):

    clf.eval()
    total, correct = 0, 0
    for batch_idx, (X_eval, y_eval) in enumerate(dataset):

        if CUDA:
            X_eval, y_eval = X_eval.cuda(), y_eval.cuda()

        X_eval, y_eval = Variable(X_eval), Variable(y_eval)
        eval_fwd = clf.forward(X_eval)
        #eval_fwd = clf.forward(X_eval.unsqueeze(1)) # because no channel dimention for mnist
        eval_pred = clf.predict(eval_fwd)
        _, predicted = torch.max(eval_pred.data, 1)

        import IPython; IPython.embed()

        total += eval_pred.size(0)
        correct += predicted.eq(y_eval.data).cpu().sum()

    clf.train()
    acc = 100.0*correct/total

    return(acc)


def parse_jpegs(files):
    mats = []
    labels = []

    for d in files:

        if 'Cat' in d:
            labels.append(0)
        else:
            labels.append(1)

        a = cv2.imread(d, cv2.IMREAD_COLOR)
        a = cv2.resize(a, (64, 64))
        mats.append(a.T)

    mats = np.stack(mats, axis=0)
    labels = np.array(labels)

    return(mats.astype(np.float), labels)


def load_dc(batch_size=64):
    """Loads cat and dog data into train / valid / test dataloders"""
    if os.path.isfile('test_mats.npy'):
        test_mats = np.load('test_mats.npy')
        train_mats = np.load('train_mats.npy')
        valid_mats = np.load('valid_mats.npy')
        test_labels = np.load('test_labels.npy')
        train_labels = np.load('train_labels.npy')
        valid_labels = np.load('valid_labels.npy')

    else:
        train_files = glob(os.path.join(DC_TRAIN, '*.jpg'))
        valid_files = glob(os.path.join(DC_VALID, '*.jpg'))

        train_mats, train_labels = parse_jpegs(train_files)
        valid_mats, valid_labels = parse_jpegs(valid_files)

        means = [np.mean(train_mats[:,0,:,:]),
                 np.mean(train_mats[:,1,:,:]),
                 np.mean(train_mats[:,2,:,:])]

        train_mats[:,0,:,:] -= means[0]
        train_mats[:,1,:,:] -= means[1]
        train_mats[:,2,:,:] -= means[2]
        valid_mats[:,0,:,:] -= means[0]
        valid_mats[:,1,:,:] -= means[1]
        valid_mats[:,2,:,:] -= means[2]

        valid_cutoff = valid_mats.shape[0] // 2
        test_mats = valid_mats[valid_cutoff:, :, :]
        valid_mats = valid_mats[:valid_cutoff, :, :]
        test_labels = valid_labels[valid_cutoff:]
        valid_labels = valid_labels[:valid_cutoff]

        np.save('test_mats.npy', test_mats)
        np.save('train_mats.npy', train_mats)
        np.save('valid_mats.npy', valid_mats)
        np.save('test_labels.npy', test_labels)
        np.save('train_labels.npy', train_labels)
        np.save('valid_labels.npy', valid_labels)

    # convert
    train_dataset = TensorDataset(torch.FloatTensor(train_mats), torch.ByteTensor(train_labels))
    valid_dataset = TensorDataset(torch.FloatTensor(valid_mats), torch.ByteTensor(valid_labels))
    test_dataset  = TensorDataset(torch.FloatTensor(test_mats),  torch.ByteTensor(test_labels))

    # loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True, num_workers=2)

    loaders = [train_loader, valid_loader, test_loader]

    return(loaders)

    def count_params(mdl):
        param_count = 0
        for k, v in mdl.named_parameters():
            param_count += np.prod(np.array(v.size()))

        print(param_count)


def main():

    epochs = 25

    conv_archs = [[(16, 1), (32, 1), (64, 1)],
                  [(32, 1), (64, 1), (128, 1)],
                  [(32, 1), (64, 1), (128, 1), (64, 1), (32, 1)]]
    batch_sizes = [4, 8, 16, 32, 64]

    counter = 1
    for arch in conv_archs:
        for batch in batch_sizes:

            print('training model {}/15'.format(counter))

            loaders = load_dc(batch_size=batch)
            boi = DeepBoi(arch, in_channels=3, n_out=2)
            model = run_experiment(boi, epochs, loaders)

            if counter == 1:
                best_model = model
            else:
                if np.max(model['valid_acc']) > np.max(best_model['valid_acc']):
                    print('new best model: {} {}'.format(arch, batch))
                    best_model = model
            counter += 1

    import IPython; IPython.embed()

    plt.plot(best_model['train_acc'])
    plt.plot(best_model['valid_acc'])
    plt.legend(['training', 'validation'])
    plt.title('learning curves for best performing model')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('10_accuracy.jpg')

    batch_size = 16
    loaders = load_dc(batch_size=batch)

    evaluate(best_model['clf'], loaders[2])



if __name__ == '__main__':
   main()

