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
from time import gmtime, strftime

CUDA = torch.cuda.is_available()
DC_TRAIN = 'datasets/train_64x64'
DC_VALID = 'datasets/valid_64x64'

class MnistMLP(torch.nn.Module):
    """MLP classifier for MNIST"""
    def __init__(self, h0, hn, ho, dropout=False):
        """
        h0 -- input size (flat)
        hn -- a list of all hidden layer sizes
        ho -- output size (number of classes; flat)
        """
        super(MnistMLP, self).__init__()

        # input --> hid1
        architecture = [torch.nn.Linear(h0, hn[0]), torch.nn.ReLU()]

        # hidden layers
        for i in range(1, len(hn)):
            architecture.append(torch.nn.Linear(hn[i-1], hn[i]))
            architecture.append(torch.nn.ReLU())

        if dropout:
            # dropout on last hidden layer
            architecture.append(torch.nn.Dropout())

        # output
        architecture.append(torch.nn.Linear(hn[-1], ho))

        # use nn to define model
        self.mlp = torch.nn.Sequential(*architecture)
        self.clf = torch.nn.LogSoftmax(dim=0)

    def forward(self, X):
        return(self.mlp(X).squeeze())

    def predict(self, X):
        return(self.clf(X).squeeze())

    def initalizer(self, init_type='glorot'):
        """
        model     -- a pytorch sequential model
        init_type -- one of 'zero', 'normal', 'glorot'

        Takes in a model, initializes it to all-zero, normal distribution
        sampled, or glorot initialization. Golorot == xavier.
        """
        if init_type not in ['zero', 'normal', 'glorot']:
            raise Exception('init_type invalid]')

        for k, v in self.mlp.named_parameters():
            if k.endswith('weight'):
                if init_type == 'zero':
                    torch.nn.init.constant(v, 0)
                elif init_type == 'normal':
                    torch.nn.init.normal(v)
                elif init_type == 'glorot':
                    torch.nn.init.xavier_uniform(v, gain=calculate_gain('relu'))
                else:
                    raise Exception('invalid init_type')

    def count_params(self):
        """
        Returns a count of all parameters
        """
        param_count = 0
        for k, v in self.mlp.named_parameters():
            param_count += np.prod(np.array(v.size()))

        return(param_count)

    def l2_norm(self):
        """Returns a plot of the l2 norm of all weight matrices"""
        norms = []
        for k, v in self.mlp.named_parameters():
            if k.endswith('weight'):
                #norms.append(v.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10)
                #norms.append(v.pow(2).sum().sqrt() + 1e-10)
                norms.append(v.norm(2))

        return(sum(torch.cat(norms)))


class CNN(torch.nn.Module):
    """
    Convnet Classifier taken from
    hsttps://git,ubs.com/MaximumEntropy/welcome_tutorials/blob/pytorch/pytorch/4.%20Image%20Classification%20with%20Convnets%20and%20ResNets.ipynb
    """
    def __init__(self, batchnorm=False):
        super(CNN, self).__init__()

        architecture = [torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)]

        if batchnorm:
            architecture.append(torch.nn.BatchNorm2d(16))

        architecture.extend([torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)])

        if batchnorm:
            architecture.append(torch.nn.BatchNorm2d(32))

        architecture.extend([torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)])

        if batchnorm:
            architecture.append(torch.nn.BatchNorm2d(64))

        architecture.extend([torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)])

        if batchnorm:
            architecture.append(torch.nn.BatchNorm2d(128))

        architecture.extend([torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)])

        # conv
        self.conv = torch.nn.Sequential(*architecture)

        # Logistic Regression
        #self.clf = torch.nn.Linear(128, 10)
        self.fc = torch.nn.Linear(128, 10)
        self.clf = torch.nn.LogSoftmax(dim=0)

    def forward(self, x):
        return(self.fc(self.conv(x).squeeze()))

    def predict(self, X):
        return(self.clf(X))

    def initalizer(self, init_type='glorot'):
        """
        model     -- a pytorch sequential model
        init_type -- one of 'zero', 'normal', 'glorot'

        Takes in a model, initializes it to all-zero, normal distribution
        sampled, or glorot initialization. Golorot == xavier.
        """
        if init_type not in ['zero', 'normal', 'glorot']:
            raise Exception('init_type invalid]')

        for k, v in self.conv.named_parameters():
            if k.endswith('weight') and len(v.shape) > 1:
                if init_type == 'zero':
                    torch.nn.init.constant(v, 0)
                elif init_type == 'normal':
                    torch.nn.init.normal(v)
                elif init_type == 'glorot':
                    torch.nn.init.xavier_uniform(v, gain=calculate_gain('relu'))
                else:
                    raise Exception('invalid init_type')


class DeepBoi(torch.nn.Module):
    """
    Convnet Classifier taken from "Very deep convolutional networks for
    large-scale image recognition"
    """
    def __init__(self, in_channels=3, n_out=2):
        super(DeepBoi, self).__init__()

        # builds the convolutional layers
        # [(n_filters, n_repeats / block), ...]
        #conv_blocks = [(64, 2), (128, 2), (256, 2), (512, 4), (512, 4)]
        #conv_blocks = [(64, 2), (128, 2), (256, 2), (512, 4)]
        conv_blocks = [(64, 2), (128, 2), (128, 2), (256, 4)]
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
        #import IPython; IPython.embed()
        init_layer = conv_blocks[-1][0] * 4 * 4 # ???
        #fc_layers = [init_layer, 4096, 4096, 1000]
        fc_layers = [init_layer, 2048, 2048, 500]
        fc_arch = []
        for i in range(1, len(fc_layers)):
            fc_arch.extend([
                torch.nn.Linear(fc_layers[i-1], fc_layers[i]),
                torch.nn.BatchNorm2d(fc_layers[i]),
                torch.nn.ReLU()])
        fc_arch.append(torch.nn.Linear(fc_layers[-1], n_out))

        # assemble the model
        self.conv = torch.nn.Sequential(*conv_arch)
        self.fc = torch.nn.Sequential(*fc_arch)
        self.clf = torch.nn.LogSoftmax(dim=0)


    def forward(self, x):
        #import IPython; IPython.embed()
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


def normalize(t, maxval):
    return(t / maxval)


def genmask(size):
    all_masks = []
    for i in range(size[0]):
        all_masks.append(np.random.choice([0, 1], size=(size[1], )))

    return(np.vstack(all_masks))


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return(sparse_tensortype(indices, values, x.size()))


def run_experiment(clf, lr, epochs, loaders, momentum=0, l2=0, init_type='glorot', dropout=False, convmode=False):

    if len(loaders) == 3:
        train, valid, test = loaders
    elif len(loaders) == 2:
        train, valid = loaders
        test = None
    else:
        raise Exception('loaders malformed')

    clf.initalizer(init_type=init_type)

    if CUDA:
        clf = clf.cuda()

    optimizer = torch.optim.SGD(clf.parameters(), lr=lr, momentum=momentum,
        weight_decay=l2)

    lossfn = torch.nn.NLLLoss()

    epoch_loss, valid_acc, train_acc = [], [], []
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
            #import IPython; IPython.embed()
            if convmode:
                if len(X_train.shape) == 3:
                    train_fwd = clf.forward(X_train.unsqueeze(1)) # because no channel dimention for mnist
                else:
                    train_fwd = clf.forward(X_train)
            else:
                train_fwd = clf.forward(X_train.view(X_train.shape[0], -1))

            if dropout:
                train_fwd = train_fwd * 0.5

            train_pred = clf.predict(train_fwd)

            # calculate loss (cross entropy)
            loss = lossfn(train_pred, y_train)
            epoch_losses.append(loss.data[0])
            all_losses.append(loss.data[0])

            # calculate dloss/dx for all parameters that have requires_grad
            loss.backward()

            # update paramater values
            optimizer.step()

            if not convmode:
                # store l2 norms of model
                norm = clf.l2_norm()
                all_norms.append(float(norm))

        # average loss for epoch
        epoch_loss.append(np.mean(epoch_losses))

        # validation accuracy for this epoch
        this_valid_acc = evaluate(clf, valid, dropout=dropout, convmode=convmode)
        valid_acc.append(this_valid_acc)

        # training accuracy for this epoch
        this_train_acc = evaluate(clf, train, dropout=dropout, convmode=convmode)
        train_acc.append(this_train_acc)

        # keep track of the best validation accuracy, generalization gap
        if this_valid_acc > best_valid_acc:
            best_valid_acc = this_valid_acc

            if test:
                this_test_acc = evaluate(clf, test, dropout=dropout, convmode=convmode)
                gen_gap = this_train_acc - this_test_acc

        # update every n epochs
        if (ep+1) % 1 == 0:
            curr_lr =  optimizer.state_dict()['param_groups'][0]['lr']
            print('+ [{:03d}] loss={:0.6f} acc={:0.2f}/{:0.2f} lr={:0.5f}'.format(
                ep+1, epoch_loss[-1], train_acc[-1], valid_acc[-1], curr_lr))

    results = {'clf': clf,
               'epoch_loss': epoch_loss,
               'all_loss': all_losses,
               'train_acc': train_acc,
               'valid_acc': valid_acc,
               'best_valid_acc': best_valid_acc,
               'gen_gap': gen_gap,
               'norms': np.array(all_norms)}

    return(results)


def evaluate(clf, dataset, dropout=False, convmode=False):

    clf.eval()
    total, correct = 0, 0
    for batch_idx, (X_eval, y_eval) in enumerate(dataset):

        if CUDA:
            X_eval, y_eval = X_eval.cuda(), y_eval.cuda()

        X_eval, y_eval = Variable(X_eval), Variable(y_eval)


        if convmode:
            if len(X_eval.shape) == 3:
                eval_fwd = clf.forward(X_eval.unsqueeze(1)) # because no channel dimention for mnist
            else:
                eval_fwd = clf.forward(X_eval)
        else:
            eval_fwd = clf.forward(X_eval.view(X_eval.shape[0], -1))

        if dropout:
            eval_fwd = eval_fwd * 0.5

        eval_pred = clf.predict(eval_fwd)
        _, predicted = torch.max(eval_pred.data, 1)

        total += eval_pred.size(0)
        correct += predicted.eq(y_eval.data).cpu().sum()

    clf.train()
    acc = 100.0*correct/total

    return(acc)


def evaluate_dropout(clf, dataset, eval_mode='dropout'):

    clf.eval()

    ns_acc = np.zeros(10)
    for i, N in enumerate(np.arange(10, 101, 10)):

        total, correct = 0, 0

        for batch_idx, (X_eval, y_eval) in enumerate(dataset):

            if CUDA:
                X_eval, y_eval = X_eval.cuda(), y_eval.cuda()

            X_eval, y_eval = Variable(X_eval), Variable(y_eval)
            eval_fwd = clf.forward(X_eval.view(X_eval.shape[0], -1))

            if eval_mode == 'dropout':
                eval_fwd = eval_fwd * 0.5
                eval_pred = clf.predict(eval_fwd)

            elif eval_mode == 'mask_pre_softmax':
                for j in range(N):
                    if j == 0:
                        mask = genmask(tuple(eval_fwd.shape))
                    else:
                        mask += genmask(tuple(eval_fwd.shape))
                mask = mask / N
                mask = torch.FloatTensor(mask)

                if CUDA:
                    mask = mask.cuda()

                eval_fwd = torch.mul(Variable(mask), eval_fwd)
                eval_pred = clf.predict(eval_fwd)

            elif eval_mode == 'mask_post_softmax':
                for j in range(N):
                    mask = torch.FloatTensor(genmask(tuple(eval_fwd.shape)))

                    if CUDA:
                        mask = mask.cuda()

                    if j == 0:
                        eval_pred = torch.mul(Variable(mask), clf.predict(eval_fwd))
                    else:
                        eval_pred = eval_pred + torch.mul(Variable(mask), clf.predict(eval_fwd))


            _, predicted = torch.max(eval_pred.data, 1)

            total += eval_pred.size(0)
            correct += predicted.eq(y_eval.data).cpu().sum()
            ns_acc[i] = 100.0*correct/total

    clf.train()

    return(ns_acc)


def parse_jpegs(files):
    mats = []
    labels = []

    for d in files:
        a = imread(d)

        if 'Cat' in d:
            labels.append(0)
        else:
            labels.append(1)

        # we have some greyscale images with no color channel, so we expand
        # it (duplicate) so we do...
        if len(a.shape) != 3:
            a = np.repeat(np.expand_dims(a, axis=-1), 3, axis=-1)

        mats.append(a.T)

    mats = np.stack(mats, axis=0)
    labels = np.array(labels)

    return(mats.astype(np.float), labels)


def load_dc(batch_size=64):
    """Loads cat and dog data into train / valid / test dataloders"""
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

    test_mats = valid_mats[valid_cutoff:, :, :, :]
    valid_mats = valid_mats[:valid_cutoff, :, :, :]
    test_labels = valid_labels[valid_cutoff:]
    valid_labels = valid_labels[:valid_cutoff]

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



def load_mnist(pct=1.0, batch_size=60):
    """
    Loads data, normalizes, and then sets up the training, testing, and
    validation sets using the specified batch_size. pct can be used to select
    a subset of the training set.
    """
    # load and normalize data : train=50000 / valid=10000 / test=10000
    data = MNIST(root='./data', download=True)

    # 0-1 normalize (min/max)
    maxval = torch.max(data.train_data.float())
    data.train_data = normalize(data.train_data.float(), maxval)
    data.valid_data = normalize(data.valid_data.float(), maxval)
    data.test_data = normalize(data.test_data.float(), maxval)

    # take a subset of the training data
    if pct < 1.0:
        m = data.train_data.shape[0]
        idx = np.random.choice(np.arange(m), round(m*pct))
        data.train_data = data.train_data[idx, :, :]
        data.train_labels = data.train_labels[idx,]

    # convert
    train_dataset = TensorDataset(data.train_data, data.train_labels)
    valid_dataset = TensorDataset(data.valid_data, data.valid_labels)
    test_dataset  = TensorDataset(data.test_data,  data.test_labels)

    # load
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    # calculated model parameters
    X_dims = valid_dataset.data_tensor.shape
    dims_input = X_dims[1]*X_dims[2]
    dims_output = len(unique(valid_dataset.target_tensor))
    loaders = [train_loader, valid_loader, test_loader]

    return(loaders, dims_input, dims_output)


def q1a():
    """question 1a -- l2 norm compare"""
    batch_size = 64
    loaders, dims_0, dims_o = load_mnist(batch_size=batch_size)
    dims_h = [800, 800] # dims of hidden layers
    mlp = MnistMLP(dims_0, dims_h, dims_o)
    lr = 0.02
    epochs = 100
    lambda_l2 = 2.5 / (len(loaders[0].dataset.data_tensor)/batch_size)

    baseline = run_experiment(mlp, lr, epochs, loaders, l2=0, init_type='glorot')
    l2_norm = run_experiment(mlp, lr, epochs, loaders, l2=lambda_l2, init_type='glorot')

    plt.plot(baseline['norms'])
    plt.plot(l2_norm['norms'])
    plt.legend(['no weight decay', 'weight decay, lambda={}'.format(lambda_l2)])
    plt.title('l2 norms with and without weight decay')
    plt.xlabel('epoch')
    plt.ylabel('l2 norm')
    plt.savefig('01_weight_decay_lambda={}.jpg'.format(lambda_l2))
    plt.close()

    plt.plot(100 - np.array(baseline['valid_acc']))
    plt.plot(100 - np.array(l2_norm['valid_acc']))
    plt.legend(['no weight decay', 'weight decay, lambda={}'.format(lambda_l2)])
    plt.title('error rate with and without weight decay')
    plt.xlabel('epoch')
    plt.ylabel('error rate')
    plt.savefig('02_error_rate_lambda={}.jpg'.format(lambda_l2))
    plt.close()

def q1b():
    """question 1b -- compare dropout masks etc"""
    batch_size = 64
    loaders, dims_0, dims_o = load_mnist(batch_size=batch_size)
    dims_h = [800, 800] # dims of hidden layers
    mlp = MnistMLP(dims_0, dims_h, dims_o, dropout=True)
    lr = 0.02
    epochs = 100
    lambda_l2 = 2.5 / (len(loaders[0].dataset.data_tensor)/batch_size)

    model = run_experiment(mlp, lr, epochs, loaders, l2=0, init_type='glorot', dropout=True)

    accs_mult = evaluate_dropout(model['clf'], loaders[1], eval_mode='dropout')
    accs_pre = evaluate_dropout(model['clf'], loaders[1], eval_mode='mask_pre_softmax')
    accs_post = evaluate_dropout(model['clf'], loaders[1], eval_mode='mask_post_softmax')

    plt.plot(accs_mult)
    plt.plot(accs_pre)
    plt.plot(accs_post)
    plt.legend(['mult 1/2', 'mask pre softmax', 'mask post softmax'])
    plt.title('evaluating dropout models')
    plt.xlabel('N')
    plt.xticks(range(10), np.arange(10, 101, 10))
    plt.ylabel('accuracy')
    plt.savefig('03_dropout.jpg')
    plt.close()

def q1c():
    """question 1c -- convnet with batchnorm"""
    lr = 0.02
    epochs = 10
    batch_size = 64
    loaders, dims_0, dims_o = load_mnist(batch_size=batch_size)

    cnn = CNN()
    model_nobatchnorm = run_experiment(cnn, lr, epochs, loaders, l2=0, init_type='glorot', convmode=True)
    cnn = CNN(batchnorm=True)
    model_batchnorm = run_experiment(cnn, lr, epochs, loaders, l2=0, init_type='glorot', convmode=True)

    plt.plot(100 - np.array(model_nobatchnorm['valid_acc']))
    plt.plot(100 - np.array(model_batchnorm['valid_acc']))
    plt.legend(['no batch normalization', 'batch normalization'])
    plt.title('CNNs with and without batch normalization')
    plt.xlabel('epoch')
    plt.ylabel('error rate')
    plt.savefig('04_cnn_batchnorm.jpg')
    plt.close()

def q2():
    lr = 0.02
    epochs = 20
    batch_size = 50
    boi = DeepBoi(in_channels=3, n_out=2)
    loaders = load_dc(batch_size=batch_size)
    l2 = 2.5 / (len(loaders[0].dataset.data_tensor)/batch_size)
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    model = run_experiment(boi, lr, epochs, loaders, l2=l2, convmode=True)
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    import IPython; IPython.embed()

if __name__ == '__main__':
   #q1a()
   #q1b()
   #q1c()
   q2()
