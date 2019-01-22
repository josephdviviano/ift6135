#!/usr/bin/env python
"""
https://github.com/mila-udem/welcome_tutorials/blob/master/pytorch/3.%20Introduction%20to%20the%20Torch%20Neural%20Network%20Library.ipynb
"""
from copy import copy
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

CUDA = torch.cuda.is_available()

class MnistMLP(torch.nn.Module):
    """MLP classifier for MNIST"""
    def __init__(self, h0, hn, ho):
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

        # output
        architecture.append(torch.nn.Linear(hn[-1], ho))

        # use nn to define model
        self.mlp = torch.nn.Sequential(*architecture)
        self.clf = torch.nn.LogSoftmax(dim=0)

    def forward(self, X):
        return(self.clf(self.mlp(X).squeeze()))

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


def run_experiment(clf, lr, epochs, loaders, momentum=False, init_type='glorot'):

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

    if momentum:
        optimizer = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(clf.parameters(), lr=lr)

    #lossfn = torch.nn.CrossEntropyLoss() # don't use! B/C I specify LogSoftmax
    lossfn = torch.nn.NLLLoss()
    #lr_scheduler = ExponentialLR(optimizer, gamma=0.9)

    epoch_loss, valid_acc, train_acc = [], [], []
    best_valid_acc, gen_gap = 0, 0

    all_losses = []
    for ep in range(epochs):

        epoch_losses = []
        #if (ep+1) % 10 == 0:
        #    lr_scheduler.step() # adjusts learning rate

        # training data
        for batch_idx, (X_train, y_train) in enumerate(train):

            if CUDA:
                X_train, y_train = X_train.cuda(), y_train.cuda()

            # initalize batch
            optimizer.zero_grad()
            X_train, y_train = Variable(X_train), Variable(y_train)

            # make predictions -- flatten each image (batchsize x pixels)
            train_pred = clf.forward(X_train.view(X_train.shape[0], -1))

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

        # validation accuracy for this epoch
        this_valid_acc = evaluate(clf, valid)
        valid_acc.append(this_valid_acc)

        # training accuracy for this epoch
        this_train_acc = evaluate(clf, train)
        train_acc.append(this_train_acc)

        # keep track of the best validation accuracy, generalization gap
        if this_valid_acc > best_valid_acc:
            best_valid_acc = this_valid_acc

            if test:
                this_test_acc = evaluate(clf, test)
                gen_gap = this_train_acc - this_test_acc

        # update every n epochs
        if (ep+1) % 5 == 0:
            curr_lr =  optimizer.state_dict()['param_groups'][0]['lr']
            print('+ [{:03d}] loss={:0.6f} acc={:0.2f}/{:0.2f} lr={:0.5f}'.format(
                ep+1, epoch_loss[-1], train_acc[-1], valid_acc[-1], curr_lr))

    results = {'clf': clf,
               'epoch_loss': epoch_loss,
               'all_loss': all_losses,
               'train_acc': train_acc,
               'valid_acc': valid_acc,
               'best_valid_acc': best_valid_acc,
               'gen_gap': gen_gap}

    return(results)


def unique(tensor1d):
    t = np.unique(tensor1d.numpy())
    return(torch.from_numpy(t))


def normalize(t, maxval):
    return(t / maxval)


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


def evaluate(clf, dataset):

    clf.eval()
    total, correct = 0, 0
    for batch_idx, (X_eval, y_eval) in enumerate(dataset):

        if CUDA:
            X_eval, y_eval = X_eval.cuda(), y_eval.cuda()
        # turned off volatile because im suspicious.
        X_eval, y_eval = Variable(X_eval), Variable(y_eval)
        eval_pred = clf.forward(X_eval.view(X_eval.shape[0], -1))
        _, predicted = torch.max(eval_pred.data, 1)

        total += eval_pred.size(0)
        correct += predicted.eq(y_eval.data).cpu().sum()

    clf.train()
    acc = 100.0*correct/total

    return(acc)


def load_documents(preproc=None, batch_size=20):
    """https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a"""
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
    from sklearn.preprocessing import StandardScaler

    train = fetch_20newsgroups()
    test = fetch_20newsgroups(subset='test')

    train_y = train.target
    test_y  = test.target

    count_vec = CountVectorizer(max_features=10000)
    count_vec.fit(train.data)
    train_X = count_vec.transform(train.data)
    test_X = count_vec.transform(test.data)

    if preproc == 'tfidf':
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.fit(train_X)
        train_X = tfidf_transformer.transform(train_X)
        test_X = tfidf_transformer.transform(test_X)


    elif preproc == 'norm':
        scaler = StandardScaler(with_mean=False)
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

    dims_input = train_X.shape[1]
    dims_output = len(np.unique(train_y))

    train_X = torch.FloatTensor(train_X.toarray())
    train_y = torch.IntTensor(train_y)
    test_X = torch.FloatTensor(test_X.toarray())
    test_y = torch.IntTensor(test_y)

    train = TensorDataset(train_X, train_y)
    test = TensorDataset(test_X, test_y)

    train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    test = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=2)

    loaders = [train, test]

    return(loaders, dims_input, dims_output)


def load_mnist(pct=1.0, batch_size=60):
    """
    Loads data, normalizes, and then sets up the training, testing, and
    validation sets using the specified batch_size. pct can be used to select
    a subset of the training set.
    """
    # load and normalize data : train=50000 / valid=10000 / test=10000
    data = MNIST(root='./data')

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True, num_workers=2)

    # calculated model parameters
    X_dims = valid_dataset.data_tensor.shape
    dims_input = X_dims[1]*X_dims[2]
    dims_output = len(unique(valid_dataset.target_tensor))
    loaders = [train_loader, valid_loader, test_loader]

    return(loaders, dims_input, dims_output)


def ex1(mlp, loaders, lr=1e-4, epochs=10):
    """
    Compare three different initial values for weight parameters: all zero,
    normal distribution, or glorot ('xavier') initialization.

    We plot the mean loss over 10 epochs
    """
    results_zero   = run_experiment(mlp, lr, epochs, loaders, init_type='zero')
    results_normal = run_experiment(mlp, lr, epochs, loaders, init_type='normal')
    results_glorot = run_experiment(mlp, lr, epochs, loaders, init_type='glorot')

    plt.plot(results_zero['epoch_loss'])
    plt.plot(results_normal['epoch_loss'])
    plt.plot(results_glorot['epoch_loss'])
    plt.legend(['zero', 'normal', 'glorot'])
    plt.title('comparison of paramater initialization methods')
    plt.xlabel('epoch')
    plt.ylim([0, 10])
    plt.ylabel('mean loss')
    plt.savefig('01_init_loss.jpg')
    plt.close()


def ex2(loaders, lr=1e-4, epochs=10):
    """
    Compare two models, one with double the number of parameters, over
    100 epochs.
    """
    #dims_h = [500, 300] # dims of hidden layers
    dims_h = [300, 100] # dims of hidden layers
    mlp = MnistMLP(dims_0, dims_h, dims_o)
    print('model SMOL: {}\nparameters={}\n'.format(mlp.mlp, mlp.count_params()))

    results = run_experiment(mlp, lr, epochs, loaders, init_type='glorot')

    plt.plot(results['train_acc'])
    plt.plot(results['valid_acc'])
    plt.legend(['training', 'validation'])
    plt.title('learning curves')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('02_accuracy_{}_params.jpg'.format(mlp.count_params()))
    plt.close()

    #dims_h = [900, 450] # dims of hidden layers
    dims_h = [500, 300] # dims of hidden layers
    mlp = MnistMLP(dims_0, dims_h, dims_o)
    print('model BIGG: {}\nparameters={}\n'.format(mlp.mlp, mlp.count_params()))

    results = run_experiment(mlp, lr, epochs, loaders, init_type='glorot')

    plt.plot(results['train_acc'])
    plt.plot(results['valid_acc'])
    plt.legend(['training', 'validation'])
    plt.title('learning curves')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('02_accuracy_{}_params.jpg'.format(mlp.count_params()))
    plt.close()


def ex3(lr=1e-4, epochs=100, batch_size=100):
    """ """
    pcts = [0.01, 0.02, 0.05, 0.1, 1.0]
    dims_h = [500, 300]

    # stores generalization gap for 5 trials across 5 training set sizes
    output = np.zeros((5,5))

    # i = training set size, j = repeats
    for i, pct in enumerate(pcts):
        for j in range(5):

            # calculate generalization gap for each training set size
            loaders, dims_0, dims_o = load_mnist(pct=pct, batch_size=batch_size)
            mlp = MnistMLP(dims_0, dims_h, dims_o)
            results = run_experiment(mlp, lr, epochs, loaders, init_type='glorot')
            output[i, j] = results['gen_gap']
            print('[{},{}] = {}'.format(i, j, results['gen_gap']))

    np.savetxt('03_gen_gap.csv', output)


def ex4(lr=0.2, epochs=20):

    dims_h = [100] # dims of hidden layers

    loaders, dims_0, dims_o = load_documents(preproc=None, batch_size=20)
    mlp = MnistMLP(dims_0, dims_h, dims_o)
    results_raw = run_experiment(mlp, 0.001, epochs, loaders, momentum=True, init_type='glorot')

    loaders, dims_0, dims_o = load_documents(preproc='tfidf', batch_size=20)
    mlp = MnistMLP(dims_0, dims_h, dims_o)
    results_tfidf = run_experiment(mlp, 0.01, epochs, loaders, momentum=True, init_type='glorot')

    loaders, dims_0, dims_o = load_documents(preproc='norm', batch_size=20)
    mlp = MnistMLP(dims_0, dims_h, dims_o)
    results_norm = run_experiment(mlp, 0.001, epochs, loaders, momentum=True, init_type='glorot')

    plt.figure(figsize=(4,12))
    plt.subplot(3,1,1)
    plt.plot(results_raw['train_acc'])
    plt.plot(results_raw['valid_acc'])
    plt.title('no preprocessing, lr=0.001')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'])
    plt.ylim([0, 100])

    plt.subplot(3,1,2)
    plt.plot(results_tfidf['train_acc'])
    plt.plot(results_tfidf['valid_acc'])
    plt.title('tf-idf preprocessing, lr=0.01')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'])
    plt.ylim([0, 100])

    plt.subplot(3,1,3)
    plt.plot(results_norm['train_acc'])
    plt.plot(results_norm['valid_acc'])
    plt.title('z-score preprocessing, lr=0.001')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'])
    plt.ylim([0, 100])

    plt.tight_layout()
    plt.savefig('04_preproc.jpg')
    plt.close()


def ex5(lr=0.2, epochs=20):

    dims_h = [100] # dims of hidden layers

    epochs = 1

    loaders, dims_0, dims_o = load_documents(preproc=None, batch_size=2)
    mlp = MnistMLP(dims_0, dims_h, dims_o)
    results_raw_mb1 = run_experiment(mlp, lr, epochs, loaders, momentum=True, init_type='glorot')

    loaders, dims_0, dims_o = load_documents(preproc='tfidf', batch_size=2)
    mlp = MnistMLP(dims_0, dims_h, dims_o)
    results_tfidf_mb1 = run_experiment(mlp, lr, epochs, loaders, momentum=True, init_type='glorot')

    loaders, dims_0, dims_o = load_documents(preproc='norm', batch_size=2)
    mlp = MnistMLP(dims_0, dims_h, dims_o)
    results_norm_mb1 = run_experiment(mlp, lr, epochs, loaders, momentum=True, init_type='glorot')

    epochs = 100

    loaders, dims_0, dims_o = load_documents(preproc=None, batch_size=100)
    mlp = MnistMLP(dims_0, dims_h, dims_o)
    results_raw_mb100 = run_experiment(mlp, lr, epochs, loaders, momentum=True, init_type='glorot')

    loaders, dims_0, dims_o = load_documents(preproc='tfidf', batch_size=100)
    mlp = MnistMLP(dims_0, dims_h, dims_o)
    results_tfidf_mb100 = run_experiment(mlp, lr, epochs, loaders, momentum=True, init_type='glorot')

    loaders, dims_0, dims_o = load_documents(preproc='norm', batch_size=100)
    mlp = MnistMLP(dims_0, dims_h, dims_o)
    results_norm_mb100 = run_experiment(mlp, lr, epochs, loaders, momentum=True, init_type='glorot')

    plt.plot(results_tfidf_mb1['all_loss'][:5000])
    plt.plot(results_tfidf_mb100['all_loss'][:5000])
    plt.title('update loss on tfidf data')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['mb=1', 'mb=100'])
    plt.savefig('05_mbsize.jpg')


# 1.0: build a model
loaders, dims_0, dims_o = load_mnist(batch_size=100)
dims_h = [512, 512] # dims of hidden layers
mlp = MnistMLP(dims_0, dims_h, dims_o)
print('model INIT: {}\nparameters={}\n'.format(mlp.mlp, mlp.count_params()))

# 1.1: compare initializations
#ex1(mlp, loaders, lr=0.01, epochs=10)

# 2.2: learning curves
#ex3(loaders, lr=0.01, epochs=100)

# 1.3: training set size, genealization gap, standard error
#ex3(lr=0.01, epochs=100)

# 2.0: preprocessing
ex4(epochs=20)

# 2.1: training variance
#ex5(epochs=1)

