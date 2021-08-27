# Used for CIFAR10 experiments
import copy

import torch
import torch.nn as nn
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score


class iNet(nn.Module):
    def __init__(self, k=None):
        super().__init__()
        print('Using Invertible Net for Guide')
        self.fp = nn.Sequential(nn.Linear(2, k))

    def forward(self, x):
        return self.fp(x)


class Net(nn.Module):
    def __init__(self, k=None):
        super().__init__()
        print('Using FC for Guide')
        self.k = k
        self.fp = nn.Sequential(nn.Linear(2, 10),
                                nn.ReLU(),
                                nn.Linear(10, 20),
                                nn.ReLU(),
                                nn.Linear(20, k)
                                )

    def forward(self, x):
        return self.fp(x)


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, k=None, X=None):
        'Initialization'
        X = X.cpu().detach().numpy()
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=100, max_iter=3000).fit(X)
        self.labels = torch.from_numpy(kmeans.labels_).long()
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.X[index]
        y = self.labels[index]

        return X, y


class ToyGuideClusterer(nn.Module):
    def __init__(self,
                 discriminator,
                 k_value=-1,  # TODO: make sure k_value is correct
                 x_cluster=None,
                 x_labels=None,
                 batch_size=100,
                 invertible=False,
                 **kwargs):
        ''' requires that self.x is not on the gpu, or else it hogs too much gpu memory '''
        super().__init__()
        self.cluster_counts = [0] * k_value
        self.k = k_value
        self.x = x_cluster
        self.x_labels = None
        self.batch_size = batch_size
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dataset = Dataset(k=self.k, X=self.x)
        self.invertible = invertible

        if invertible:
            self.classifier_model = iNet(k=k_value)
        else:
            self.classifier_model = Net(k=k_value)
        self.classifier_model.cuda()

    def recluster(self, discriminator, **kwargs):

        trainloader = torch.utils.data.DataLoader(self.dataset, batch_size=5000)

        criterion = nn.CrossEntropyLoss()

        if self.invertible:
            N_epoch = 2000
            lr = 0.07
        else:
            N_epoch = 2000
            lr = 0.07

        optimizer = torch.optim.Adagrad(self.classifier_model.parameters(), lr=lr)

        for epoch in range(N_epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.classifier_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            if epoch % 50 == 0:
                print('Epoch %d, loss: %.3f' % (epoch, running_loss / i))

        outputs = self.classifier_model(self.dataset.X.cuda())
        pred_labels = outputs.argmax(dim=1)
        nmi = normalized_mutual_info_score(pred_labels.cpu().detach().numpy(), self.dataset.labels.detach().numpy())
        print('Finished Training, NMI = %.3f' % nmi)
        return None

    def fill_x_labels(self, dataloader, N=50000):
        raise Exception('Should this be called with an outside dataloader?')
        x = []
        y = []
        n = 0
        for x_next, y_next in dataloader:
            x.append(x_next)
            y.append(self.get_labels(x_next.cuda(), None))
            n += x_next.size(0)
            if n > N:
                break
        x = torch.cat(x, dim=0)[:N]
        y = torch.cat(y, dim=0)[:N]
        self.x, self.x_labels = x, y

    def get_labels(self, x, y):

        self.classifier_model.eval()

        outputs = self.classifier_model(x)
        y = outputs.argmax(dim=1)

        return y

    def label_guide(self, x, y):
        ''' Returns the relu guider '''
        self.classifier_model.eval()
        logits = self.classifier_model(x)
        batch_size = logits.shape[0]
        true_label_logit = logits[torch.arange(0, batch_size), y]
        return -(self.relu(logits - true_label_logit.unsqueeze(-1))).mean(dim=-1)  # TODO: try sum

    def reg_label_guide(self, x, y):
        label_guide = self.label_guide(x, y)
        return -label_guide.mean()

    def get_label_distribution(self):  # TODO: adjust for imagenet
        '''returns the empirical distributon of clustering'''
        y = self.x_labels
        nclusters = self.k
        counts = [0] * nclusters
        for yi in y:
            counts[yi] += 1
        print('get_label_distribution counts', counts)
        return counts

    def sample_y(self, batch_size):
        '''samples y according to the empirical distribution (not sure if used anymore)'''
        distribution = self.get_label_distribution()
        distribution = [i / sum(distribution) for i in distribution]
        m = torch.distributions.Multinomial(batch_size,
                                            torch.tensor(distribution))
        return m.sample()

    def print_label_distribution(self, x=None):
        print(self.get_label_distribution(x))
