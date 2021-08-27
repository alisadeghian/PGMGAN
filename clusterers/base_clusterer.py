import copy

import torch
import numpy as np

class BaseClusterer():
    def __init__(self,
                 k_value=-1,
                 x_cluster=None,
                 batch_size=100,
                 **kwargs):
        ''' requires that self.x is not on the gpu, or else it hogs too much gpu memory '''
        self.cluster_counts = [0] * k_value
        self.k = k_value
        self.kmeans = None
        self.x = x_cluster
        self.x_labels = None
        self.batch_size = batch_size

    def get_labels(self, x, y):
        return y

    def get_label_distribution(self, x=None):
        '''returns the empirical distributon of clustering'''
        y = self.x_labels if x is None else self.get_labels(x, None)
        counts = [0] * self.k
        for yi in y:
            counts[yi] += 1
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
