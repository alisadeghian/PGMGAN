import torch
from clusterers import base_clusterer


class Clusterer(base_clusterer.BaseClusterer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k = 1
        print('Setting k to 1 regardless of config.')

    def get_labels(self, x, y):
        return torch.randint(low=0, high=1, size=y.shape).long().cuda()
