import copy

import torch
import torch.nn as nn
import numpy as np

from guide_utils.load_model import get_guide_model


class ScanSelflabelGuide(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cluster_counts = [0] * kwargs['k_value']
        self.k = kwargs['k_value']
        self.x_labels = None
        self.scan_selflabel_model, self.head = get_guide_model(setup=kwargs['setup'],
                                                               k_value=kwargs['k_value'],
                                                               num_heads=kwargs['num_heads'],
                                                               model_path=kwargs['model_path'],
                                                               backbone_name=kwargs['backbone_name'])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.discriminator = None

    def fill_x_labels(self, dataloader, N=50000):
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
        self.scan_selflabel_model.eval()
        logits = self.scan_selflabel_model(x, forward_pass='default')[0]
        y = torch.argmax(logits, dim=1)
        return y

    def label_guide(self, x, y):
        ''' Returns the relu guider '''
        self.scan_selflabel_model.eval()
        logits = self.scan_selflabel_model(
            x, forward_pass='default')[
            self.head]
        batch_size = logits.shape[0]
        true_label_logit = logits[torch.arange(0, batch_size), y]
        return -(self.relu(logits - true_label_logit.unsqueeze(-1))).mean(dim=-1)

    def reg_label_guide(self, x, y):
        label_guide = self.label_guide(x, y)
        return -label_guide.mean()

    def get_label_distribution(self):
        '''returns the empirical distributon of clustering'''
        y = self.x_labels
        nclusters = self.scan_selflabel_model.nclusters
        counts = [0] * nclusters
        for yi in y:
            counts[yi] += 1
        print('get_label_distribution counts', counts)
        return counts

    def sample_y(self, batch_size):
        '''samples y according to the empirical distribution'''
        distribution = self.get_label_distribution()
        distribution = [i / sum(distribution) for i in distribution]
        m = torch.distributions.Multinomial(batch_size,
                                            torch.tensor(distribution))
        return m.sample()

    def print_label_distribution(self, x=None):
        print(self.get_label_distribution(x))

    def test_acc(self, testloader):
        from sklearn import metrics
        self.scan_selflabel_model.eval()
        total = 0
        predictions = []
        true_labels = []
        with torch.no_grad():
            for images, labels in testloader:
                outputs = self.scan_selflabel_model(images.cuda(), forward_pass='default')[0]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                predictions.extend(list(predicted.cpu().numpy()))
                true_labels.extend(list(labels.cpu().numpy()))
        nmi = metrics.normalized_mutual_info_score(true_labels, predictions)

        print('NMI of the network on the %d images: %0.4f %%' % (total, nmi))
