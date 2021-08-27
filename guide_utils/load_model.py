import torch

from .resnet_cifar import resnet18
from .resnet import resnet50
from .resnet_cifar_leaky_relu import resnet18_leaky_relu
# from .models import ClusteringModel

from models.models import ClusteringModel
from models.invertible_resnet.models.conv_iResNet import iresnet


def get_guide_model(setup, k_value, num_heads, model_path, backbone_name):

    if backbone_name == 'resnet18':
        backbone = resnet18()
    elif backbone_name == 'iresnet':
        model_checkpoint = torch.load(model_path)
        model = model_checkpoint['model']
        head = model_checkpoint['head']
        print("check_point['head']", model_checkpoint['head'])
        model.nclusters = k_value
        return model, head
    elif backbone_name == 'resnet18_leaky_relu':
        backbone = resnet18_leaky_relu()
    else:
        raise NotImplementedError

    model = ClusteringModel(backbone, k_value, num_heads)
    print('Loaded partitioner model...')
    print(model)
    print('_'*20)
    print('Loading model parameters from: %s' % model_path)
    check_point = torch.load(model_path, map_location='cpu')

    if setup in ['simclr', 'moco', 'selflabel']:
        model.load_state_dict(check_point)

    elif setup == 'scan':
        model.load_state_dict(check_point['model'])
        head = check_point['head']
        print("check_point['head']", check_point['head'])

    model.nclusters = k_value

    return model, head
