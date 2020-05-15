import torch


def get_device(params):
    if params['resource']['num_gpus'] > 0:
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')
