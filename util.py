import os
import numpy as np
import torch


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d" % params)
    return params


def custom_load(models, names, path, strict = True):
    if type(models) is not list:
        models = [models]
    if type(names) is not list:
        names = [names]
    assert len(models) == len(names)

    whole_dict = torch.load(path)

    for i in range(len(models)):
        models[i].load_state_dict(whole_dict[names[i]], strict = strict)

    return whole_dict


def custom_save(path, parts, names):
    if type(parts) is not list:
        parts = [parts]
    if type(names) is not list:
        names = [names]
    assert len(parts) == len(names)

    whole_dict = {}
    for i in range(len(parts)):
        if torch.is_tensor(parts[i]):
            whole_dict.update({names[i]: parts[i]})
        else:
            whole_dict.update({names[i]: parts[i].state_dict()})

    torch.save(whole_dict, path)
