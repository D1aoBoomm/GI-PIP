import torch.nn as nn
import torch.nn.parallel as parallel

from . import vgg, resnet, ae_cifar10

def BuildAutoEncoder(arch):
    parallel = 0
    batch_size = 1
    workers = 0

    if arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        configs = vgg.get_configs(arch)
        model = vgg.VGGAutoEncoder(configs)

    elif arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        configs, bottleneck = resnet.get_configs(arch)
        model = resnet.ResNetAutoEncoder(configs, bottleneck)
        
    elif arch == 'ae_cifar10':
        model = ae_cifar10.Autoencoder()
        return model.cuda()
    
    elif arch == 'ae_I64':
        model = ae_cifar10.Autoencoder()
        return model.cuda()
    
    else:
        return None
    
    model = nn.DataParallel(model).cuda()

    return model