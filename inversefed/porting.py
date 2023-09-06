import torch
import torch.nn as nn
import yaml
import pickle
from .AE_model.models import *

def load_decoder_stylegan2(config, device, dataset='FFHQ', untrained=True, ada=False, cond=False):
    
    if ada:
        if cond:
            if untrained:
                path = f'inversefed/genmodels/stylegan2/{dataset}_untrained.pkl'
            else:
                path = f'inversefed/genmodels/stylegan2/{dataset}.pkl'
        else:
            if untrained:
                path = f'inversefed/genmodels/stylegan2/{dataset}_uc_untrained.pkl'
            else:
                path = f'inversefed/genmodels/stylegan2/{dataset}_uc.pkl'

    else:
        if dataset.startswith('FF'):
            path = f'inversefed/genmodels/stylegan2/Gs.pth'
        elif dataset.startswith('I'):
            path = f'inversefed/genmodels/stylegan2/imagenet.pth'
        
            
                
        
    if ada:
        from .genmodels.stylegan2_ada_pytorch import legacy
        with open(path, 'rb') as f:
            G = legacy.load_network_pkl(f)['G_ema']
            # G.random_noise()
            G_mapping = G.mapping
            G_synthesis = G.synthesis
    else:
        from .genmodels import stylegan2
        G = stylegan2.models.load(path)
        G.random_noise()
        G_mapping = G.G_mapping
        G_synthesis = G.G_synthesis

    

    
    G.requires_grad_(True)
    G_mapping.requires_grad_(True)
    G_synthesis.requires_grad_(True)

    if torch.cuda.device_count() > 1:
        # G = nn.DataParallel(G)
        G_mapping = nn.DataParallel(G_mapping)
        G_synthesis = nn.DataParallel(G_synthesis)

    G.requires_grad_(False)
    G_mapping.requires_grad_(False)
    G_synthesis.requires_grad_(False)

        

    return G, G_mapping, G_synthesis



def load_decoder_stylegan2_ada(config, device, dataset='I128'):
    from .genmodels.stylegan2_ada_pytorch import legacy
    network_pkl = ''
    if dataset.startswith('I'):
        network_pkl = f'inversefed/genmodels/stylegan2_ada_pytorch/I64.pkl'

    elif dataset == 'C10':
        network_pkl = 'inversefed/genmodels/stylegan2_ada_pytorch/cifar10u-cifar-ada-best-fid.pkl'
        # with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig11b-cifar10/cifar10u-cifar-ada-best-fid.pkl') as f:
        #     G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(True).to(device) # type: ignore
    
    with open(network_pkl, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(True).to(device) 
        # data = legacy.load_network_pkl(f)
        # print(data.keys())
    
    return G


def load_decoder_stylegan2_untrained(config, device, dataset='I128'):
    from .genmodels.stylegan2_ada_pytorch import legacy

    if dataset == 'I128' or dataset == 'I64' or dataset == 'I32':
        network_pkl = f'/home/jjw/projects/inverting-quantized-gradient/models/GANs/stylegan2_ada_pytorch/output/00010-ImageNet128x128-auto2/network-snapshot-025000.pkl'
        print('Loading networks from "%s"...' % network_pkl)
        G = None
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(True).to(device) # type: ignore

    elif dataset == 'C10':
        with open('models/GANs/stylegan2_ada_pytorch/cifar10u-untrained.pkl', 'rb') as f:
            G = pickle.load(f).requires_grad_(True).to(device) 
    
    return G


def load_decoder_dcgan(config, device, dataset='C10'):
    from inversefed.genmodels.cifar10_dcgan.dcgan import Generator as DCGAN

    G = DCGAN(ngpu=1).eval()
    G.load_state_dict(torch.load('inversefed/genmodels/cifar10_dcgan/weights/DCGAN.pth'))
    G.to(device)

    return G

def load_decoder_dcgan_untrained(config, device, dataset='C10'):
    if dataset == 'PERM':
        from inversefed.genmodels.deep_image_prior.generator import Generator64 as DCGAN64
        G = DCGAN64(ngpu=1)
    else:
        from inversefed.genmodels.cifar10_dcgan.dcgan import Generator as DCGAN

        G = DCGAN(ngpu=1).eval()
    G.to(device)

    return G

import os
def load_dict(resume_path, model):
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        if('state_dict' not in checkpoint):
            model.load_state_dict(checkpoint)
        else:
            model_dict = model.state_dict()
            model_dict.update(checkpoint['state_dict'])
            model.load_state_dict(model_dict)
            # delete to release more space
            del checkpoint
    else:
        sys.exit("=> No checkpoint found at '{}'".format(resume_path))
    return model

def build_autoencoder(dataset='ImageNet'):
    # parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    # parser.add_argument('--arch', default='vgg19', type=str, 
    #                     help='backbone architechture')
    # parser.add_argument('--resume', type=str)
    # parser.add_argument('--val_list', type=str)           
    
    # args = parser.parse_args()
    if(dataset=='ImageNet'):
        arch = 'vgg19'
    elif(dataset=='CIFAR10'):
        arch = "ae_cifar10"
    elif(dataset=='I64'):
        arch = "ae_I64"
    
    model = builder.BuildAutoEncoder(arch)
    model = load_dict('./inversefed/AE_model/{}/ae.pth'.format(dataset),model)
    model.eval()
    
    return model