"""Run reconstruction in a terminal prompt.
Optional arguments can be found in inversefed/options.py

This CLI can recover the baseline experiments.
"""

import torch
import torchvision
import torch.nn as nn

import numpy as np

import inversefed
torch.backends.cudnn.benchmark = inversefed.consts.BENCHMARK

from collections import defaultdict
import datetime
import time
import os
import json
import hashlib
import csv
import copy
import pickle
import random
from defense import *

nclass_dict = {'I32': 1000, 'I64': 1000, 'I128': 1000, 
               'CIFAR10': 10, 'CIFAR100': 100, 'CA': 8, 'ImageNet':1000,
               'FFHQ': 10, 'FFHQ64': 10, 'FFHQ128': 10,
               }
# Parse input arguments
parser = inversefed.options()

args = parser.parse_args()
if args.target_id is None:
    args.target_id = 0
args.save_image = True
args.signed = not args.unsigned


# Parse training strategy
defs = inversefed.training_strategy('conservative')
defs.epochs = args.epochs

if __name__ == "__main__":
    # Choose GPU device and print status information:
    setup = inversefed.utils.system_startup(args)
    start_time = time.time()

    # Prepare for training

    # Get data:
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(args.dataset, defs, data_path=args.data_path, shuffle=True)

    model, model_seed = inversefed.construct_model(args.model, num_classes=nclass_dict[args.dataset], num_channels=3, seed=0)
    
    if args.trained_model:
        epochs = args.epochs
        file = f'{args.model}_{epochs}.pth'
        try:
            model.load_state_dict(torch.load(f'models/{file}'))
        except FileNotFoundError:
            inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup)
        torch.save(model.state_dict(), f'models/{file}')
    
    if args.dataset.startswith('FFHQ'):
        dm = torch.as_tensor(getattr(inversefed.consts, f'cifar10_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'cifar10_std'), **setup)[:, None, None]
    else:
        dm = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_std'), **setup)[:, None, None]
    model = nn.DataParallel(model)
    model.to(**setup)
    model.eval()

    if args.optim == 'gias':
        config = dict(signed=args.signed,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      anomaly_loss=args.ano,
                      bn_stat=args.bn_stat,
                      image_norm=args.image_norm,
                      z_norm=args.z_norm,
                      group_lazy=args.group_lazy,
                      init=args.init,
                      lr_decay=True,
                      dataset=args.dataset,
                      
                      generative_model=args.generative_model,
                      gen_dataset=args.gen_dataset,
                      giml=args.giml,
                      gias=args.gias,
                      gias_lr=args.gias_lr,
                      gias_iterations=args.gias_iterations,
                      )
    elif args.optim == 'yin':
        config = dict(signed=args.signed,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      anomaly_loss=args.ano,
                      bn_stat=args.bn_stat,
                      image_norm=args.image_norm,
                      z_norm=args.z_norm,
                      group_lazy=args.group_lazy,
                      init=args.init,
                      lr_decay=True,
                      dataset=args.dataset,
                      
                      generative_model='',
                      gen_dataset='',
                      giml=False,
                      gias=False,
                      gias_lr=0.0,
                      gias_iterations=args.gias_iterations,
                      )
    elif args.optim == 'gen':
        config = dict(signed=args.signed,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      anomaly_loss=args.ano,
                      bn_stat=args.bn_stat,
                      image_norm=args.image_norm,
                      z_norm=args.z_norm,
                      group_lazy=args.group_lazy,
                      init=args.init,
                      lr_decay=True,
                      dataset=args.dataset,
                      
                      generative_model=args.generative_model,
                      gen_dataset=args.gen_dataset,
                      giml=False,
                      gias=False,
                      gias_lr=0.0,
                      gias_iterations=0,
                      )
    elif args.optim == 'geiping':
        config = dict(signed=args.signed,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      bn_stat=-1.0,
                      image_norm=-1.0,
                      z_norm=-1.0,
                      group_lazy=-1.0,
                      init=args.init,
                      lr_decay=True,
                      dataset=args.dataset,
                      
                      generative_model='',
                      gen_dataset='',
                      giml=False,
                      gias=False,
                      gias_lr=0.0,
                      gias_iterations=0,
                      )      
    elif args.optim == 'zhu':
        config = dict(signed=False,
                      cost_fn='l2',
                      indices='def',
                      weights='equal',
                      lr=args.lr if args.lr is not None else 1.0,
                      optim='LBFGS',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      init=args.init,
                      lr_decay=False,
                      )
    elif args.optim == 'ours':
        config = dict(signed=args.signed,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      anomaly_loss=args.ano,
                      bn_stat=args.bn_stat,
                      image_norm=-1.0,
                      z_norm=-1.0,
                      group_lazy=-1.0,
                      init=args.init,
                      lr_decay=True,
                      dataset=args.dataset,
                      
                      generative_model='',
                      gen_dataset='',
                      giml=False,
                      gias=False,
                      gias_lr=0.0,
                      gias_iterations=0,
                      )   
    
    # psnr list
    psnrs = []

    # hash configuration

    config_comp = config.copy()
    config_comp['optim'] = args.optim
    config_comp['dataset'] = args.dataset
    config_comp['model'] = args.model
    config_comp['trained'] = args.trained_model
    config_comp['num_exp'] = args.num_exp
    config_comp['num_images'] = args.num_images
    config_comp['bn_stat'] = args.bn_stat
    config_comp['image_norm'] = args.image_norm
    config_comp['z_norm'] = args.z_norm
    config_comp['group_lazy'] = args.group_lazy
    config_comp['checkpoint_path'] = args.checkpoint_path
    config_comp['accumulation'] = args.accumulation
    config_comp['batch_size'] = args.batch_size
    config_comp['local_lr'] = args.trained_model
    config_hash = hashlib.md5(json.dumps(config_comp, sort_keys=True).encode()).hexdigest()

    print(config_comp)
    print(config_hash)
    # exit()

    os.makedirs(args.table_path, exist_ok=True)
    os.makedirs(os.path.join(args.table_path, f'{config_hash}'), exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(os.path.join(args.result_path, f'{config_hash}'), exist_ok=True)

    G = None
    if args.checkpoint_path:
        with open(args.checkpoint_path, 'rb') as f:
            G, _ = pickle.load(f)
            G = G.requires_grad_(True).to(setup['device'])

    for i in range(args.num_exp):
        target_id = args.target_id + args.num_exp *i
        tid_list = []
        target_id_ = random.randint(0,len(trainloader.dataset))
        if args.num_images == 1:
            #ground_truth, labels = validloader.dataset[target_id]
            ground_truth, labels = trainloader.dataset[target_id_]
            ground_truth, labels = ground_truth.unsqueeze(0).to(**setup), torch.as_tensor((labels,), device=setup['device'])
            target_id_ = target_id
            print("loaded img %d" % (target_id_ - 1))
            tid_list.append(target_id_ - 1)
        else:
            ground_truth, labels = [], []
            while len(labels) < args.num_images:
                target_id_ = random.randint(0,len(trainloader.dataset))
                #img, label = validloader.dataset[target_id_]
                img, label = trainloader.dataset[target_id_]
                #target_id_ += 1
                if (label not in labels):
                    print("loaded img %d" % (target_id_ - 1))
                    labels.append(torch.as_tensor((label,), device=setup['device']))
                    ground_truth.append(img.to(**setup))
                    tid_list.append(target_id_ - 1)

            ground_truth = torch.stack(ground_truth)
            labels = torch.cat(labels)
        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])
        # print(labels)

        # Run reconstruction
        if args.bn_stat > 0:
            bn_layers = []
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_layers.append(inversefed.BNStatisticsHook(module))


        if args.accumulation == 0:
            target_loss, _, _ = loss_fn(model(ground_truth), labels)
            input_gradient = torch.autograd.grad(target_loss, model.parameters())
            input_gradient = [grad.detach() for grad in input_gradient]

            if(args.defense != None):
                if args.sparse_ratio > 0:
                    input_gradient = gradient_compression(input_gradient, args.sparse_ratio)
                elif args.noise_std > 0:
                    input_gradient = additive_noise(input_gradient, args.noise_std)
                elif args.clip_bound > 0:
                    input_gradient = gradient_clipping(input_gradient, args.clip_bound)
            
            bn_prior = []
            if args.bn_stat > 0:
                for idx, mod in enumerate(bn_layers):
                    mean_var = mod.mean_var[0].detach(), mod.mean_var[1].detach()
                    bn_prior.append(mean_var)
            # with open(f'exp_{i}_bn_prior.pkl', 'wb') as f:
            #     pickle.dump(bn_prior, f)
            rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images, bn_prior=bn_prior, G=G, dataset=args.dataset)

            if G is None:
                G = rec_machine.G

            output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape, dryrun=args.dryrun)

        # Compute stats and save to a table:
        output_den = torch.clamp(output * ds + dm, 0, 1)
        ground_truth_den = torch.clamp(ground_truth * ds + dm, 0, 1)
        feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
        test_mse = (output_den - ground_truth_den).pow(2).mean().item()
        test_psnr = inversefed.metrics.psnr(output_den, ground_truth_den, factor=1)
        test_pips = inversefed.metrics.lpips_loss(output_den.cuda(),ground_truth_den.cuda())
        test_ssim = inversefed.metrics.ssim_permute(ground_truth_den, output_den)[0]
        print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} | LPIPS: {test_pips:2.4e} | SSIM:{test_ssim:2.2f}")

        inversefed.utils.save_to_table(os.path.join(args.table_path, f'{config_hash}'), name=f'mul_exp_{args.name}', dryrun=args.dryrun,

                                       config_hash=config_hash,
                                       model=args.model,
                                       dataset=args.dataset,
                                       trained=args.trained_model,
                                       restarts=args.restarts,
                                       OPTIM=args.optim,
                                       cost_fn=args.cost_fn,
                                       indices=args.indices,
                                       weights=args.weights,
                                       init=args.init,
                                       tv=args.tv,

                                       rec_loss=stats["opt"],
                                       psnr=test_psnr,
                                       lpips=test_pips,
                                       test_mse=test_mse,
                                       feat_mse=feat_mse,
                                       ssim=test_ssim,

                                       target_id=args.target_id,
                                       seed=model_seed,
                                       epochs=defs.epochs,
                                    #    val_acc=training_stats["valid_" + name][-1],
                                       )


        # Save the resulting image
        if args.save_image and not args.dryrun:
            # if args.giml or args.gias:

            #     latent_img = rec_machine.gen_dummy_data(rec_machine.G_synthesis.to(setup['device']), rec_machine.generative_model_name, rec_machine.dummy_z.to(setup['device']))
            #     latent_denormalized = torch.clamp(latent_img * ds + dm, 0, 1)    

            #     latent_psnr = inversefed.metrics.psnr(latent_denormalized, ground_truth_den, factor=1)
            #     print(f"Latent PSNR: {latent_psnr:4.2f} |")

            output_denormalized = torch.clamp(output * ds + dm, 0, 1)
            for j in range(args.num_images):
                # if args.giml or args.gias:
                #     torchvision.utils.save_image(latent_denormalized[j:j + 1, ...], os.path.join(args.result_path, f'{config_hash}', f'{tid_list[j]}_latent.png'))
                torchvision.utils.save_image(output_denormalized[j:j + 1, ...], os.path.join(args.result_path, f'{config_hash}', f'{tid_list[j]}.png'))
                torchvision.utils.save_image(ground_truth_den[j:j + 1, ...], os.path.join(args.result_path, f'{config_hash}', f'{tid_list[j]}_gt.png'))

        # Save psnr values
        psnrs.append(test_psnr)
        inversefed.utils.save_to_table(os.path.join(args.table_path, f'{config_hash}'), name='psnrs', dryrun=args.dryrun, target_id=target_id, psnr=test_psnr,pips=test_pips)

        # Update target id
        target_id = target_id_


    # psnr statistics
    psnrs = np.nan_to_num(np.array(psnrs))
    psnr_mean = psnrs.mean()
    psnr_std = np.std(psnrs)
    psnr_max = psnrs.max()
    psnr_min = psnrs.min()
    psnr_median = np.median(psnrs)
    timing = datetime.timedelta(seconds=time.time() - start_time)
    inversefed.utils.save_to_table(os.path.join(args.table_path, f'{config_hash}'), name='psnr_stats', dryrun=args.dryrun,
                                   number_of_samples=len(psnrs),
                                   timing=str(timing),
                                   mean=psnr_mean,
                                   std=psnr_std,
                                   max=psnr_max,
                                   min=psnr_min,
                                   median=psnr_median)

    config_exists = False
    if os.path.isfile(os.path.join(args.table_path, 'table_configs.csv')):
        with open(os.path.join(args.table_path, 'table_configs.csv')) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                if row[-1] == config_hash:
                    config_exists = True
                    break

    if not config_exists:
        inversefed.utils.save_to_table(args.table_path, name='configs', dryrun=args.dryrun,
                                       config_hash=config_hash,
                                       **config_comp,
                                       number_of_samples=len(psnrs),
                                       timing=str(timing),
                                       mean=psnr_mean,
                                       std=psnr_std,
                                       max=psnr_max,
                                       min=psnr_min,
                                       median=psnr_median)

    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')
