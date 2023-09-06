"""Parser options."""

import argparse

def options():
    """Construct the central argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')

    # Central:
    parser.add_argument('--model', default='ResNet18', type=str, help='Vision model.')
    parser.add_argument('--dataset', default='CIFAR10', type=str)
 

    parser.add_argument('--trained_model', action='store_true', default=False, help='Use a trained model.')
    parser.add_argument('--epochs', default=120, type=int, help='If using a trained model, how many epochs was it trained?')

    parser.add_argument('--num_images', default=4, type=int, help='How many images should be recovered from the given gradient.')
    parser.add_argument('--target_id', default=0, type=int, help='Cifar validation image used for reconstruction.')

    # Rec. parameters
    parser.add_argument('--optim', default='ours', type=str, help='Use our reconstruction method or the DLG method.')

    parser.add_argument('--restarts', default=1, type=int, help='How many restarts to run.')
    parser.add_argument('--cost_fn', default='sim', type=str, help='Choice of cost function.')
    parser.add_argument('--indices', default='def', type=str, help='Choice of indices from the parameter list.')
    parser.add_argument('--weights', default='equal', type=str, help='Weigh the parameter list differently.')

    parser.add_argument('--optimizer', default='adam', type=str, help='Weigh the parameter list differently.')
    parser.add_argument('--signed', action='store_false', help='Do not used signed gradients.')

    parser.add_argument('--init', default='randn', type=str, help='Choice of image initialization.')


    # Files and folders:
    parser.add_argument('--save_image', action='store_true', help='Save the output to a file.')

    parser.add_argument('--image_path', default='images/', type=str)
    parser.add_argument('--model_path', default='models/', type=str)
    parser.add_argument('--table_path', default='tables/', type=str)
    parser.add_argument('--result_path', default='results/', type=str)
    parser.add_argument('--data_path', default='~/data', type=str)

    # Debugging:
    parser.add_argument('--name', default='iv', type=str, help='Name tag for the result table and model.')
    parser.add_argument('--dryrun', action='store_true', help='Run everything for just one step to test functionality.')

    # Ablation Study
    parser.add_argument('--tv', default=0, type=float, help='Weight of TV penalty.')
    parser.add_argument('--bn_stat', default=0, type=float, help='Weight of BN statistics.')
    parser.add_argument('--group_lazy', default=0, type=float, help='Weight of group (lazy) regularizer.')
    parser.add_argument('--image_norm', default=0, type=float, help='Weight of image l2 norm')
    parser.add_argument('--z_norm', default=0, type=float, help='Weight of image l2 norm')
    parser.add_argument('--ano', default=0, type=float, help='Anomaly Score loss')
    
    # for generative model
    parser.add_argument('--generative_model', default='', type=str, help='XXX')
    parser.add_argument('--gen_dataset', default='', type=str, help='XXX')
    parser.add_argument('--giml', action='store_true', help='XXX')
    parser.add_argument('--gias', action='store_true', help='XXX')
    parser.add_argument('--lr', default=1e-1, type=float, help='XXX')
    parser.add_argument('--gias_lr', default=1e-2, type=float, help='XXX')

    # supplementary
    parser.add_argument('--accumulation', default=0, type=int, help='Accumulation 0 is rec. from gradient, accumulation > 0 is reconstruction from fed. averaging.')

    parser.add_argument('--unsigned', action='store_true', help='Use signed gradient descent')
    parser.add_argument('--num_exp', default=1, type=int, help='Number of consecutive experiments')
    parser.add_argument('--max_iterations', default=6000, type=int, help='Maximum number of iterations for reconstruction.')
    parser.add_argument('--gias_iterations', default=0, type=int, help='Maximum number of gias iterations for reconstruction.')
    parser.add_argument('--seed', default=1234, type=float, help='random seed')

    parser.add_argument('--batch_size', default=4, type=int, help='Number of batch size for federated averaging')
    parser.add_argument('--local_lr', default=1e-4, type=float, help='Local learning rate for federated averaging')

    parser.add_argument('--checkpoint_path', default='', type=str, help='Local learning rate for federated averaging')

    # defense
    parser.add_argument('--defense', action='store_true', help='defense scheme on gradients')
    parser.add_argument('--sparse_ratio', default=0, type=float, help='Ratio of gradient sparsity')
    parser.add_argument('--noise_std', default=0, type=float, help='The STD of Gaussian Noise for DP')
    parser.add_argument('--clip_bound',default=0, type=float, help='The Clipping Bound for Gradient Clipping')

    return parser
