import torch
import numpy as np
import argparse

from cont_trainer import cont_DCNN
from bayes_cont_trainer import baye_DCNN
from non_cont_trainer import DCNN

from utils import str2bool, set_seed

def main(args):
    torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    set_seed(args.model_seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    if args.cnn_type == 'dcnn':
        if args.continual == 'ucl':
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
            net = baye_DCNN(args)
        elif args.continual == 'none':
            net = cont_DCNN(args)
        else:
            net = cont_DCNN(args)
    else:
        raise ValueError('cnn_type should be one of DCNN,')

    if args.mode == 'train':
        net.train()
    elif args.mode == 'eval':
        net.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DCNN')

    # Mode
    parser.add_argument('--mode', default='train', type=str, help='operation modes: train / eval')

    # Evaluation
    parser.add_argument('--date', default='190801', type=str, help='experiment date')
    parser.add_argument('--eval_dir', default='./evaluation/', type=str, help='evaluation(test) result directory')

    # Optimization
    parser.add_argument('--epoch', default=20, type=int, help='epoch size')
    parser.add_argument('--train_batch_size', default=16, type=int, help='train batch size')
    parser.add_argument('--test_batch_size', default=3, type=int, help='test batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate for the network')
    parser.add_argument('--early_stopping', default=False, type=str2bool, help='early stopping (bool)')
    parser.add_argument('--early_stopping_iter', default=500, type=int, help='training stops when train loss not improved in this iteration')

    # Network
    parser.add_argument('--cnn_type', default='dcnn', type=str, help='CNN types : dcnn,')
    parser.add_argument('--load_ckpt', default=True, type=str2bool, help='load previous checkpoint')
    parser.add_argument('--ckpt_dir', default='cnn_checkpoint', type=str, help='weight directory')
    parser.add_argument('--image_size', default=32, type=int, help='image size')
    parser.add_argument('--model_seed', default=1, type=int, help='pytorch seed')
    parser.add_argument('--subject_seed', default=1, type=int, help='pytorch seed')
    parser.add_argument('--model_dir', default='./trained_models/', type=str, help='save directory for trained models')


    # Dataset
    parser.add_argument('--inter_fold_subject_shuffle', default=False, type=bool, help='subject shuffle inter-folds')
    parser.add_argument('--time_window', default=3, type=float, help='time window seconds')
    parser.add_argument('--num_workers', default=1, type=int, help='num_workers for data loader')
    parser.add_argument('--channel', default=1, type=int, help='input image channel')
    parser.add_argument('--trivial_augmentation', default=False, type=str2bool, help='crop & zoom, bright, noise.')
    parser.add_argument('--sliding_augmentation', default=False, type=str2bool, help='random slice augmentation.')
    parser.add_argument('--incremental', default=False, action='store_true', help='apply class incremental learning')

    # Visualization / Log
    parser.add_argument('--env_name', default='main', type=str, help='experiment name')
    parser.add_argument('--reset_env', default=False, type=str2bool, help='delete log folders')
    parser.add_argument('--visdom', default=True, type=str2bool, help='enable visdom')
    parser.add_argument('--visdom_port', default=8085, type=int, help='visdom port number')
    parser.add_argument('--output_dir', default='cnn_output', type=str, help='inter train result directory')

    # Misc
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--multi_gpu', default=False, type=str2bool, help='enable multi gpu')

    # Continual Learning
    parser.add_argument('--continual', default='', type=str, required=True, help='continual learning method',
                        choices=['ewc', 'hat_ewc', 'ewc_online', 'si', 'l2', 'ucl', 'none'])

    parser.add_argument('--lamb', default=0., type=float, help='regularization strength')
    parser.add_argument('--multi', default=False, type=str2bool, help='apply multi-head approach')
    parser.add_argument('--num-tasks', default=12, type=int, help='number of tasks for continual training')
    parser.add_argument('--subject_shuffle', default=False, type=str2bool, help='shuffle subjects')

    # EWC
    parser.add_argument('--gamma', default=1.0, type=float, help='online ewc gamma')
    parser.add_argument('--fisher_normalize', default=False, type=str2bool, help='normalize fisher')

    # SI
    parser.add_argument('--si_eps', default=0.1, type=float, help='si epsilon')

    # UCL
    parser.add_argument('--alpha', default=0.01, type=float, help='(default=%(default)f)')
    parser.add_argument('--ratio', default=0.5, type=float, help='(default=%(default)f)')
    parser.add_argument('--rho', type=float, default=-2.783, help='initial rho')
    parser.add_argument('--lr_rho', type=float, default=0.001, help='initial lr rho for ucl')
    args = parser.parse_args()

    main(args)
