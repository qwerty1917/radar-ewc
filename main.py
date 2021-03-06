import torch
import numpy as np
import argparse

from cont_trainer import DCNN

from utils import str2bool, set_seed

def main(args):
    torch.backends.cudnn.enabled = True
    # torch.multiprocessing.set_start_method("spawn")
    # torch.backends.cudnn.benchmark = True

    seed = args.seed
    set_seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    if args.cnn_type == 'dcnn':
        net = DCNN(args)
    else:
        raise ValueError('cnn_type should be one of DCNN,')

    if args.mode == 'train':
        net.train()
    elif args.mode == 'eval':
        net.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DCNN')

    # TODO: Hyeongmin park: cont_trainer.py, gan_trainer.py 에 새로 추가된 args 확인하고 여기에 추가

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
    parser.add_argument('--load_ckpt', default=False, type=str2bool, help='load previous checkpoint')
    parser.add_argument('--ckpt_dir', default='cnn_checkpoint', type=str, help='weight directory')
    parser.add_argument('--image_size', default=32, type=int, help='image size')
    parser.add_argument('--seed', default=1, type=int, help='pytorch seed')

    # Dataset
    parser.add_argument('--inter_fold_subject_shuffle', default=False, type=bool, help='subject shuffle inter-folds')
    parser.add_argument('--time_window', default=3, type=float, help='time window seconds')
    parser.add_argument('--num_workers', default=1, type=int, help='num_workers for data loader')
    parser.add_argument('--channel', default=1, type=int, help='input image channel')
    parser.add_argument('--trivial_augmentation', default=False, type=str2bool, help='crop & zoom, bright, noise.')
    parser.add_argument('--sliding_augmentation', default=False, type=str2bool, help='random slice augmentation.')
    parser.add_argument('--incremental', default=False, action='store_true', help='apply class incremental learning')
    parser.add_argument('--darker_threshold', default=0, type=int, help='RetouchDarker threshold')

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
    parser.add_argument('--continual', default=True, type=str2bool, help='continual learning')
    parser.add_argument('--task_upper_bound', default=False, type=str2bool, help='task trained with formal task data. must set with continual.')
    parser.add_argument('--lamb', default=0., type=float, help='regularization strength')

    # EWC
    parser.add_argument('--ewc', default=False, type=str2bool, help='Apply ewc constraint')
    parser.add_argument('--hat_ewc', default=False, type=str2bool, help='Apply hat ewc constraint')
    parser.add_argument('--online', default=False, type=str2bool, help='Apply online EWC')
    parser.add_argument('--gamma', default=1.0, type=float, help='online ewc gamma')

    # SI
    parser.add_argument('--si', default=False, type=str2bool, help='Apply si constraint')
    parser.add_argument('--si_eps', default=0.1, type=float, help='si epsilon')

    # l2
    parser.add_argument('--l2', default=False, type=str2bool, help='Apply l2 constraint')

    # Generative replay
    parser.add_argument('--gr', default=False, type=str2bool, help='Apply Generative replay')
    parser.add_argument('--replay_r', default=0.5, type=float, help='real sample ratio')

    parser.add_argument('--gan_D_lr', default=1e-4, type=float, help='lr for Discriminator')
    parser.add_argument('--gan_G_lr', default=1e-4, type=float, help='lr for Generator')
    parser.add_argument('--gan_d_residual', default=False, type=str2bool, help='GAN D residual connection')
    parser.add_argument('--gan_g_residual', default=False, type=str2bool, help='GAN G residual connection')
    parser.add_argument('--gan_d_iters', default=5, type=int, help='update count of D while update G once')
    parser.add_argument('--gan_g_iters', default=10000, type=int, help='total G update count')
    parser.add_argument('--gan_gp_lambda', default=10, type=float, help='WGAN-GP gradient penalty lambda')
    parser.add_argument('--gan_sample_num', default=100, type=int, help='GAN sample from visdom number')
    parser.add_argument('--gan_multi_gpu', default=False, type=str2bool, help='GAN multi GPU')

    args = parser.parse_args()

    main(args)
