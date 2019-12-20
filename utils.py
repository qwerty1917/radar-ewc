"""utils.py"""

import argparse
from pathlib import Path

import torch
from torch import nn
from visdom import Visdom
import numpy as np
import random
import math
import os
from torch.optim import Optimizer

class One_Hot(nn.Module):
    # from :
    # https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/

    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)

    def forward(self, X_in):
        X_in = X_in.long()
        return self.ones.index_select(0, X_in.data)

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def rm_dir(dir_path, silent=True):
    p = Path(dir_path)
    if (not p.is_file()) and (not p.is_dir()):
        print('It is not path for file nor directory :', p)
        return

    paths = list(p.iterdir())
    if not paths and p.is_dir():
        p.rmdir()
        if not silent:
            print('removed empty dir :', p)
    else:
        for path in paths:
            if path.is_file():
                path.unlink()
                if not silent:
                    print('removed file :', path)
            else:
                rm_dir(path)
        p.rmdir()
        if not silent:
            print('removed empty dir :', p)


def where(cond, x, y):
    """Do same operation as np.where
    code from:
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def make_log_name(args):

    if args.incremental:
        log_name = 'incremental'
    else:
        log_name = 'subject'

    if args.subject_shuffle:
        log_name += '_shuffle'

    log_name += '_m_seed{}_s_seed{}'.format(args.model_seed, args.subject_seed)
    log_name += '_window{}'.format(args.time_window)
    if args.multi:
        log_name += '_multi'
        if args.init_from_prehead:
            log_name += '_preinit'

    if args.fisher_normalize:
        log_name += '_normalized'

    log_name += '_lamb{}'.format(args.lamb)

    if args.continual == 'ewc_online':
        log_name += '_gamma{}'.format(args.gamma)

    elif args.continual == 'si':
        log_name += '_eps{}'.format(args.si_eps)

    elif args.continual == 'ucl':
        log_name += '_alpha{}_ratio{}_lr-rho{}'.format(args.alpha, args.ratio, args.lr_rho)

    elif args.continual == 'gem' or args.continual == 'er':
        log_name += '_n_memories{}'.format(args.n_memories)
        if args.continual == 'gem':
            log_name += '_margin{}'.format(args.memory_strength)

    if args.pretrain or args.load_pretrain:
        log_name += '_pre_{}tasks'.format(args.num_pre_tasks)
        if args.pre_reg_param:
            log_name += '_apply_reg'

    log_name += '_epochs{}'.format(args.epoch)
    log_name += '_{}ep'.format(args.eval_period)

    if args.lr_decay:
        log_name += '_decay'

    return log_name


def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
            print("Failed to create directory!!")


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name: str, port: int):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x]),
                                                 Y=np.array([y]),
                                                 env=self.env,
                                                 opts=dict(
                                                     legend=[split_name],
                                                     title=title_name,
                                                     xlabel='Iteration',
                                                     ylabel=var_name
                                                 ))
        else:
            self.viz.line(X=np.array([x]),
                          Y=np.array([y]),
                          env=self.env,
                          win=self.plots[var_name],
                          name=split_name,
                          update='append')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

class Adam(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, lr_rho=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, param_name=None, lr_scale=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.param_name = param_name
        self.lr_rho = lr_rho
        self.lr_scale = lr_scale
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                n = self.param_name[i]

                if 'rho' in self.param_name[i]:
                    step_size = self.lr_rho * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                #                 p.data.addcdiv_(-step_size, self.lr_scale[n] * exp_avg, denom)
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
