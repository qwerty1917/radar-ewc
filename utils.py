"""utils.py"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from visdom import Visdom
import numpy as np
import random

import os


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
    log_name = args.date
    if args.incremental:
        log_name += '_incremental'
    else:
        log_name += '_subject'

    log_name += '_seed{}'.format(args.seed)
    log_name += '_window{}'.format(args.time_window)

    if args.ewc:
        log_name += '_ewc'
        log_name += '_lamb{}'.format(args.lamb)
    elif args.hat_ewc:
        log_name += '_hatewc'
        log_name += '_lamb{}'.format(args.lamb)
    elif args.l2:
        log_name += '_l2'
        log_name += '_lamb{}'.format(args.lamb)
    elif args.si:
        log_name += '_si'
        log_name += '_lamb{}_eps{}'.format(args.lamb, args.si_eps)
    elif args.gr:
        log_name += '_gr'
        log_name += '_replay_r{}_gp_lamb{}'.format(args.replay_r, args.gan_gp_lambda)
    else:
        log_name += '_fine'

    log_name += '_epochs{}'.format(args.epoch)

    return log_name


def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
            print("Failed to create directory!!")


class VisdomPlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name: str, port: int):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.line_plots = {}
        self.image_frames = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.line_plots:
            self.line_plots[var_name] = self.viz.line(X=np.array([x]),
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
                          win=self.line_plots[var_name],
                          name=split_name,
                          update='append')

    def draw(self, caption, images):
        if caption not in self.image_frames:
            self.image_frames[caption] = self.viz.images(images, nrow=10, padding=2, env=self.env, opts={"caption": caption, "title": caption})

        else:
            self.viz.images(images, nrow=10, padding=2, env=self.env, opts={"caption": caption, "title": caption}, win=self.image_frames[caption])


class VisdomImagesPlotter(object):
    """Show images to Visdom"""
    def __init__(self, env_name: str, port: int):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.frames = {}

    def draw(self, caption, images):
        if caption not in self.frames:
            self.frames[caption] = self.viz.images(images, nrow=10, padding=2, opts={"caption": caption})

        else:
            self.viz.images(images, nrow=10, padding=2, opts={"caption": caption}, win=self.frames[caption])


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ScholarDataset(Dataset):
    def __init__(self, image_x, label_y, transform=None):
        self.image_x = image_x
        self.label_y = label_y
        self.transform = transform

    def __len__(self):
        return len(self.image_x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.image_x[idx], self.label_y[idx])
        return sample


def make_combined_dataloader(real_data_loader: DataLoader, replay_x, replay_y, transform):
    task_dataset = real_data_loader.dataset
    replay_dataset = ScholarDataset(replay_x.cpu().detach(), replay_y.cpu().numpy(), transform=transform)
    merged_dataset = ConcatDataset([task_dataset, replay_dataset])
    merged_dataloader = DataLoader(merged_dataset,
                                   batch_size=real_data_loader.batch_size,
                                   shuffle=True,
                                   num_workers=real_data_loader.num_workers,
                                   pin_memory=True,
                                   drop_last=True,
                                   worker_init_fn=real_data_loader.worker_init_fn)

    return merged_dataloader


def make_replay_dataloader(real_data_loader: DataLoader, replay_x, replay_y, transform):
    replay_dataset = ScholarDataset(replay_x.cpu().detach(), replay_y.cpu().numpy(), transform=transform)
    replay_dataloader = DataLoader(replay_dataset,
                                   batch_size=real_data_loader.batch_size,
                                   shuffle=True,
                                   num_workers=real_data_loader.num_workers,
                                   pin_memory=True,
                                   drop_last=True,
                                   worker_init_fn=real_data_loader.worker_init_fn)

    return replay_dataloader
