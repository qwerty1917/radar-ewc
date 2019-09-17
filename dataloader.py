from pathlib import Path

import PIL
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import random
import os
from os.path import join
from utils import list_dir, set_seed

def return_data(args):
    # TODO: cnn_datasets return_data
    # train_dset_dir = args.train_dset_dir
    # test_dset_dir = args.test_dset_dir

    if args.subject_shuffle:
        set_seed(0)
        random.seed(args.seed)
    else:
        set_seed(args.seed)

    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    time_window = args.time_window
    trivial_augmentation = bool(args.trivial_augmentation)
    sliding_augmentation = bool(args.sliding_augmentation)

    transform_list = [transforms.Resize((image_size, image_size))]

    if args.channel == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    if trivial_augmentation:
        trivial_transform_list = [
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(1, 1)),
            RandomNoise(mean=0, std=10),
        ]
        transform_list.append(transforms.RandomChoice(trivial_transform_list))

    if sliding_augmentation:
        transform_list.append(RandomTimeWindow(time_window=time_window))
    else:
        transform_list.append(TimeWindow(time_window=time_window))

    transform_list.append(transforms.ToTensor())

    if args.channel == 1:
        transform_list.append(transforms.Normalize([0.5], [0.5]))
    else:
        transform_list.append(transforms.Normalize([0.5] * args.channel, [0.5] * args.channel))
    print(transform_list)
    transform = transforms.Compose(transform_list)

    # if args.channel == 1:
    #     transform = transforms.Compose([
    #         transforms.Resize((image_size, image_size)),
    #         transforms.Grayscale(num_output_channels=1),
    #         TimeWindow(time_window=time_window),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #     ])
    # else:
    #     transform = transforms.Compose([
    #         transforms.Resize((image_size, image_size)),
    #         TimeWindow(time_window=time_window),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5] * args.channel, [0.5] * args.channel),
    #     ])
    """
    train_root = Path(train_dset_dir)
    test_root = Path(test_dset_dir)
    train_kwargs = {'root': train_root, 'transform': transform}
    test_kwargs = {'root': test_root, 'transform': transform}
    dset = ImageFolder

    train_data = dset(**train_kwargs)
    test_data = dset(**test_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_data,
                             batch_size=test_batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True)

    data_loader = dict()
    data_loader['train'] = train_loader
    data_loader['test'] = test_loader
    """

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))

    data_loader = {}
    if args.incremental:
        root = './data/per_acitivity/'
    else:
        root = './data/per_subject/'

    tasks = list_dir(root)
    num_tasks = len(tasks)

    train_imagefolders = []
    test_imagefolders = []

    if args.subject_shuffle:
        random.shuffle(tasks)
        for i, task in enumerate(tasks):

            data_loader['task{}'.format(i)] = {}

            target_subject = join(root, task)

            train_data = ImageFolder(root=target_subject + '/train', transform=transform)
            test_data = ImageFolder(root=target_subject + '/test', transform=transform)

            train_imagefolders.append(train_data)
            test_imagefolders.append(test_data)

    else:
        for i in range(num_tasks):

            data_loader['task{}'.format(i)] = {}

            target_subject = join(root, 'Subject{}'.format(i+1))

            train_data = ImageFolder(root=target_subject + '/train', transform=transform)
            test_data = ImageFolder(root=target_subject + '/test', transform=transform)

            train_imagefolders.append(train_data)
            test_imagefolders.append(test_data)

    if args.continual:
        for i in range(num_tasks):

            train_loader = DataLoader(train_imagefolders[i], batch_size=train_batch_size,
                                      shuffle=True, num_workers=num_workers,
                                      pin_memory=True, drop_last=True, worker_init_fn=_init_fn)
            test_loader = DataLoader(test_imagefolders[i], batch_size=test_batch_size,
                                     shuffle=True, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, worker_init_fn=_init_fn)

            data_loader['task{}'.format(i)]['train'] = train_loader
            data_loader['task{}'.format(i)]['test'] = test_loader
    else:
        num_tasks = 1
        train_data_concat = ConcatDataset(train_imagefolders)
        test_data_concat = ConcatDataset(test_imagefolders)

        train_loader = DataLoader(train_data_concat, batch_size=train_batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True, drop_last=True, worker_init_fn=_init_fn)
        test_loader = DataLoader(test_data_concat, batch_size=test_batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, worker_init_fn=_init_fn)

        data_loader['train'] = train_loader
        data_loader['test'] = test_loader

    return data_loader, num_tasks


class TimeWindow(object):
    def __init__(self, time_window=3):
        self.time_window = time_window
        self.max_time_window = 3

    def __call__(self, img):
        np_img = np.array(img)  # [row, col, ch]
        img_w = img.size[0]
        img_h = img.size[1]
        ch = len(img.getbands())

        trim_width = int((self.time_window/self.max_time_window) * img_w)

        if ch > 1:
            np_trimmed = np_img[:, :trim_width, :]
        else:
            np_trimmed = np_img[:, :trim_width]

        trimmed_img = PIL.Image.fromarray(np_trimmed.astype('uint8'))

        return trimmed_img

    def __repr__(self):
        return self.__class__.__name__ + '(time_window={0})'.format(self.time_window)


class RandomTimeWindow(object):
    def __init__(self, time_window=3):
        self.time_window = time_window
        self.max_time_window = 3

    def __call__(self, img):
        np_img = np.array(img)  # [row, col, ch]
        img_w = img.size[0]
        img_h = img.size[1]
        ch = len(img.getbands())

        trim_width = int((self.time_window/self.max_time_window) * img_w)

        crop_start = random.randint(0, img_w-trim_width)

        if ch > 1:
            np_trimmed = np_img[:, crop_start:crop_start+trim_width, :]
        else:
            np_trimmed = np_img[:, crop_start:crop_start+trim_width]

        trimmed_img = PIL.Image.fromarray(np_trimmed.astype('uint8'))

        return trimmed_img

    def __repr__(self):
        return self.__class__.__name__ + '(time_window={0})'.format(self.time_window)


class RandomNoise(object):
    """Add random noise on image.
    Args:
        mean (float): mean of noise
        std (float): std of noise
    Returns:
        PIL Image: noise added image.
    """

    def __init__(self, mean=0, std=0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to add noise.
        Returns:
            PIL Image: Noise added image.
        """

        # Convert PIL image to numpy.
        np_img = np.array(img)
        img_w = img.size[0]
        img_h = img.size[1]
        ch = len(img.getbands())

        # Add noise.
        if ch > 1:
            noise = np.random.normal(self.mean, self.std, (img_h, img_w, ch))
        else:
            noise = np.random.normal(self.mean, self.std, (img_h, img_w))
        np_noisy = np.clip(np_img + noise, 0, 255)

        # Convert numpy array to PUL image.
        noisy = PIL.Image.fromarray(np_noisy.astype('uint8'))
        return noisy

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
