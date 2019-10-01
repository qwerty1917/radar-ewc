from pathlib import Path

import PIL
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import random
import os
from os.path import join
from utils import list_dir, set_seed


def make_transform(args):
    set_seed(args.seed)

    image_size = args.image_size
    time_window = args.time_window
    darker_threshold = args.darker_threshold
    trivial_augmentation = bool(args.trivial_augmentation)
    sliding_augmentation = bool(args.sliding_augmentation)

    transform_list = [transforms.Resize((image_size, image_size)), RetouchDarker(darker_threshold)]

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
    transform = transforms.Compose(transform_list)

    return transform


def return_data(args, class_range=None):
    # train_dset_dir = args.train_dset_dir
    # test_dset_dir = args.test_dset_dir

    set_seed(args.seed)

    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    num_workers = args.num_workers

    transform = make_transform(args)

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))

    data_loader = {}
    root = './data/per_subject/'

    num_tasks = len(list_dir(root))

    if class_range is None:
        class_range = range(7)

    train_imagefolders = []
    test_imagefolders = []
    eval_imagefolders = []
    for i in range(num_tasks):

        data_loader['task{}'.format(i)] = {}

        target_subject = join(root, 'Subject{}'.format(i+1))

        train_data = IcarlDataset(root=target_subject + '/train', classes=class_range, transform=transform)
        test_data = IcarlDataset(root=target_subject + '/test', classes=range(list(class_range)[-1]+1), transform=transform)
        eval_data = IcarlDataset(root=target_subject + '/test', classes=class_range, transform=transform)

        train_imagefolders.append(train_data)
        test_imagefolders.append(test_data)
        eval_imagefolders.append(eval_data)

    if args.continual:
        for i in range(num_tasks):
            # data loader가 cnn model 학습 이전에 이미 생성완료되어 선언되므로 여기서 replay를 하는건 불가능.
            if args.task_upper_bound:
                train_dataset = ConcatDataset(train_imagefolders[:i + 1])
                test_dataset = ConcatDataset(test_imagefolders[:i + 1])
            else:
                train_dataset = train_imagefolders[i]
                test_dataset = test_imagefolders[i]
            train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                      shuffle=True, num_workers=num_workers,
                                      pin_memory=True, drop_last=True, worker_init_fn=_init_fn)
            test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                                     shuffle=True, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, worker_init_fn=_init_fn)

            data_loader['task{}'.format(i)]['train'] = train_loader
            data_loader['task{}'.format(i)]['test'] = test_loader
    else:
        num_tasks = 1
        train_data_concat = train_imagefolders[0]
        for i in range(1, len(train_imagefolders)):
            train_data_concat.append(train_imagefolders[i].samples, train_imagefolders[i].targets)

        test_data_concat = test_imagefolders[0]
        for i in range(1, len(test_imagefolders)):
            test_data_concat.append(test_imagefolders[i].samples, test_imagefolders[i].targets)

        eval_data_concat = eval_imagefolders[0]
        for i in range(1, len(eval_imagefolders)):
            eval_data_concat.append(eval_imagefolders[i].samples, eval_imagefolders[i].targets)

        train_loader = DataLoader(train_data_concat, batch_size=train_batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True, drop_last=True, worker_init_fn=_init_fn)
        test_loader = DataLoader(test_data_concat, batch_size=test_batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, worker_init_fn=_init_fn)
        eval_loader = DataLoader(eval_data_concat, batch_size=test_batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, worker_init_fn=_init_fn)


        data_loader['train'] = train_loader
        data_loader['test'] = test_loader
        data_loader['eval'] = eval_loader

    return data_loader, num_tasks, transform


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


class RetouchDarker(object):
    """Cutout pixels that those of value is under threshold as 0.
    Args:
        threshold (int): cutout threshold value. Shoud be int between 0~255.
    Returns:
        PIL Image: Retouched images.
    """

    def __init__(self, threshold: int=10):
        self.threshold = threshold

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image retouch
        Returns:
            PIL Image: Retouched images.
        """
        np_img = np.array(img)
        retouched = np_img.copy()
        dark_idx = retouched < self.threshold
        retouched[dark_idx] = 0
        # Convert numpy array to PUL image.
        retouched = PIL.Image.fromarray(retouched.astype('uint8'))
        return retouched

    def __repr__(self):
        return self.__class__.__name__ + '(threshold={0})'.format(self.threshold)


class IcarlDataset(ImageFolder):
    def __init__(self, root,
                 classes=range(7),
                 transform=None,
                 target_transform=None):
        super(IcarlDataset, self).__init__(root,
                                           transform=transform,
                                           target_transform=target_transform)

        task_data = []
        task_labels = []

        for i in range(self.__len__()):
            if self.targets[i] in classes:
                task_data.append(self.samples[i])
                task_labels.append(self.targets[i])

        self.samples = np.array(task_data)
        self.targets = task_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, sample, int(target), path

    def get_image_class(self, label):
        samples = self.samples[np.array(self.targets) == label]
        targets = [label] * len(samples)
        single_class_dataset = ImageFolder(root='',
                                           transform=self.transform,
                                           target_transform=self.target_transform)
        single_class_dataset.samples = samples
        single_class_dataset.targets = targets
        return single_class_dataset

    def append(self, images, labels):
        self.samples = np.concatenate((self.samples, images), axis=0)
        self.targets = self.targets + labels
