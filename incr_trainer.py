import csv
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils
from dataloader import return_data
from gan_trainer import WGAN
from models.cnn import Dcnn
from models.icarl import Icarl
from utils import cuda, make_log_name, check_log_dir, VisdomPlotter, VisdomImagesPlotter, set_seed


## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)


class IcarlTrainer(object):
    def __init__(self, args):
        self.args = args

        set_seed(self.args.seed)

        # Evaluation
        # self.eval_dir = Path(args.eval_dir).joinpath(args.env_name)
        self.eval_dir = args.eval_dir
        check_log_dir(self.eval_dir)
        self.log_name = make_log_name(args)
        self.eval_file_path = self.eval_dir + self.log_name + '.txt'

        # Misc
        self.cuda = args.cuda and torch.cuda.is_available()
        self.multi_gpu = args.multi_gpu

        # Visualization
        self.date = args.date
        self.reset_env = args.reset_env
        self.env_name = args.env_name
        self.visdom = args.visdom
        self.visdom_port = args.visdom_port
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        self.output_dir = Path(args.output_dir).joinpath(args.env_name)
        self.line_plotter = None
        self.images_plotter = None
        self.visualization_init()

        # Network
        self.icarl = None
        self.model_init()

        # Dataset
        self.data_loader, self.num_tasks, self.transform = return_data(args)

        # icarl
        self.K = args.icarl_K
        self.num_cls_per_task = args.icarl_num_cls_per_task
        self.feature_size = args.icarl_feature_size

    def model_init(self):
        self.icarl = Icarl(args=self.args)
        self.icarl.apply(weights_init)

        self.icarl = cuda(self.icarl, self.args.cuda)

        if self.multi_gpu:
            self.icarl = nn.DataParallel(self.icarl).cuda()

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.icarl.train()
        elif mode == 'eval':
            self.icarl.eval()
        else:
            raise ('mode error. It should be either train or eval')

    def visualization_init(self):
        if self.reset_env:
            self.delete_logs()

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.visdom:
            self.line_plotter = VisdomPlotter(env_name=self.env_name, port=self.visdom_port)
            self.images_plotter = VisdomImagesPlotter(env_name=self.env_name, port=self.visdom_port)

    def delete_logs(self):
        dirs_to_del = [self.ckpt_dir, self.output_dir, self.eval_file_path]

        for dir_to_del in dirs_to_del:
            if os.path.exists(str(dir_to_del)):
                if os.path.isdir(str(dir_to_del)):
                    shutil.rmtree(str(dir_to_del))
                else:
                    os.remove(str(dir_to_del))

    def log_csv(self, task, epoch, g_iter, train_loss, train_acc, test_loss, test_acc, filename='log.csv'):
        file_path = self.output_dir.joinpath(filename)
        if not file_path.is_file():
            file = open(file_path, 'w', encoding='utf-8')
        else:
            file = open(file_path, 'a', encoding='utf-8')
        wr = csv.writer(file)
        wr.writerow([task, g_iter, epoch, round(train_loss, 4), round(train_acc, 4), round(test_loss, 4), round(test_acc, 4)])
        file.close()

    def _init_fn(self, worker_id):
        np.random.seed(int(self.args.seed))

    def train(self):
        self.set_mode('train')

        acc_log = np.zeros((7-1, 7), dtype=np.float32)

        for s in range(1, 7, self.num_cls_per_task):
            if s == 1:
                data_loader, _, transform = return_data(self.args, class_range=range(0, s+self.num_cls_per_task))
            else:
                data_loader, _, transform = return_data(self.args, class_range=range(s, s+self.num_cls_per_task))

            train_loader = data_loader['train']
            test_loader = data_loader['test']

            train_set = train_loader.dataset
            test_set = test_loader.dataset

            # Update representation via BackProp
            self.icarl.update_representation(train_set, current_class_count=s+self.num_cls_per_task)
            m = self.K // self.icarl.n_classes

            # Reduce exemplar sets for known classes
            if self.K != 0:
                self.icarl.reduce_exemplar_sets(m)

                # Construct exemplar sets for new classes
                for y in range(self.icarl.n_known, self.icarl.n_classes):
                    print("Constructing exemplar set for class-{}...".format(y+1))
                    y_class_dataloader, _, _ = return_data(self.args, class_range=range(y, y+1))
                    images_concat = None
                    paths_concat = None
                    for indices, images, _, paths in y_class_dataloader['train']:
                        if images_concat is None:
                            images_concat = images
                            paths_concat = paths
                        else:
                            images_concat = np.concatenate((images_concat, images))
                            paths_concat = np.concatenate((paths_concat, paths))
                    self.icarl.construct_exemplar_set(paths_concat, images_concat, m)
                    print("Done")

                for y, P_y in enumerate(self.icarl.exemplar_sets):
                    print("Exemplar set for class-{}: {}".format(y, P_y.shape))
                    # show_images(P_y[:10])

            # n_known 값 업데이트
            self.icarl.n_known = self.icarl.n_classes
            print("iCaRL classes: {}".format(self.icarl.n_known))
            print("iCaRL model last nodes: {}".format(self.icarl.fc.out_features))

            # print train_set_accuracy
            total = 0.0
            correct = 0.0
            for indices, images, labels, _ in train_loader:
                images = cuda(Variable(images), self.cuda)
                preds = self.icarl.classify(images, transform)
                total += labels.size(0)
                correct += (preds.data.cpu() == labels.data.cpu()).sum()
            train_acc = float(correct) / float(total)

            print('Train Accuracy: %d %%' % (100 * train_acc))

            # print test_set_accuracy
            total = 0.0
            correct = 0.0
            for indices, images, labels, _ in test_loader:
                images = cuda(Variable(images), self.cuda)
                preds = self.icarl.classify(images, transform)
                total += labels.size(0)
                correct += (preds.data.cpu() == labels.data.cpu()).sum()
            test_acc = float(correct )/ float(total)

            print('Test Accuracy: %d %%' % (100 * test_acc))

            # log
            for old_task_idx in range(s+1):
                log_data_loader, _, transform = return_data(self.args, class_range=range(old_task_idx, old_task_idx+1))
                log_test_loader = log_data_loader['eval']
                for indices, images, labels, _ in log_test_loader:
                    images = cuda(Variable(images), self.cuda)
                    preds = self.icarl.classify(images, transform)
                    total += labels.size(0)
                    correct += (preds.data.cpu() == labels.data.cpu()).sum()
                test_acc = float(correct) / float(total)
                acc_log[s-1, old_task_idx] = test_acc

                np.savetxt(self.eval_file_path, acc_log, '%.4f')
                print('Save at ' + self.eval_file_path)

        utils.append_settings_to_file(self.eval_file_path, self.args)






