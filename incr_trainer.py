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
from models.gem_inc import GemInc
from utils import cuda, make_log_name, check_log_dir, VisdomPlotter, VisdomImagesPlotter, set_seed, append_conf_mat_to_file


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


class IncrementalTrainer(object):
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
        self.model = None
        self.model_init()

        # icarl
        # self.K = args.icarl_K
        self.num_cls_per_task = args.icarl_num_cls_per_task
        self.feature_size = args.icarl_feature_size

    def model_init(self):
        if self.args.icarl:
            self.model = Icarl(args=self.args)
        elif self.args.gem_inc:
            self.model = GemInc(args=self.args)
        else:
            raise ValueError("incremental learning should choose at least one method")
        self.model.apply(weights_init)

        self.model = cuda(self.model, self.args.cuda)

        if self.multi_gpu:
            self.model = nn.DataParallel(self.model).cuda()

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':

            self.model.eval()
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
        conf_mats = {}

        for s in range(1, 7, self.num_cls_per_task):
            if s == 1:
                data_loader, _, transform = return_data(self.args, class_range=range(0, s+self.num_cls_per_task))
            else:
                data_loader, _, transform = return_data(self.args, class_range=range(s, s+self.num_cls_per_task))

            train_loader = data_loader['train']
            test_loader = data_loader['test']

            train_set = train_loader.dataset
            test_set = test_loader.dataset

            print("# n_known: {}".format(self.model.n_known + 1))

            # Update representation via BackProp
            self.model.update_representation(train_set,
                                             current_class_count=s+self.num_cls_per_task,
                                             train_loader=train_loader,
                                             test_loader=test_loader,
                                             line_plotter=self.line_plotter)

            # n_known 값 업데이트
            self.model.update_n_known()

            # print train_set_accuracy
            total = 0.0
            correct = 0.0
            for indices, images, labels, _ in train_loader:
                images = cuda(Variable(images), self.cuda)
                preds = self.model.classify(images)
                total += labels.size(0)
                correct += (preds.data.cpu() == labels.data.cpu()).sum()
            train_acc = float(correct) / float(total)

            print('Train Accuracy: %d %%' % (100 * train_acc))

            # print test_set_accuracy
            total = 0.0
            correct = 0.0
            for indices, images, labels, _ in test_loader:
                images = cuda(Variable(images), self.cuda)
                preds = self.model.classify(images)
                total += labels.size(0)
                correct += (preds.data.cpu() == labels.data.cpu()).sum()
            test_acc = float(correct)/ float(total)

            print('Test Accuracy: %d %%' % (100 * test_acc))

            # log
            conf_mat = np.zeros((s+1, s+1), dtype=np.float32)
            for old_task_idx in range(s+1):
                log_data_loader, _, transform = return_data(self.args, class_range=range(old_task_idx, old_task_idx+1))
                log_test_loader = log_data_loader['eval']

                for indices, images, labels, _ in log_test_loader:
                    images = cuda(Variable(images), self.cuda)
                    preds = self.model.classify(images)
                    total += labels.size(0)
                    correct += (preds.data.cpu() == labels.data.cpu()).sum()
                    for i in range(labels.data.cpu().size()[0]):
                        conf_mat[labels.data.cpu()[i], preds.data.cpu()[i]] += 1
                test_acc = float(correct) / float(total)
                acc_log[s-1, old_task_idx] = test_acc

                np.savetxt(self.eval_file_path, acc_log, '%.4f')
                print('Save at ' + self.eval_file_path)

            print("conf mat:")
            print(conf_mat)
            conf_mats[s] = conf_mat.tolist()

        append_conf_mat_to_file(self.eval_file_path, conf_mats)
        utils.append_settings_to_file(self.eval_file_path, self.args)
