import csv
import os
from pathlib import Path

import torch
from torch import optim, nn
import shutil

import numpy as np
from dataloader import return_data
from model import Dcnn
from utils import cuda, make_log_name, check_log_dir, VisdomLinePlotter


## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)


class DCNN(object):
    def __init__(self, args):
        self.args = args

        # Evaluation
        # self.eval_dir = Path(args.eval_dir).joinpath(args.env_name)
        self.eval_dir = args.eval_dir
        check_log_dir(self.eval_dir)
        self.log_name = make_log_name(args)

        # Misc
        self.cuda = args.cuda and torch.cuda.is_available()
        self.multi_gpu = args.multi_gpu

        # Optimization
        self.epoch = args.epoch
        self.epoch_i = 0
        self.task_idx = 0
        self.train_batch_size = args.train_batch_size
        self.lr = args.lr
        self.global_iter = 0
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = args.early_stopping
        self.early_stopping_iter = args.early_stopping_iter

        # Visualization
        self.reset_env = args.reset_env
        self.env_name = args.env_name
        self.visdom = args.visdom
        self.visdom_port = args.visdom_port
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        self.output_dir = Path(args.output_dir).joinpath(args.env_name)
        self.summary_dir = Path("./tensorboard_logs/").joinpath(self.env_name)
        self.visualization_init()

        # Network
        self.cnn_type = args.cnn_type
        self.load_ckpt = args.load_ckpt
        self.input_channel = args.channel
        self.image_size = args.image_size
        self.model_init()

        # Dataset
        self.data_loader, self.num_tasks = return_data(args)

        # Continual Learning
        self.continual = args.continual

        # EWC
        self.ewc = args.ewc
        self.lamb = args.lamb
        self.online = args.online
        self.gamma = args.gamma
        self.EWC_task_count = 0

        if self.ewc and not self.continual:
            raise ValueError("Cannot set EWC with no continual setting")

    def model_init(self):
        # TODO: CNN model_init
        self.C = Dcnn(self.input_channel)

        self.C.apply(weights_init)

        self.C_optim = optim.Adam(self.C.parameters(), lr=self.lr)

        if self.cuda:
            self.C = cuda(self.C, self.cuda)

        if self.multi_gpu:
            self.C = nn.DataParallel(self.C).cuda()

        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.load_ckpt:
            self.load_checkpoint()

    def visualization_init(self):
        if self.reset_env:
            self.delete_logs()

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.visdom:
            self.plotter = VisdomLinePlotter(env_name=self.env_name, port=self.visdom_port)

    def delete_logs(self):
        dirs_to_del = [self.ckpt_dir, self.output_dir, self.summary_dir]

        for dir_to_del in dirs_to_del:
            if os.path.exists(str(dir_to_del)):
                shutil.rmtree(str(dir_to_del))

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.C.train()
        elif mode == 'eval':
            self.C.eval()
        else:
            raise ('mode error. It should be either train or eval')

    def save_checkpoint(self, filename='ckpt_cnn.tar'):
        # TODO: CNN save_checkpoint
        model_states = {'C': self.C.state_dict()}
        optim_states = {'C_optim':self.C_optim.state_dict()}
        states = {'args': self.args,
                  'epoch': self.epoch,
                  'epoch_i': self.epoch_i,
                  'task_idx': self.task_idx,
                  'global_iter': self.global_iter,
                  'model_states': model_states,
                  'optim_states': optim_states}

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states, file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='ckpt_cnn.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            checkpoint = torch.load(file_path.open('rb'))
            self.args = checkpoint['args']
            self.epoch = checkpoint['epoch']
            self.epoch_i = checkpoint['epoch_i']
            self.task_idx = checkpoint['task_idx']
            self.global_iter = checkpoint['global_iter']
            self.C.load_state_dict(checkpoint['model_states']['C'], strict=False)
            self.C_optim.load_state_dict(checkpoint['optim_states']['C_optim'])
            self.data_loader = return_data(self.args)
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def train(self):
        self.set_mode('train')
        min_loss = None
        min_loss_not_updated = 0
        early_stop = False

        acc_log = np.zeros((self.num_tasks, self.num_tasks), dtype=np.float32)

        while self.task_idx < self.num_tasks:
            while True:
                if self.epoch_i >= self.epoch or early_stop:
                    self.epoch_i = 0
                    break
                self.epoch_i += 1

                if self.continual:
                    data_loader = self.data_loader['task{}'.format(self.task_idx)]['train']
                else:
                    data_loader = self.data_loader['train']

                for i, (images, labels) in enumerate(data_loader):
                    images = cuda(images, self.cuda)
                    labels = cuda(labels, self.cuda)

                    self.global_iter += 1
                    # Forward
                    outputs = self.C(images)
                    train_loss = self.compute_loss(outputs, labels)

                    # Backward
                    self.C_optim.zero_grad()
                    train_loss.backward()
                    self.C_optim.step()

                    # train acc
                    _, predicted = torch.max(outputs, 1)
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()
                    train_acc = 100 * correct / total

                    if self.global_iter % 1 == 0:

                        test_loss, test_acc = self.evaluate(self.task_idx)

                        print('Task [{}], Epoch [{}/{}], Iter [{}], train loss: {:.4f}, train acc.: {:.4f}, test loss:{:.4f}, test acc.: {:.4f}, min_loss_not_updated: {}'
                              .format(self.task_idx + 1, self.epoch_i, self.epoch, self.global_iter, train_loss.item(), train_acc, test_loss.item(), test_acc, min_loss_not_updated))

                    if self.global_iter % 10 == 0:
                        # make csv file
                        self.log_csv(self.task_idx, self.epoch_i, self.global_iter, train_loss.item(), train_acc, test_loss.item(), test_acc)
                        self.save_checkpoint()

                        # visdom
                        if self.visdom:
                            self.plotter.plot(var_name='loss',
                                              split_name='train',
                                              title_name='Current task Loss',
                                              x=self.global_iter,
                                              y=train_loss.item())
                            self.plotter.plot(var_name='loss',
                                              split_name='test',
                                              title_name='Current task Loss',
                                              x=self.global_iter,
                                              y=test_loss.item())
                            self.plotter.plot(var_name='acc.',
                                              split_name='train',
                                              title_name='Current task Accuracy',
                                              x=self.global_iter,
                                              y=train_acc)
                            self.plotter.plot(var_name='acc.',
                                              split_name='test',
                                              title_name='Current task Accuracy',
                                              x=self.global_iter,
                                              y=test_acc)

                            for old_task_idx in range(self.task_idx+1):
                                eval_loss, eval_acc = self.evaluate(old_task_idx)
                                if not isinstance(eval_loss, float):
                                    eval_loss = eval_loss.item()
                                self.plotter.plot(var_name='task acc.',
                                                  split_name='task {}'.format(old_task_idx+1),
                                                  title_name='Task Accuracy',
                                                  x=self.global_iter,
                                                  y=eval_acc)

                                self.plotter.plot(var_name='task loss',
                                                  split_name='task {}'.format(old_task_idx+1),
                                                  title_name='Task Loss',
                                                  x=self.global_iter,
                                                  y=eval_loss)


                    if min_loss is None:
                        min_loss = train_loss.item()
                    elif train_loss.item() < min_loss:
                        min_loss = train_loss.item()
                        min_loss_not_updated = 0
                    else:
                        min_loss_not_updated += 1

                    if self.early_stopping and (min_loss_not_updated >= self.early_stopping_iter):
                        early_stop = True

            for old_task_idx in range(self.task_idx+1):
                eval_loss, eval_acc = self.evaluate(old_task_idx)
                print("Task{} test loss: {:.3f}, Test acc.: {:.3f}".format(old_task_idx + 1, eval_loss, eval_acc))

                np.savetxt(self.eval_dir + self.log_name, acc_log, '%.4f')
                print('Save at ' + self.eval_dir + self.log_name)

            fisher_mat = self.estimate_fisher(self.task_idx)
            self.store_fisher_n_params(fisher_mat)
            print('Fisher matrix for task {} stored successfully!'.format(self.task_idx+1))
            self.task_idx += 1

    def log_csv(self, task, epoch, g_iter, train_loss, train_acc, test_loss, test_acc, filename='log.csv'):
        file_path = self.output_dir.joinpath(filename)
        if not file_path.is_file():
            file = open(file_path, 'w', encoding='utf-8')
        else:
            file = open(file_path, 'a', encoding='utf-8')
        wr = csv.writer(file)
        wr.writerow([task, g_iter, epoch, round(train_loss, 4), round(train_acc, 4), round(test_loss, 4), round(test_acc, 4)])
        file.close()

    def evaluate(self, task_idx):
        # self.load_checkpoint()
        self.set_mode('eval')

        eval_acc = 0
        test_loss = 0
        with torch.no_grad():
            if self.continual:
                data_loader = self.data_loader['task{}'.format(task_idx)]['test']
            else:
                data_loader = self.data_loader['test']

            for i, (images, labels) in enumerate(data_loader):
                images = cuda(images, self.cuda)
                labels = cuda(labels, self.cuda)

                outputs = self.C(images)
                _, predicted = torch.max(outputs, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                eval_acc += 100 * correct / total
                test_loss += self.compute_loss(outputs, labels)

                # env_name = self.args.env_name
                # print("##### Env name: {} #####".format(env_name))

                # print("Epoch: {}, iter: {}, test loss: {:.3f}, Test acc.: {:.3f}".format(self.epoch_i, self.global_iter, test_loss, eval_acc))
            eval_acc = eval_acc / (i+1)
            test_loss = test_loss / (i+1)

        return test_loss, eval_acc

    def compute_loss(self,outputs, targets):
        loss = self.criterion(outputs, targets)

        # Regularization for all previous tasks
        reg_loss = 0.
        if self.ewc:
            reg_loss = self.ewc_loss()

        return loss + self.lamb * reg_loss

    # ----------------- EWC-specifc functions -----------------#

    def estimate_fisher(self, task_idx):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.
        [data_loader]:          <DataLoadert> to be used to estimate FI-matrix'''

        self.set_mode('eval')

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.C.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        for i, (images, labels) in enumerate(self.data_loader['task{}'.format(task_idx)]['train']):
            images = cuda(images, self.cuda)
            labels = cuda(labels, self.cuda)

            # Forward
            outputs = self.C(images)
            train_loss = self.compute_loss(outputs, labels)
            self.C_optim.zero_grad()
            train_loss.backward()

            # Square gradients and keep running sum
            for n, p in self.C.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        for n, p in self.C.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] /= (i+1)

        return est_fisher_info

    def store_fisher_n_params(self, fisher):

        # Store new values in the network
        for n, p in self.C.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.C.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count == 1:
                    existing_values = getattr(self.C, '{}_EWC_estimated_fisher'.format(n))
                    fisher[n] += self.gamma * existing_values
                self.C.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                                     fisher[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count > 0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count + 1):
                for n, p in self.C.named_parameters(): # TODO: actor_critic -> C 로 변경함 (unresolved reference)
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self.C, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self.C, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma * fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p - mean) ** 2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1. / 2) * sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return 0.
