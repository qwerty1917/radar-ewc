import csv
import os
from pathlib import Path

import torch
from torch import optim, nn
import shutil
import math

import numpy as np
from dataloader import return_data
from models.baye_cnn import Dcnn
from copy import deepcopy
from utils import cuda, make_log_name, check_log_dir, VisdomLinePlotter, set_seed, freeze_model, Adam
from models.bayes_layer import BayesianLinear, BayesianConv2D, _calculate_fan_in_and_fan_out

## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     # m.weight.data.normal_(0.0, 0.02)
    #     nn.init.kaiming_normal_(m.weight)
    #     m.bias.data.fill_(0)
    # elif classname.find('Linear') != -1:
    #     nn.init.kaiming_normal_(m.weight)
    #     m.bias.data.fill_(0)
    if classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)


class baye_DCNN(object):
    def __init__(self, args):
        self.args = args

        if args.subject_shuffle:
            set_seed(0)
        else:
            set_seed(args.seed)

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
        self.lr_rho = args.lr_rho

        self.global_iter = 0
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = args.early_stopping
        self.early_stopping_iter = args.early_stopping_iter

        # Visualization
        self.date = args.date
        self.reset_env = args.reset_env
        self.env_name = args.env_name
        self.visdom = args.visdom
        self.visdom_port = args.visdom_port
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        self.output_dir = Path(args.output_dir).joinpath(args.env_name)
        self.summary_dir = Path("./tensorboard_logs/").joinpath(self.env_name)
        self.visualization_init()

        # Continual Learning
        self.continual = args.continual
        self.lamb = args.lamb

        # UCL
        self.beta = args.lamb
        self.alpha = args.alpha
        self.saved = 0
        self.ratio = args.ratio
        self.param_name = []

        # Network
        self.cnn_type = args.cnn_type
        self.load_ckpt = args.load_ckpt
        self.input_channel = args.channel
        self.image_size = args.image_size
        self.multi = args.multi
        self.num_tasks = args.num_tasks
        self.model_init()

        # Dataset
        self.data_loader, self.num_tasks = return_data(args)

        if self.ewc and not self.continual:
            raise ValueError("Cannot set EWC with no continual setting")

    def model_init(self):
        self.C = Dcnn(self.input_channel, self.ratio, self.multi, self.num_tasks)
        self.C_old = Dcnn(self.input_channel, self.ratio, self.multi, self.num_tasks)
        self.C_old.load_state_dict(self.C.state_dict())

        self.C.apply(weights_init)

        for (name, p) in self.C.named_parameters():
            self.param_name.append(name)

        self.C_optim = Adam(self.C.parameters(), lr=self.lr, lr_rho=self.lr_rho, param_name=self.param_name)

        if self.cuda:
            self.C = cuda(self.C, self.cuda)
            self.C_old = cuda(self.C_old, self.cuda)

        if self.multi_gpu:
            self.C = nn.DataParallel(self.C).cuda()
            self.C_old = nn.DataParallel(self.C_old).cuda()


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

            if self.continual:
                data_loader = self.data_loader['task{}'.format(self.task_idx)]['train']
            else:
                data_loader = self.data_loader['train']

            while True:
                if self.epoch_i >= self.epoch or early_stop:
                    self.epoch_i = 0
                    break
                self.epoch_i += 1

                for i, (images, labels) in enumerate(data_loader):
                    images = cuda(images, self.cuda)
                    labels = cuda(labels, self.cuda)

                    self.global_iter += 1
                    # Forward
                    if self.multi:
                        outputs = self.C(images, self.task_idx)
                    else:
                        outputs = self.C(images)
                    train_loss = self.compute_loss(outputs, labels, mini_batch_size=outputs.size[0])

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
                        self.log_csv(self.task_idx, self.epoch_i, self.global_iter, train_loss.item(), train_acc, test_loss.item(), test_acc, filename=self.log_name)
                        self.save_checkpoint(filename=self.log_name+'_ckpt.tar')

                        # visdom
                        if self.visdom:
                            self.plotter.plot(var_name='loss',
                                              split_name='train',
                                              title_name=self.date + ' Current task Loss' + ' lamb{}'.format(self.lamb),
                                              x=self.global_iter,
                                              y=train_loss.item())
                            self.plotter.plot(var_name='loss',
                                              split_name='test',
                                              title_name=self.date + ' Current task Loss' + ' lamb{}'.format(self.lamb),
                                              x=self.global_iter,
                                              y=test_loss.item())
                            self.plotter.plot(var_name='acc.',
                                              split_name='train',
                                              title_name=self.date + ' Current task Accuracy' + ' lamb{}'.format(self.lamb),
                                              x=self.global_iter,
                                              y=train_acc)
                            self.plotter.plot(var_name='acc.',
                                              split_name='test',
                                              title_name=self.date + ' Current task Accuracy' + ' lamb{}'.format(self.lamb),
                                              x=self.global_iter,
                                              y=test_acc)

                            task_loss_sum = 0
                            task_acc_sum = 0
                            for old_task_idx in range(self.task_idx+1):
                                eval_loss, eval_acc = self.evaluate(old_task_idx)
                                if not isinstance(eval_loss, float):
                                    eval_loss = eval_loss.item()

                                task_loss_sum += eval_loss
                                task_acc_sum += eval_acc
                                self.plotter.plot(var_name='task acc.',
                                                  split_name='task {}'.format(old_task_idx+1),
                                                  title_name=self.date + ' Task Accuracy' + ' lamb{}'.format(self.lamb),
                                                  x=self.global_iter,
                                                  y=eval_acc)

                                self.plotter.plot(var_name='task loss',
                                                  split_name='task {}'.format(old_task_idx+1),
                                                  title_name=self.date + ' Task Loss' + ' lamb{}'.format(self.lamb),
                                                  x=self.global_iter,
                                                  y=eval_loss)

                            self.plotter.plot(var_name='task average acc.',
                                              split_name='until task {}'.format(self.task_idx+1),
                                              title_name = self.date + ' Task average acc.' + ' lamb{}'.format(self.lamb),
                                              x=self.global_iter,
                                              y=task_acc_sum/(self.task_idx+1))

                            self.plotter.plot(var_name='task average loss',
                                              split_name='until task {}'.format(self.task_idx+1),
                                              title_name = self.date + ' Task average loss' + ' lamb{}'.format(self.lamb),
                                              x=self.global_iter,
                                              y=task_loss_sum/(self.task_idx+1))



                    if min_loss is None:
                        min_loss = train_loss.item()
                    elif train_loss.item() < min_loss:
                        min_loss = train_loss.item()
                        min_loss_not_updated = 0
                    else:
                        min_loss_not_updated += 1

                    if self.early_stopping and (min_loss_not_updated >= self.early_stopping_iter):
                        early_stop = True

                freeze_model(self.C_old)

            for old_task_idx in range(self.task_idx+1):
                eval_loss, eval_acc = self.evaluate(old_task_idx)
                print("Task{} test loss: {:.3f}, Test acc.: {:.3f}".format(old_task_idx + 1, eval_loss, eval_acc))
                acc_log[self.task_idx, old_task_idx] = eval_acc

                np.savetxt(self.eval_dir + self.log_name + '.txt', acc_log, '%.4f')
                print('Save at ' + self.eval_dir + self.log_name)

            # self.C_old = deepcopy(self.C)
            self.C_old.load_state_dict(self.C.state_dict())
            self.saved = 1
            print('Old model saved successfully!')

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

                if self.multi:
                    outputs = self.C(images, task_idx)
                else:
                    outputs = self.C(images)

                _, predicted = torch.max(outputs, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                eval_acc += 100 * correct / total
                test_loss += self.compute_loss(outputs, labels, mini_batch_size=outputs.size[0])

                # env_name = self.args.env_name
                # print("##### Env name: {} #####".format(env_name))

                # print("Epoch: {}, iter: {}, test loss: {:.3f}, Test acc.: {:.3f}".format(self.epoch_i, self.global_iter, test_loss, eval_acc))
            eval_acc = eval_acc / (i+1)
            test_loss = test_loss / (i+1)
        # reset model to train mode
        self.set_mode('train')
        return test_loss, eval_acc

    def compute_loss(self, outputs, targets, mini_batch_size):
        loss = self.criterion(outputs, targets)

        # Regularization for all previous tasks
        reg_loss = self.custom_regularization(self.C_old, self.C, mini_batch_size)

        return loss + self.lamb * reg_loss

    # custom regularization

    def custom_regularization(self, saver_net, trainer_net, mini_batch_size):

        sigma_weight_reg_sum = 0
        sigma_bias_reg_sum = 0
        sigma_weight_normal_reg_sum = 0
        sigma_bias_normal_reg_sum = 0
        mu_weight_reg_sum = 0
        mu_bias_reg_sum = 0
        L1_mu_weight_reg_sum = 0
        L1_mu_bias_reg_sum = 0

        out_features_max = 512
        alpha = self.alpha
        #         alpha = 0.01
        #         if args.conv_net:
        #             alpha = 1
        if self.saved:
            alpha = 1

        prev_weight_strength = nn.Parameter(torch.Tensor(1, 1, 1, 1).uniform_(0, 0))
            # else:
            #     prev_weight_strength = nn.Parameter(torch.Tensor(3, 1, 1, 1).uniform_(0, 0))


        for (_, saver_layer), (_, trainer_layer) in zip(saver_net.named_children(), trainer_net.named_children()):
            if isinstance(trainer_layer, BayesianLinear) == False and isinstance(trainer_layer,
                                                                                 BayesianConv2D) == False:
                continue
            # calculate mu regularization
            trainer_weight_mu = trainer_layer.weight_mu
            saver_weight_mu = saver_layer.weight_mu
            trainer_bias = trainer_layer.bias
            saver_bias = saver_layer.bias

            fan_in, fan_out = _calculate_fan_in_and_fan_out(trainer_weight_mu)

            trainer_weight_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
            saver_weight_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))

            if isinstance(trainer_layer, BayesianLinear):
                std_init = math.sqrt((2 / fan_in) * self.ratio)
            if isinstance(trainer_layer, BayesianConv2D):
                std_init = math.sqrt((2 / fan_out) * self.ratio)
            #             std_init = np.log(1+np.exp(args.rho))

            saver_weight_strength = (std_init / saver_weight_sigma)

            if len(saver_weight_mu.shape) == 4:
                out_features, in_features, _, _ = saver_weight_mu.shape
                curr_strength = saver_weight_strength.expand(out_features, in_features, 1, 1)
                prev_strength = prev_weight_strength.permute(1, 0, 2, 3).expand(out_features, in_features, 1, 1)

            else:
                out_features, in_features = saver_weight_mu.shape
                curr_strength = saver_weight_strength.expand(out_features, in_features)
                if len(prev_weight_strength.shape) == 4:
                    feature_size = in_features // (prev_weight_strength.shape[0])
                    prev_weight_strength = prev_weight_strength.reshape(prev_weight_strength.shape[0], -1)
                    prev_weight_strength = prev_weight_strength.expand(prev_weight_strength.shape[0], feature_size)
                    prev_weight_strength = prev_weight_strength.reshape(-1, 1)
                prev_strength = prev_weight_strength.permute(1, 0).expand(out_features, in_features)

            L2_strength = torch.max(curr_strength, prev_strength)
            bias_strength = torch.squeeze(saver_weight_strength)

            L1_sigma = saver_weight_sigma
            bias_sigma = torch.squeeze(saver_weight_sigma)

            prev_weight_strength = saver_weight_strength

            mu_weight_reg = (L2_strength * (trainer_weight_mu - saver_weight_mu)).norm(2) ** 2
            mu_bias_reg = (bias_strength * (trainer_bias - saver_bias)).norm(2) ** 2

            L1_mu_weight_reg = (
                        torch.div(saver_weight_mu ** 2, L1_sigma ** 2) * (trainer_weight_mu - saver_weight_mu)).norm(1)
            L1_mu_bias_reg = (torch.div(saver_bias ** 2, bias_sigma ** 2) * (trainer_bias - saver_bias)).norm(1)

            L1_mu_weight_reg = L1_mu_weight_reg * (std_init ** 2)
            L1_mu_bias_reg = L1_mu_bias_reg * (std_init ** 2)

            weight_sigma = (trainer_weight_sigma ** 2 / saver_weight_sigma ** 2)

            normal_weight_sigma = trainer_weight_sigma ** 2

            sigma_weight_reg_sum = sigma_weight_reg_sum + (weight_sigma - torch.log(weight_sigma)).sum()
            sigma_weight_normal_reg_sum = sigma_weight_normal_reg_sum + (
                        normal_weight_sigma - torch.log(normal_weight_sigma)).sum()

            mu_weight_reg_sum = mu_weight_reg_sum + mu_weight_reg
            mu_bias_reg_sum = mu_bias_reg_sum + mu_bias_reg
            L1_mu_weight_reg_sum = L1_mu_weight_reg_sum + L1_mu_weight_reg
            L1_mu_bias_reg_sum = L1_mu_bias_reg_sum + L1_mu_bias_reg

        # L2 loss
        loss = alpha * (mu_weight_reg_sum + mu_bias_reg_sum) / (2 * mini_batch_size)
        # L1 loss
        loss = loss + self.saved * (L1_mu_weight_reg_sum + L1_mu_bias_reg_sum) / (mini_batch_size)
        # sigma regularization
        loss = loss + self.beta * (sigma_weight_reg_sum + sigma_weight_normal_reg_sum) / (2 * mini_batch_size)

        return loss
