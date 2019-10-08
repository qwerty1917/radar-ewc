import csv
import os
from pathlib import Path

import torch
from torch import optim, nn
import shutil

import numpy as np
from dataloader import return_data
from models.cnn import Dcnn
from utils import cuda, make_log_name, check_log_dir, VisdomLinePlotter, set_seed


## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)

# TODO: multi seed for model and subject shuffle
class DCNN(object):
    def __init__(self, args):
        self.args = args

        set_seed(args.model_seed)

        # Evaluation
        # self.eval_dir = Path(args.eval_dir).joinpath(args.env_name)
        self.eval_dir = os.path.join(args.eval_dir, args.date, args.continual)
        self.model_dir = os.path.join(args.model_dir, args.date, args.continual)

        check_log_dir(self.eval_dir)
        check_log_dir(self.model_dir)
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
        self.date = args.date
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
        self.multi = args.multi
        self.num_tasks = args.num_tasks if self.multi else 1
        self.continual = args.continual

        self.model_init()

        # Dataset
        self.pretrain = args.pretrain
        self.num_pre_tasks = args.num_pre_tasks
        self.data_loader, self.num_tasks = return_data(args)

    def model_init(self):
        self.C = Dcnn(self.input_channel, self.multi, continual=False, num_tasks=self.num_tasks)

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

        if self.pretrain:
            acc_log = np.zeros((12, 12), dtype=np.float32)
        else:
            acc_log = np.zeros((1, 1), dtype=np.float32)

        data_loader = self.data_loader['train']

        while True:
            if self.epoch_i >= self.epoch or early_stop:
                self.epoch_i = 0
                break
            self.epoch_i += 1

            for i, (images, sub_idxs, labels) in enumerate(data_loader):
                images = cuda(images, self.cuda)
                labels = cuda(labels, self.cuda)
                sub_idxs = cuda(sub_idxs, self.cuda)
                sub_idxs = sub_idxs.long()

                self.global_iter += 1
                # Forward
                outputs = self.C(images)

                if self.multi:
                    # turn output list of heads to tensor and switch index
                    outputs = torch.stack(outputs).permute(1,0,2)

                    outputs = torch.gather(outputs, dim=1, index=sub_idxs.view(-1,1,1).expand(-1, 1, outputs.shape[-1]))
                    outputs = outputs.squeeze(1)  # remove redundant dim
                train_loss = self.criterion(outputs, labels)

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

                    test_loss, test_acc = self.evaluate()

                    print('Epoch [{}/{}], Iter [{}], train loss: {:.4f}, train acc.: {:.4f}, test loss:{:.4f}, test acc.: {:.4f}, min_loss_not_updated: {}'
                          .format(self.epoch_i, self.epoch, self.global_iter, train_loss.item(), train_acc, test_loss.item(), test_acc, min_loss_not_updated))

                if self.global_iter % 10 == 0:
                    # make csv file
                    self.log_csv(self.task_idx, self.epoch_i, self.global_iter, train_loss.item(), train_acc, test_loss.item(), test_acc, filename=self.log_name)
                    self.save_checkpoint(filename=self.log_name+'_ckpt.tar')

                    # visdom
                    if self.visdom:
                        self.plotter.plot(var_name='loss',
                                          split_name='train',
                                          title_name=self.date + ' Current task Loss',
                                          x=self.global_iter,
                                          y=train_loss.item())
                        self.plotter.plot(var_name='loss',
                                          split_name='test',
                                          title_name=self.date + ' Current task Loss',
                                          x=self.global_iter,
                                          y=test_loss.item())
                        self.plotter.plot(var_name='acc.',
                                          split_name='train',
                                          title_name=self.date + ' Current task Accuracy',
                                          x=self.global_iter,
                                          y=train_acc)
                        self.plotter.plot(var_name='acc.',
                                          split_name='test',
                                          title_name=self.date + ' Current task Accuracy',
                                          x=self.global_iter,
                                          y=test_acc)


                if min_loss is None:
                    min_loss = train_loss.item()
                elif train_loss.item() < min_loss:
                    min_loss = train_loss.item()
                    min_loss_not_updated = 0
                else:
                    min_loss_not_updated += 1

                if self.early_stopping and (min_loss_not_updated >= self.early_stopping_iter):
                    early_stop = True

        eval_loss, eval_acc = self.evaluate()
        print("Final test loss: {:.3f}, Test acc.: {:.3f}".format(eval_loss, eval_acc))
        if self.pretrain:
            for task_idx in range(12):
                _, eval_acc = self.evaluate(task_idx)
                print("Task{} - Test acc.: {:.3f}".format(task_idx + 1, eval_acc))
                acc_log[self.task_idx, task_idx] = eval_acc

        else:
            acc_log[self.task_idx, self.task_idx] = eval_acc

        np.savetxt(os.path.join(self.eval_dir, self.log_name) + '.txt', acc_log, '%.4f')
        print('Log saved at ' + os.path.join(self.eval_dir, self.log_name))
        torch.save(self.C.state_dict(), os.path.join(self.model_dir, self.log_name) + '.pt')
        print('Model saved at ' + os.path.join(self.eval_dir, self.log_name))

    def log_csv(self, task, epoch, g_iter, train_loss, train_acc, test_loss, test_acc, filename='log.csv'):
        file_path = self.output_dir.joinpath(filename)
        if not file_path.is_file():
            file = open(file_path, 'w', encoding='utf-8')
        else:
            file = open(file_path, 'a', encoding='utf-8')
        wr = csv.writer(file)
        wr.writerow([task, g_iter, epoch, round(train_loss, 4), round(train_acc, 4), round(test_loss, 4), round(test_acc, 4)])
        file.close()

    def evaluate(self, task_idx=None):
        # self.load_checkpoint()
        self.set_mode('eval')

        eval_acc = 0
        test_loss = 0
        with torch.no_grad():

            if task_idx is None:
                data_loader = self.data_loader['test']
            else:
                data_loader = self.data_loader['task{}'.format(task_idx)]['test']

            for i, (images, sub_idxs, labels) in enumerate(data_loader):
                images = cuda(images, self.cuda)
                labels = cuda(labels, self.cuda)
                sub_idxs = cuda(sub_idxs, self.cuda)
                sub_idxs = sub_idxs.long()

                outputs = self.C(images)

                if self.multi:
                    # turn output list of heads to tensor and switch index
                    outputs = torch.stack(outputs).permute(1,0,2)

                    outputs = torch.gather(outputs, dim=1, index=sub_idxs.view(-1,1,1).expand(-1, 1, outputs.shape[-1]))
                    outputs = outputs.squeeze(1)  # remove redundant dim

                _, predicted = torch.max(outputs, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                eval_acc += 100 * correct / total
                test_loss += self.criterion(outputs, labels)

                # env_name = self.args.env_name
                # print("##### Env name: {} #####".format(env_name))

                # print("Epoch: {}, iter: {}, test loss: {:.3f}, Test acc.: {:.3f}".format(self.epoch_i, self.global_iter, test_loss, eval_acc))
            eval_acc = eval_acc / (i+1)
            test_loss = test_loss / (i+1)
        # reset model to train mode
        self.set_mode('train')
        return test_loss, eval_acc