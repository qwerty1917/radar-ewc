import csv
import os
from pathlib import Path

import torch
from torch import optim, nn
import shutil

import numpy as np
import quadprog
from dataloader import return_data
from models.cnn import Dcnn
from copy import deepcopy
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


class gem_DCNN(object):
    def __init__(self, args):
        self.args = args

        # pretrain
        self.num_pre_tasks = args.num_pre_tasks
        self.load_pretrain = args.load_pretrain
        self.model_seed = args.model_seed
        self.pre_reg_param = args.pre_reg_param

        set_seed(self.model_seed)

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
        self.task_idx = self.num_pre_tasks
        self.train_batch_size = args.train_batch_size
        self.lr = args.lr

        self.lr_decay = args.lr_decay
        self.lr_min = args.lr / (args.lr_factor ** 5)
        self.lr_factor = args.lr_factor
        self.lr_patience = args.lr_patience

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

        # Dataset
        self.data_loader, self.num_tasks = return_data(args)

        # Continual Learning
        self.continual = args.continual
        self.is_incremental = args.incremental
        self.lamb = args.lamb
        self.task_count = 0

        # Network
        self.cnn_type = args.cnn_type
        self.load_ckpt = args.load_ckpt
        self.input_channel = args.channel
        self.image_size = args.image_size
        self.multi = args.multi
        self.num_tasks = args.num_tasks
        self.train_tasks = self.num_tasks - self.num_pre_tasks
        self.model_init()

        self.margin = args.memory_strength
        self.n_memories = args.n_memories
        # allocate episodic memory
        self.n_inputs = self.image_size.pow(2)
        self.memory_data = torch.FloatTensor(self.num_tasks, self.n_memories, self.n_inputs)
        self.memory_labs = torch.LongTensor(self.num_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.C.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), self.num_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        # self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.is_incremental:
            self.n_outputs = 7
            self.nc_per_task = int(self.n_outputs / self.num_tasks)
        else:
        # TODO: n_outputs is # of activity class here(should be modified later to be applied in incremental setting)
            self.n_outputs = 7
        self.nc_per_task = self.n_outputs

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return optim.Adam(self.C.parameters(), lr=lr)

    def model_init(self):
        # TODO: CNN model_init
        self.C = Dcnn(self.input_channel, self.multi, continual=True, num_tasks=self.num_tasks)

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

        if self.load_pretrain:
            param_loaded = torch.load(
                './trained_models/20191008/none/subject_m_seed{}_s_seed1_window3_lamb0.0_pre_{}tasks_epochs100.pt'
                    .format(self.model_seed, self.num_pre_tasks))
            self.C.load_state_dict(param_loaded)

            if self.pre_reg_param:
                pass


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
        optim_states = {'C_optim': self.C_optim.state_dict()}
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

    def forward(self, x, t):
        output = self.C(x)
        if self.is_incremental:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def train(self):
        self.set_mode('train')
        min_loss = None
        min_loss_not_updated = 0
        early_stop = False

        acc_log = np.zeros((self.train_tasks, self.num_tasks), dtype=np.float32)

        while self.task_idx < self.num_tasks:

            data_loader = self.data_loader['task{}'.format(self.task_idx)]['train']
            best_loss = np.inf
            best_model = deepcopy(self.C.state_dict())
            lr = self.lr
            patience = self.lr_patience
            self.C_optim = self._get_optimizer(lr)

            while True:
                if self.epoch_i >= self.epoch or early_stop:
                    self.epoch_i = 0
                    break
                self.epoch_i += 1

                for i, (images, labels) in enumerate(data_loader):
                    images = cuda(images, self.cuda)
                    labels = cuda(labels, self.cuda)

                    if self.epoch_i == 1:
                        # Update ring buffer storing examples from current task when first epoch
                        bsz = labels.size(0)
                        endcnt = min(self.mem_cnt + bsz, self.n_memories)
                        effbsz = endcnt - self.mem_cnt
                        self.memory_data[self.task_idx, self.mem_cnt: endcnt].copy_(
                            images.data[: effbsz])
                        if bsz == 1:
                            self.memory_labs[self.task_idx, self.mem_cnt] = labels.data[0]
                        else:
                            self.memory_labs[self.task_idx, self.mem_cnt: endcnt].copy_(
                                labels.data[: effbsz])
                        self.mem_cnt += effbsz
                        if self.mem_cnt == self.n_memories:
                            self.mem_cnt = 0

                    # compute gradient on previous tasks
                    if self.task_idx > 0:
                        for past_task in range(self.task_idx):
                            self.C.zero_grad()
                            # fwd/bwd on the examples in the memory

                            offset1, offset2 = self.compute_offsets(past_task, self.nc_per_task, self.is_incremental)
                            ptloss = self.criterion(
                                self.forward(
                                    self.memory_data[past_task],
                                    past_task)[:, offset1: offset2],
                                self.memory_labs[past_task] - offset1)
                            ptloss.backward()
                            self.store_grad(self.C.parameters, self.grads, self.grad_dims, past_task)


                    # now compute the grad on the current minibatch
                    self.C.zero_grad()

                    offset1, offset2 = self.compute_offsets(self.task_idx, self.nc_per_task, self.is_incremental)
                    outputs = self.forward(images, labels)[:, offset1: offset2]
                    train_loss = self.criterion(outputs, labels - offset1)
                    train_loss.backward()

                    # check if gradient violates constraints
                    if self.task_idx > 0:
                        # copy gradient
                        self.store_grad(self.C.parameters, self.grads, self.grad_dims, self.task_idx)

                        indx = cuda(torch.arange(self.task_idx,dtype=torch.long), self.cuda)
                        dotp = torch.mm(self.grads[:, self.task_idx].unsqueeze(0),
                                        self.grads.index_select(1, indx))
                        if (dotp < 0).sum() != 0:
                            self.project2cone2(self.grads[:, self.task_idx].unsqueeze(1),
                                          self.grads.index_select(1, indx), self.margin)
                            # copy gradients back
                            self.overwrite_grad(self.C.parameters, self.grads[:, self.task_idx],
                                           self.grad_dims)

                    self.C_optim.step()
                    self.global_iter += 1
                    # # Forward
                    # if self.multi:
                    #     outputs = self.C(images, self.task_idx)
                    # else:
                    #     outputs = self.C(images)
                    # train_loss = self.compute_loss(outputs, labels)
                    #
                    # # Backward
                    # self.C_optim.zero_grad()
                    # train_loss.backward()
                    # self.C_optim.step()

                    # train acc
                    _, predicted = torch.max(outputs, 1)
                    total = labels.size(0)
                    correct = (predicted == (labels-offset1)).sum().item()
                    train_acc = 100 * correct / total

                    # TODO : Run and verify
                    if self.global_iter % 5 == 0:

                        test_loss, test_acc = self.evaluate(self.task_idx)

                        print('Task [{}], Epoch [{}/{}], Iter [{}], train loss: {:.4f}, train acc.: {:.4f}, '
                              'test loss:{:.4f}, test acc.: {:.4f}, min_loss_not_updated: {}'
                              .format(self.task_idx + 1, self.epoch_i, self.epoch, self.global_iter, train_loss.item(),
                                      train_acc, test_loss.item(), test_acc, min_loss_not_updated))

                        if self.global_iter % 10 == 0:
                            # make csv file

                            self.log_csv(self.task_idx, self.epoch_i, self.global_iter, train_loss.item(), train_acc,
                                         test_loss.item(), test_acc, filename=self.log_name)
                            self.save_checkpoint(filename=self.log_name + '_ckpt.tar')

                            # visdom
                            if self.visdom:
                                self.plotter.plot(var_name='loss',
                                                  split_name='train',
                                                  title_name=self.date + ' Current task Loss' + ' lamb{}'.format(
                                                      self.lamb),
                                                  x=self.global_iter,
                                                  y=train_loss.item())
                                self.plotter.plot(var_name='loss',
                                                  split_name='test',
                                                  title_name=self.date + ' Current task Loss' + ' lamb{}'.format(
                                                      self.lamb),
                                                  x=self.global_iter,
                                                  y=test_loss.item())
                                self.plotter.plot(var_name='acc.',
                                                  split_name='train',
                                                  title_name=self.date + ' Current task Accuracy' + ' lamb{}'.format(
                                                      self.lamb),
                                                  x=self.global_iter,
                                                  y=train_acc)
                                self.plotter.plot(var_name='acc.',
                                                  split_name='test',
                                                  title_name=self.date + ' Current task Accuracy' + ' lamb{}'.format(
                                                      self.lamb),
                                                  x=self.global_iter,
                                                  y=test_acc)

                                task_loss_sum = 0
                                task_acc_sum = 0
                                for old_task_idx in range(self.task_idx + 1):
                                    eval_loss, eval_acc = self.evaluate(old_task_idx)
                                    if not isinstance(eval_loss, float):
                                        eval_loss = eval_loss.item()

                                    task_loss_sum += eval_loss
                                    task_acc_sum += eval_acc
                                    self.plotter.plot(var_name='task acc.',
                                                      split_name='task {}'.format(old_task_idx + 1),
                                                      title_name=self.date + ' Task Accuracy' + ' lamb{}'.format(
                                                          self.lamb),
                                                      x=self.global_iter,
                                                      y=eval_acc)

                                    self.plotter.plot(var_name='task loss',
                                                      split_name='task {}'.format(old_task_idx + 1),
                                                      title_name=self.date + ' Task Loss' + ' lamb{}'.format(self.lamb),
                                                      x=self.global_iter,
                                                      y=eval_loss)

                                self.plotter.plot(var_name='task average acc.',
                                                  split_name='until task {}'.format(self.task_idx + 1),
                                                  title_name=self.date + ' Task average acc.' + ' lamb{}'.format(
                                                      self.lamb),
                                                  x=self.global_iter,
                                                  y=task_acc_sum / (self.task_idx + 1))

                                self.plotter.plot(var_name='task average loss',
                                                  split_name='until task {}'.format(self.task_idx + 1),
                                                  title_name=self.date + ' Task average loss' + ' lamb{}'.format(
                                                      self.lamb),
                                                  x=self.global_iter,
                                                  y=task_loss_sum / (self.task_idx + 1))

                if self.lr_decay:
                    eval_loss, eval_acc = self.evaluate(self.task_idx)

                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        best_model = deepcopy(self.C.state_dict())
                        patience = self.lr_patience
                        print(' *', end='')
                    else:
                        patience -= 1
                        if patience <= 0:
                            lr /= self.lr_factor
                            print(' lr={:.1e}'.format(lr), end='')
                            if lr < self.lr_min:
                                lr = self.lr_min
                                print()

                            patience = self.lr_patience
                            self.optimizer = self._get_optimizer(lr)

                    # if min_loss is None:
                    #     min_loss = train_loss.item()
                    # elif train_loss.item() < min_loss:
                    #     min_loss = train_loss.item()
                    #     min_loss_not_updated = 0
                    # else:
                    #     min_loss_not_updated += 1
                    #
                    # if self.early_stopping and (min_loss_not_updated >= self.early_stopping_iter):
                    #     early_stop = True

            if self.lr_decay:
                self.C.load_state_dict(best_model)

            # for old_task_idx in range(self.task_idx + 1):
            for t_idx in range(self.num_tasks):
                eval_loss, eval_acc = self.evaluate(t_idx)
                print("Task{} test loss: {:.3f}, Test acc.: {:.3f}".format(t_idx + 1, eval_loss, eval_acc))
                acc_log[self.task_idx-self.num_pre_tasks, t_idx] = eval_acc

            np.savetxt(os.path.join(self.eval_dir, self.log_name) + '.txt', acc_log, '%.4f')
            print('Log saved at ' + os.path.join(self.eval_dir, self.log_name))
            torch.save(self.C.state_dict(), os.path.join(self.model_dir, self.log_name) + '.pt')
            print('Model saved at ' + os.path.join(self.eval_dir, self.log_name))

            self.task_idx += 1

    def log_csv(self, task, epoch, g_iter, train_loss, train_acc, test_loss, test_acc, filename='log.csv'):
        file_path = self.output_dir.joinpath(filename)
        if not file_path.is_file():
            file = open(file_path, 'w', encoding='utf-8')
        else:
            file = open(file_path, 'a', encoding='utf-8')
        wr = csv.writer(file)
        wr.writerow(
            [task, g_iter, epoch, round(train_loss, 4), round(train_acc, 4), round(test_loss, 4), round(test_acc, 4)])
        file.close()

    def evaluate(self, task_idx):
        # self.load_checkpoint()
        self.set_mode('eval')

        eval_acc = 0
        test_loss = 0
        with torch.no_grad():
            data_loader = self.data_loader['task{}'.format(task_idx)]['test']

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
                test_loss += self.compute_loss(outputs, labels)

                # env_name = self.args.env_name
                # print("##### Env name: {} #####".format(env_name))

                # print("Epoch: {}, iter: {}, test loss: {:.3f}, Test acc.: {:.3f}".format(self.epoch_i, self.global_iter, test_loss, eval_acc))
            eval_acc = eval_acc / (i + 1)
            test_loss = test_loss / (i + 1)
        # reset model to train mode
        self.set_mode('train')
        return test_loss, eval_acc

    def compute_loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)

        # Regularization for all previous tasks
        reg_loss = 0.


        return loss + self.lamb * reg_loss

    def compute_offsets(self, task, nc_per_task, is_incremental):
        """
            Compute offsets for cifar to determine which
            outputs to select for a given task.
        """
        if is_incremental:
            offset1 = task * nc_per_task
            offset2 = (task + 1) * nc_per_task
        else:
            offset1 = 0
            offset2 = nc_per_task
        return offset1, offset2

    def store_grad(self, pp, grads, grad_dims, tid):
        """
            This stores parameter gradients of past tasks.
            pp: parameters
            grads: gradients
            grad_dims: list with number of parameters per layers
            tid: task id
        """
        # store the gradients
        grads[:, tid].fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en, tid].copy_(param.grad.data.view(-1))
            cnt += 1

    def project2cone2(self, gradient, memories, margin=0.5, eps=1e-3):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.
            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector
        """
        memories_np = memories.cpu().t().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.Tensor(x).view(-1, 1))

    def overwrite_grad(self, pp, newgrad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                this_grad = newgrad[beg: en].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            cnt += 1