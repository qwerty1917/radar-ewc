import numpy as np
import quadprog
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import IcarlDataset
from models.cnn import Dcnn
from models.incremental_model import IncrementalModel
from utils import cuda


class GemInc(IncrementalModel):
    def __init__(self, args):
        super(GemInc, self).__init__(args)

        self.n_known = 0
        self.n_start = 2
        self.n_classes = self.n_start
        self.n_tasks = args.gem_inc_n_tasks

        self.M = args.gem_inc_M
        self.m = self.M / self.n_start
        self.margin = args.gem_inc_mem_strength

        # Network architecture
        self.net = Dcnn(input_channel=1, num_classes=2)

        # Loss
        self.loss = nn.CrossEntropyLoss()

        # Opt
        self.opt = optim.SGD(self.parameters(), lr=args.lr)

        # allocate episodic memory
        self.memory_data = cuda(torch.zeros([self.M, args.channel, args.image_size, args.image_size], dtype=torch.float), self.args.cuda)
        self.memory_labs = cuda(torch.zeros([self.M], dtype=torch.long), self.args.cuda)
        self.memory_n_per_task = self.M

        # allocate temporary synaptic memory
        self.grad_dims = []
        self.update_grad_dims_and_grads()
        self.grads = None

        # allocate counters
        self.observed_tasks = []
        self.n_outputs = self.n_start
        self.old_task = -1
        self.cur_task = 0
        self.nc_per_task = args.gem_inc_num_cls_per_task

        # GPU
        if self.args.cuda and torch.cuda.is_available():
            print("# CUDA available.")
            self.net.cuda()
            self.device = torch.device("cuda:0")
            self.net = self.net.to(self.device)
        print("# device: {}".format(self.device))

    def update_grad_dims_and_grads(self):
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = cuda(torch.zeros([sum(self.grad_dims), self.n_tasks]), self.args.cuda)

    def forward(self, x):
        x = cuda(x, self.args.cuda).to(self.device)
        output = self.net.forward(x)
        return output

    def get_memory_samples(self, class_begin, class_end):
        sample_begin = class_begin * self.memory_n_per_task
        sample_end = class_end * self.memory_n_per_task
        sample_images = self.memory_data[sample_begin:sample_end]
        sample_labels = self.memory_labs[sample_begin:sample_end]

        return sample_images, sample_labels

    def store_grad(self, past_task):
        self.grads[:, past_task].fill_(0.0)
        for param_i, param in enumerate(self.parameters()):
            if param.grad is not None:
                begin = 0 if param_i == 0 else sum(self.grad_dims[:param_i])
                end = sum(self.grad_dims[:param_i + 1])
                self.grads[begin:end, past_task].copy_(param.grad.data.view(-1))

    def overwrite_grad(self, newgrad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in self.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                this_grad = newgrad[beg: en].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
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

    def _init_fn(self, worker_id):
        np.random.seed(int(self.args.seed))

    def update_representation(self, dataset, current_class_count, train_loader, test_loader, line_plotter):

        # Increment number of weights in final fc layer
        nodes_to_add = self.n_start + self.cur_task * self.nc_per_task - self.n_outputs
        self.n_outputs = self.increment_classes(nodes_to_add)
        self.update_grad_dims_and_grads()
        if self.args.cuda:
            self.cuda()

        # get dataloader
        current_task_loader = DataLoader(dataset, batch_size=self.args.train_batch_size,
                            shuffle=True, num_workers=self.args.num_workers,
                            pin_memory=True, worker_init_fn=self._init_fn)

        # update memory
        self.update_exemplar_sets(dataset)

        # update params
        for epoch_i in range(self.args.epoch):
            # compute grad on previous tasks
            if len(self.observed_tasks) > 0:
                for t_i, past_task in enumerate(self.observed_tasks):
                    self.zero_grad()
                    task_begin = 0 if past_task == 0 else self.n_start - 1 + self.nc_per_task * past_task
                    task_end = self.n_start if past_task == 0 else task_begin + self.nc_per_task * past_task
                    memory_samples, memory_labels = self.get_memory_samples(task_begin, task_end)
                    memory_samples = cuda(Variable(memory_samples), self.args.cuda)
                    memory_labels = cuda(Variable(memory_labels), self.args.cuda)
                    output = self.forward(memory_samples)
                    past_task_loss = self.loss(output, memory_labels)
                    past_task_loss.backward()
                    self.store_grad(past_task)
            else:
                print("# no observed tasks yet")

            # compute grad on current tasks
            self.zero_grad()
            for i, (indices, images, labels, _) in enumerate(current_task_loader):
                # compute gradients on current tasks
                images = cuda(Variable(images), self.args.cuda)
                labels = cuda(Variable(labels), self.args.cuda)

                output = self.forward(images)
                train_loss = self.loss(output, labels.type(torch.long))

                train_loss.backward()

                # check if gradient violated constraints
                if len(self.observed_tasks) > 0:
                    # copy gradient
                    self.store_grad(self.cur_task)
                    indx = torch.cuda.LongTensor(self.observed_tasks) if self.args.cuda \
                        else torch.LongTensor(self.observed_tasks)
                    dopt = torch.mm(self.grads[:,self.cur_task].unsqueeze(0),
                                    self.grads.index_select(1, indx))

                    if (dopt < 0).sum() != 0:
                        self.project2cone2(self.grads[:, self.cur_task].unsqueeze(1),
                                           self.grads.index_select(1, indx),
                                           self.margin)
                        # copy gradients back
                        self.overwrite_grad(self.grads[:, self.cur_task],
                                            self.grad_dims)

                self.opt.step()

        # update counters
        self.observed_tasks.append(self.cur_task)
        self.old_task = self.cur_task
        self.cur_task += 1

    def update_exemplar_sets(self, dataset: IcarlDataset):
        old_sample_n_per_task = self.memory_n_per_task
        new_sample_n_per_task = self.M // (self.cur_task + 1)

        new_exemplar_data = cuda(torch.zeros_like(self.memory_data), self.args.cuda)
        new_exemplar_labs = cuda(torch.zeros_like(self.memory_labs), self.args.cuda)

        for old_task_i in self.observed_tasks:
            for sample_i in range(new_sample_n_per_task):
                old_sample_i = old_task_i * old_sample_n_per_task + sample_i
                new_sample_i = old_task_i * new_sample_n_per_task + sample_i

                new_exemplar_data[new_sample_i] = self.memory_data[old_sample_i]
                new_exemplar_labs[new_sample_i] = self.memory_labs[old_sample_i]

        new_task_sample_cnt = 0

        for index, image, label, path in dataset:
            if new_task_sample_cnt < new_sample_n_per_task:
                new_exemplar_data[len(self.observed_tasks) * new_sample_n_per_task + new_task_sample_cnt] = image
                new_exemplar_labs[len(self.observed_tasks) * new_sample_n_per_task + new_task_sample_cnt] = label
            else:
                break

            new_task_sample_cnt += 1

        self.memory_data = new_exemplar_data
        self.memory_labs = new_exemplar_labs

        self.memory_n_per_task = new_sample_n_per_task
        print("Exemplar set updated.")

    def update_n_known(self):
        self.n_known = self.n_classes
        print("GEM icr classes: {}".format(self.n_known))
        print("GEM icr model last nodes: {}".format(self.net.fc_layers[-1].out_features))

    def classify(self, x):
        images = cuda(Variable(x), self.args.cuda)
        outputs = F.sigmoid(self.forward(images))
        _, preds = torch.max(outputs, 1)

        return preds

    def increment_classes(self, n):
        in_features = self.net.fc_layers[-1].in_features
        out_features = self.net.fc_layers[-1].out_features

        weight = self.net.fc_layers[-1].weight.data

        self.net.fc_layers[-1] = nn.Linear(in_features, out_features + n, bias=False)
        self.net.fc_layers[-1].weight.data[:out_features] = weight
        self.n_classes += n

        return out_features + n




