import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.cnn import Dcnn
from models.incremental_model import IncrementalModel
from utils import cuda


class GemInc(IncrementalModel):
    def __init__(self, args):
        super(GemInc, self).__init__(args)

        self.n_known = 0
        self.n_start = 2
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

        # GPU
        self.cuda = args.cuda

        # allocate episodic memory
        self.memory_data = cuda(torch.zeros([self.M, args.image_size, args.image_size], dtype=torch.float), self.cuda)
        self.memory_labs = cuda(torch.zeros([self.M], dtype=torch.long), self.cuda)
        self.memory_n_per_class = int(self.M / self.n_start)

        # allocate temporary synaptic memory
        self.grad_dims = []
        self.update_grad_dims()
        self.grads = cuda(torch.zeros([sum(self.grad_dims), self.n_tasks]), self.cuda)

        # allocate counters
        self.observed_tasks = []
        self.n_outputs = self.n_start
        self.old_task = -1
        self.cur_task = 0
        self.mem_cnt = 0
        self.nc_per_task = args.gem_inc_num_cls_per_task

    def update_grad_dims(self):
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())

    def forward(self, x):
        output = self.net.forward(x)
        return output

    def get_memory_samples(self, class_begin, class_end):
        sample_begin = class_begin * self.memory_n_per_class
        sample_end = class_end * self.memory_n_per_class
        sample_images = self.memory_data[sample_begin:sample_end]
        sample_labels = self.memory_labs[sample_begin:sample_end]

        return sample_images, sample_labels

    def store_grad(self, past_task):
        self.grads[:, past_task].fill(0.0)
        for param_i, param in enumerate(self.parameters()):
            if param.grad is not None:
                begin = 0 if param_i == 0 else sum(self.grad_dims[:param_i])
                end = sum(self.grad_dims[:param_i + 1])
                self.grads[begin:end, past_task].copy_(param.grad.data.view(-1))

    def update_representation(self, dataset):

        # Increment number of weights in final fc layer
        nodes_to_add = self.n_start + self.cur_task * self.nc_per_task - self.n_outputs
        self.n_outputs = self.increment_classes(nodes_to_add)
        self.update_grad_dims()

        # get dataloader
        loader = DataLoader(dataset, batch_size=self.args.train_batch_size,
                            shuffle=True, num_workers=self.args.num_workers,
                            pin_memory=True, worker_init_fn=self._init_fn)

        # update memory
        # TODO: update memory

        # update params
        for epoch_i in range(self.args.epoch):
            for i, (indices, images, labels, _) in enumerate(loader):
                # compute grad on previous tasks
                if len(self.observed_tasks) > 1:
                    for t_i, past_task in enumerate(self.observed_tasks):
                        self.zero_grad()
                        class_begin = 0 if past_task == 0 else self.n_start - 1 + self.nc_per_task * past_task
                        class_end = self.n_start if past_task == 0 else class_begin + self.nc_per_task * past_task
                        memory_samples, memory_labels = self.get_memory_samples(class_begin, class_end)
                        output = self.forawrd(memory_samples)
                        past_task_loss = self.loss(output, memory_labels)
                        past_task_loss.backward()
                        self.store_grad(past_task)
                # compute grad on current tasks
                # TODO: compute gradients on current tasks
                pass


        # update counters
        self.observed_tasks.append(self.cur_task)
        self.old_task = self.cur_task
        self.cur_task += 1

    def update_exemplar_sets(self):
        pass

    def update_n_known(self):
        pass

    def classify(self, x):
        pass

    def increment_classes(self, n):
        in_features = self.net.fc_layers[-1].in_features
        out_features = self.net.fc_layers[-1].in_features

        weight = self.net.fc_layers[-1].weight.data

        self.net.fc_layers[-1] = nn.Linear(in_features, out_features + n, bias=False)
        self.net.fc_layers[-1].weight.data[:out_features] = weight
        self.n_classes += n

        return out_features + n




