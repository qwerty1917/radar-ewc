import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import Dcnn
from models.incremental_model import IncrementalModel
from utils import cuda


class GemInc(IncrementalModel):
    def __init__(self, args):
        super(GemInc, self).__init__(args)

        self.n_known = 0
        self.n_start = 2

        self.M = args.gem_inc_M
        self.m = self.M / self.n_start
        self.margin = args.gem_inc_mem_strength

        # Network architecture
        self.net = Dcnn(input_channel=1, num_classes=2)
        self.n_outputs = self.n_start

        # Loss
        self.loss = nn.CrossEntropyLoss()

        # Opt
        self.opt = optim.SGD(self.parameters(), lr=args.lr)

        # GPU
        self.cuda = args.cuda

        # allocate episodic memory
        self.memory_data = cuda(torch.zeros([self.M, args.image_size, args.image_size], dtype=torch.float), self.cuda)
        self.memory_labs = cuda(torch.zeros([self.M], dtype=torch.long), self.cuda)

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = cuda(torch.zeros([sum(self.grad_dims), self.n_outputs]), self.cuda)

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.nc_per_task = args.gem_inc_num_cls_per_task

    def forward(self, x, t):
        output = self.net.forward(x)
        return output

    def update_representation(self, dataset):
        # https://github.com/facebookresearch/GradientEpisodicMemory/blob/34c6b8e9a0607db7567301c48b727430d20bee7e/model/gem.py
        # line 153 부터
        pass

    def update_exemplar_sets(self):
        pass

    def update_n_known(self):
        pass

    def classify(self, x):
        pass
