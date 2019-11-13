from abc import *

import torch.nn as nn

from utils import cuda, set_seed


class IncrementalModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args):
        super(IncrementalModel, self).__init__()
        self.args = args

        set_seed(self.args.seed)

        @abstractmethod
        def update_representation(self, *a, **ka):
            pass

        @abstractmethod
        def update_exemplar_sets(self, *a, **ka):
            pass

        @abstractmethod
        def update_n_known(self, *a, **ka):
            pass

        @abstractmethod
        def classify(self, *a, **ka):
            pass
