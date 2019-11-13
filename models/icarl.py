import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import return_data
from models.cnn import Dcnn
from models.incremental_model import IncrementalModel
from utils import cuda


class Icarl(IncrementalModel):
    def __init__(self, args):
        super(Icarl, self).__init__(args)
        # self.args = args

        # set_seed(self.args.seed)

        # Network architecture
        self.feature_size = args.icarl_feature_size
        self.feature_extractor = Dcnn(input_channel=1, num_classes=2)

        fc1_features = self.feature_extractor.fc_layers[0].in_features
        fc2_features = self.feature_size

        self.feature_extractor.fc_layers = nn.Sequential(
            nn.Linear(fc1_features, fc2_features)
        )

        self.bn = nn.BatchNorm1d(self.feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(self.feature_size, 1, bias=False)

        self.n_classes = 1
        self.n_known = 0

        # List containing exemplar_sets
        # Each exemplar_set is a np.array of N images
        # with shape (N, C, H, W)
        self.K = args.icarl_K
        self.m = self.K // self.n_classes
        self.exemplar_sets = []
        self.exemplar_path_sets = []

        # Learning method
        self.cls_loss = nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)

        return x

    def increment_classes(self, n):
        """Add n classes in the final fc layer"""
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features + n, bias=False)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

    def classify(self, x):
        if self.args.icarl_K == 0:
            images = cuda(Variable(x), self.args.cuda)
            outputs = F.sigmoid(self.forward(images))
            _, preds = torch.max(outputs, 1)
        else:
            batch_size = x.size(0)
            exemplar_array = np.array(self.exemplar_sets)  # (n_classes, samples_per_class, channel, row, col)
            # print("#### single exemplar set size: {}".format(exemplar_array.shape))

            if self.compute_means:
                exemplar_means = []
                for P_y in self.exemplar_sets:  # P_y: (samples_per_class, channel, row, col)
                    features = []
                    # Extract feature for each exemplar in P_y
                    for ex in P_y:  # ex: (channel, row, col)
                        ex = cuda(torch.tensor(ex), self.args.cuda)
                        feature = self.feature_extractor(ex.unsqueeze(0)) # TODO:feature extractor 제대로 학습하나?
                        feature = feature.squeeze()  # feature: (128)
                        feature.data = feature.data / feature.data.norm()  # Normalize
                        features.append(feature)
                    features = torch.stack(features)
                    mu_y = features.mean(0).squeeze()
                    mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
                    exemplar_means.append(mu_y)  # mu_y: (128)
                self.exemplar_means = exemplar_means  # self.exemplar_means: (n_classes, feature_size)
                self.compute_means = False

            exemplar_means = self.exemplar_means  # self.exemplar_means: (n_classes, feature_size)
            means = torch.stack(exemplar_means)  # (n_classes, feature_size)
            means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
            means = means.transpose(1, 2)  # (batch_size, feature_size, n_classes)
            feature = self.feature_extractor(x)  # (batch_size, feature_size)
            for i in range(feature.size(0)):  # Normalize
                feature.data[i] = feature.data[i] / feature.data[i].norm()
            feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
            feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

            if feature.size(2) > 1:
                dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
            else:
                dists = (feature - means).pow(2).sum(1) # (batch_size, n_classes)
            _, preds = dists.min(1)

        return preds

    def construct_exemplar_set(self, paths, images, m):
        images = torch.tensor(images, dtype=torch.float)
        images = cuda(images, self.args.cuda)
        features = self.feature_extractor.forward(images)
        features.data = features.data / features.data.norm(dim=0)

        class_mean = torch.mean(features, dim=0)
        class_mean = class_mean / class_mean.norm(dim=0)  # Normalize

        exemplar_set = []
        exemplar_path_set = []
        exemplar_features = []  # list of Variables of shape (feature_size,)
        if self.args.icarl_random_example:
            indices = list(range(images.size()[0]))
            indices = np.random.choice(indices, m, replace=False)
            for i in indices:
                exemplar_set.append(np.array(images[i].cpu()))
                exemplar_path_set.append(np.array(paths[i]))
                exemplar_features.append(features[i])
        else:
            for k in range(1, m + 1):
                S = np.sum(exemplar_features, axis=0)
                phi = features
                mu = class_mean
                mu_p = (1.0 / k) * (phi + S)
                mu_p = mu_p / mu_p.norm()
                i = torch.argmin(torch.sqrt(torch.sum((mu - mu_p) ** 2, dim=1)))

                exemplar_set.append(np.array(images[i].cpu()))
                exemplar_path_set.append(np.array(paths[i]))
                exemplar_features.append(features[i])
        self.exemplar_sets.append(np.array(exemplar_set))
        self.exemplar_path_sets.append(np.array(exemplar_path_set))

    def update_exemplar_sets(self):
        self.m = self.K // self.n_classes
        if self.K != 0:
            self.reduce_exemplar_sets(self.m)

        # Construct exemplar sets for new classes
        for y in range(self.n_known, self.n_classes):
            print("Constructing exemplar set for class-{}...".format(y + 1))
            y_class_dataloader, _, _ = return_data(self.args, class_range=range(y, y + 1))
            images_concat = None
            paths_concat = None
            for indices, images, _, paths in y_class_dataloader['train']:
                if images_concat is None:
                    images_concat = images
                    paths_concat = paths
                else:
                    images_concat = np.concatenate((images_concat, images))
                    paths_concat = np.concatenate((paths_concat, paths))
            self.construct_exemplar_set(paths_concat, images_concat, self.m)
            print("Done")

        for y, P_y in enumerate(self.exemplar_sets):
            print("Exemplar set for class-{}: {}".format(y, P_y.shape))

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]
        for y, P_y in enumerate(self.exemplar_path_sets):
            self.exemplar_path_sets[y] = P_y[:m]

    def combine_dataset_with_exemplars(self, dataset):
        for y, P_y in enumerate(self.exemplar_path_sets):
            exemplar_image_path = P_y
            exemplar_labels = [y] * len(P_y)
            dataset.append(list(zip(exemplar_image_path, exemplar_labels)), exemplar_labels)

    def _init_fn(self, worker_id):
        np.random.seed(int(self.args.seed))

    def update_representation(self, dataset, current_class_count, train_loader, test_loader, line_plotter):
        self.compute_means = True

        # Increment number of weights in final fc layer
        classes = range(current_class_count)
        new_classes = [cls for cls in classes if cls > self.n_classes - 1]
        self.increment_classes(len(new_classes))
        if self.args.cuda:
            self.cuda()
        print("known classes count: {}".format(self.n_known))
        print("total {} new_classes: {}".format(len(new_classes), new_classes))

        # Form combined training set
        self.combine_dataset_with_exemplars(dataset)

        loader = DataLoader(dataset, batch_size=self.args.train_batch_size,
                            shuffle=True, num_workers=self.args.num_workers,
                            pin_memory=True, worker_init_fn=self._init_fn)

        # Store network outputs with pre-update parameters
        q = cuda(torch.zeros(len(dataset), self.n_classes), self.args.cuda)
        for indices, images, labels, _ in loader:
            images = cuda(Variable(images), self.args.cuda)
            indices = cuda(indices, self.args.cuda)
            g = F.sigmoid(self.forward(images))
            q[indices] = g.data
        q = cuda(Variable(q), self.args.cuda)

        # Run network training
        optimizer = self.optimizer

        iteration = 0
        for epoch_i in range(self.args.epoch):
            for i, (indices, images, labels, _) in enumerate(loader):
                images = cuda(Variable(images), self.args.cuda)
                labels = cuda(Variable(torch.tensor(labels, dtype=torch.uint8)), self.args.cuda)
                indices = cuda(indices, self.args.cuda)

                optimizer.zero_grad()
                g = self.forward(images)

                # Classification loss for new classes
                train_loss = self.cls_loss(g, labels.type(torch.long))
                # loss = loss / len(range(self.n_known, self.n_classes))

                dist_loss = None
                # Distilation loss for old classes
                if self.n_known > 0 and self.args.icarl_K != 0:
                    g = F.sigmoid(g)
                    q_i = q[indices]
                    dist_loss = sum(self.dist_loss(g[:, y], q_i[:, y]) for y in range(self.n_known))
                    # dist_loss = dist_loss / self.n_known
                    train_loss += dist_loss

                train_loss.backward()

                if self.args.icarl_fixed_rep and current_class_count > 2:
                    if epoch_i == 0:
                        for param in self.feature_extractor.fc_layers[-1].parameters():
                            param.requires_grad = True

                    else:
                        for param in self.feature_extractor.fc_layers[-1].parameters():
                            param.requires_grad = False

                    fc_grad = self.fc.weight.grad
                    mask = torch.ones_like(fc_grad)
                    mask[-1:,:] = 0
                    fc_grad *= mask
                    self.fc.weight.grad = fc_grad

                optimizer.step()

                if (i + 1) % 5 == 0:
                    print('Epoch [{}/{}], Iter [{}/{}] Loss: {}, Dist_loss: {}, known class: {}, new class: {}'.format(
                        epoch_i + 1, self.args.epoch, i + 1, len(dataset) // self.args.train_batch_size, train_loss.item(),
                        dist_loss, self.n_known, len(new_classes)
                    ))

                    if self.args.visdom and self.args.icarl_K==0:
                        total = 0.0
                        correct = 0.0
                        for indices, images, labels, _ in train_loader:
                            images = cuda(Variable(images), self.args.cuda)
                            preds = self.classify(images)
                            total += labels.size(0)
                            correct += (preds.data.cpu() == labels.data.cpu()).sum()
                        train_acc = float(correct) / float(total)

                        total = 0.0
                        correct = 0.0
                        for indices, images, labels, _ in test_loader:
                            images = cuda(Variable(images), self.args.cuda)
                            preds = self.classify(images)
                            total += labels.size(0)
                            correct += (preds.data.cpu() == labels.data.cpu()).sum()
                        test_acc = float(correct) / float(total)

                        line_plotter.plot(var_name='loss',
                                          split_name='train {} class'.format(current_class_count),
                                          title_name=self.args.date + ' Task Loss',
                                          x=iteration,
                                          y=train_loss.item())
                        line_plotter.plot(var_name='acc.',
                                          split_name='train {} class'.format(current_class_count),
                                          title_name=self.args.date + ' Task Accuracy',
                                          x=iteration,
                                          y=train_acc)
                        line_plotter.plot(var_name='acc.',
                                          split_name='test {} class'.format(current_class_count),
                                          title_name=self.args.date + ' Task Accuracy',
                                          x=iteration,
                                          y=test_acc)
                iteration += 1

    def update_n_known(self):
        self.n_known = self.n_classes
        print("iCaRL classes: {}".format(self.n_known))
        print("iCaRL model last nodes: {}".format(self.fc.out_features))
