from pathlib import Path

import torch
from torch import nn, optim, randn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from models.wgan import Discriminator, Generator
from utils import VisdomImagesPlotter, cuda


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


class WGAN(object):
    def __init__(self, args=None, line_plotter=None, images_plotter=None, sample_num: int=100):

        self.args = args

        # optimization
        self.cuda = args.cuda
        self.G_optim = None
        self.D_optim = None
        self.batch_size = args.train_batch_size
        self.current_batch_size = 0
        self.D_lr = args.gan_D_lr
        self.G_lr = args.gan_G_lr
        self.d_iters = args.gan_d_iters
        self.g_iters = args.gan_g_iters
        self.gp_lambda = args.gan_gp_lambda
        self.global_iter = 0
        self.current_g_iters = 0
        self.D_loss = 0
        self.Wasserstein_D = 0

        # visualization
        self.date = args.date
        self.env_name = args.env_name
        self.visdom = args.visdom
        self.visdom_port = args.visdom_port
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        self.output_dir = Path(args.output_dir).joinpath(args.env_name)
        self.sample_num = args.gan_sample_num
        self.vis_plotter = line_plotter
        self.images_plotter = images_plotter

        # network
        self.G = None
        self.D = None
        self.d_residual = args.gan_d_residual
        self.g_residual = args.gan_g_residual
        self.input_channel = args.channel
        self.multi_gpu = args.gan_multi_gpu
        self.model_init()
        self.fixed_z = cuda(self.sample_z(batch_size=sample_num), self.cuda)

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.G.train()
            self.D.train()
        elif mode == 'eval':
            self.G.eval()
            self.D.eval()

    def model_init(self):
        self.D = Discriminator(self.input_channel, residual=self.d_residual)
        self.G = Generator(self.input_channel, residual=self.g_residual)

        self.D.apply(weights_init)
        self.G.apply(weights_init)

        self.D_optim = optim.RMSprop(self.D.parameters(), lr=self.D_lr)
        self.G_optim = optim.RMSprop(self.G.parameters(), lr=self.G_lr)

        self.D = cuda(self.D, self.cuda)
        self.G = cuda(self.G, self.cuda)

        if self.multi_gpu:
            self.D = nn.DataParallel(self.D).cuda()
            self.G = nn.DataParallel(self.G).cuda()

    def sample_z(self, batch_size: int=None):
        if batch_size is None:
            return randn(self.batch_size, 100)
        else:
            return randn(batch_size, 100)

    def unscale(self, tensor):
        return tensor.mul(0.5).add(0.5)

    def sample_img(self, _type='fixed', nrow=10):
        self.set_mode('eval')

        if _type == 'fixed':
            z = self.fixed_z
        elif _type == 'random':
            z = self.sample_z(self.sample_num)
            z = Variable(cuda(z, self.cuda))
        else:
            raise ValueError('_type must be one of fixed or random')

        z = torch.unsqueeze(torch.unsqueeze(z, -1), -1)

        # samples = self.unscale(self.G(z))
        samples = self.G(z)
        samples = samples.data.cpu()

        self.set_mode('train')
        return samples

    def plot_visdom_images(self, caption: str, images):
        self.images_plotter.draw(caption=caption, images=images)

    def save_images(self, images, nrow=10):
        filename = self.output_dir.joinpath('Task_{}_g_iter_{}_iteration_{}.jpg'.format(self.args.task_idx + 1,
                                                                                       self.current_g_iters,
                                                                                       self.global_iter))
        grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
        save_image(grid, filename=filename)

    def replay(self, replay_n: int):
        self.set_mode('eval')

        z = self.sample_z(replay_n)
        z = Variable(cuda(z, self.cuda))

        z = torch.unsqueeze(torch.unsqueeze(z, -1), -1)

        replays = self.G(z)

        # replays = self.unscale(replays)

        return replays

    def save_model(self, filename='scholar.tar'):
        model_states = {'G': self.G.state_dict(),
                        'D': self.D.state_dict()}

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(model_states, file_path.open('wb+'))
        print("=> saved scholar model '{}'".format(file_path))

    def load_model(self, filename='scholar.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            model_states = torch.load(file_path.open('rb'))
            self.G.load_state_dict(model_states['G'])
            self.D.load_state_dict(model_states['D'])
            print("=> loaded scholar model '{}'".format(file_path))
        else:
            print("=> no scholar model found at '{}'".format(file_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.current_batch_size, 1, 1, 1)
        alpha = cuda(alpha, self.cuda)
        interpolates = torch.mul(alpha, real_data) + torch.mul((1 - alpha), fake_data)
        interpolates = cuda(interpolates, self.cuda)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=cuda(torch.ones(disc_interpolates.size()), self.cuda),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
        return gradient_penalty

    def train(self, data_loader: DataLoader, task_num: int):
        self.set_mode('train')

        self.data = self.get_infinite_batches(data_loader)

        one = torch.FloatTensor([1])
        mone = one * -1

        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()

        for g_iter in range(self.g_iters):

            ## Discriminator training
            for p in self.D.parameters():
                p.requires_grad = True

            # for p in self.G.parameters():
            #     p.requires_grad = False

            for d_iter in range(self.d_iters):
                self.global_iter += 1
                self.D.zero_grad()

                images = self.data.__next__()

                self.current_batch_size = images.size()[0]
                batch_size_ratio = self.current_batch_size / self.batch_size

                images = Variable(cuda(images, self.cuda))

                # Discriminator Training with real image
                x_real = Variable(cuda(images, self.cuda))
                D_loss_real = self.D(x_real)
                D_loss_real = D_loss_real.mean(0).view(1)
                D_loss_real = torch.mul(D_loss_real, batch_size_ratio)  # batch size 에 맞추어 loss 조정
                D_loss_real.backward(one)

                # Discriminator Training with fake image
                z = Variable(torch.randn(self.current_batch_size, 100, 1, 1))
                z = cuda(z, self.cuda)

                x_fake = self.G(z)
                D_loss_fake = self.D(x_fake)
                D_loss_fake = D_loss_fake.mean(0).view(1)
                D_loss_fake = torch.mul(D_loss_fake, batch_size_ratio)   # batch size 에 맞추어 loss 조정
                D_loss_fake.backward(mone)

                gradient_penalty = self.calc_gradient_penalty(x_real.data, x_fake.data)
                gradient_penalty.backward()

                self.D_loss = D_loss_fake - D_loss_real + gradient_penalty
                self.Wasserstein_D = D_loss_real - D_loss_fake

                self.D_optim.step()

            ## Generator training
            for p in self.D.parameters():
                p.requires_grad = False

            # for p in self.G.parameters():
            #     p.requires_grad = True

            self.G.zero_grad()

            # Generator update
            # Compute loss with fake images
            z = Variable(torch.randn(self.batch_size, 100, 1, 1))
            z = cuda(z, self.cuda)

            x_fake = self.G(z)
            G_loss = self.D(x_fake)
            G_loss = G_loss.mean().mean(0).view(1)
            G_loss.backward(one)
            G_cost = -G_loss
            self.G_optim.step()

            # Visdom
            if self.visdom:
                self.vis_plotter.draw("{} task {} WGAN training x_real images".format(self.date, task_num+1), x_real)
                self.vis_plotter.draw("{} task {} WGAN training x_fake images".format(self.date, task_num+1), x_fake)
                sample_images = self.sample_img(_type='fixed')
                self.vis_plotter.draw("{} task {} WGAN training Sample images".format(self.date, task_num+1), sample_images)
                self.vis_plotter.plot(var_name='W distance',
                                      split_name='task {}'.format(task_num+1),
                                      title_name='{} W distance gp lamb {}'.format(self.date, self.gp_lambda),
                                      x=g_iter,
                                      y=self.Wasserstein_D.cpu().data[0])

            # Console output
            if self.global_iter % 10 == 0:
                print('generator_iter:{:d}, global_iter:{:d}'.format(g_iter, self.global_iter))
                print('W distance: {:.3f}, Loss D: {:.3f}, Loss G: {:.3f}, Loss D Real: {:.3f}, Loss D fake: {:.3f}'.
                      format(self.Wasserstein_D.data[0], self.D_loss.data[0], G_cost.data[0], D_loss_real.data[0],
                             D_loss_fake.data[0]))

        # Visdom
        if self.visdom:
            self.images_plotter.draw("{} WGAN generated image task:{}".format(self.date, task_num+1), self.sample_img(_type='fixed'))






