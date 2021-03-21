import math
import os
import time
from abc import ABC, abstractmethod
from timeit import default_timer as timer

import numpy as np
from tensorboardX import SummaryWriter

from invnet import calc_gradient_penalty, \
    weights_init
from models.wgan import *


class BaseInvNet(ABC):
    '''An abstract class implementing functionality that is common to all InvNet classes. This class
    doesn't specify any particular data source or data type as this is left to inheriting classes.

    Arguments:
            batch_size (int): size of training batches
            output_path (str): the location to save trained models
            data_dir (str): location of training data
            lr (float): learning rate for all models
            critic_iters (int): number of updates performed for the critic in each training iteration
            proj_iters (int): number of updates performed for the critic in each training iteration
            output_dim (int): dimension of generator output (same as size of each datapoint)
            hidden_size (int): depth of largest hidden layers in generator and critic
            device (str): name of device to do training on




    '''
    def __init__(self, batch_size, output_path, data_dir, lr, critic_iters, proj_iters, output_dim, hidden_size, device, lambda_gp,ctrl_dim,proj_lambda=1,restore_mode=False,hparams={}):
        self.writer = SummaryWriter()
        print('output dir:', self.writer.logdir)
        self.device = device

        self.data_dir = data_dir
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.batch_size = batch_size
        self.output_dim = output_dim
        self.lambda_gp = lambda_gp

        self.train_loader, self.val_loader = self.load_data()
        self.dataiter, self.val_iter = iter(self.train_loader), iter(self.val_loader)

        self.proj_lambda=proj_lambda
        self.critic_iters = critic_iters
        self.proj_iters = proj_iters
        hparams.update({'proj_iters': self.proj_iters,
                      'critic_iters': self.critic_iters,
                      'proj_lambda': self.proj_lambda})
        self.hparams=hparams


        if restore_mode:
            self.D = torch.load(output_path + "generator.pt").to(device)
            self.G = torch.load(output_path + "discriminator.pt").to(device)
        else:
            self.G = GoodGenerator(hidden_size, self.output_dim, ctrl_dim=ctrl_dim).to(device)
            self.D = GoodDiscriminator(dim=hidden_size).to(device)
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.optim_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0, 0.9))
        self.optim_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0, 0.9))
        self.optim_pj = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0, 0.9))

        self.fixed_noise = self.gen_rand_noise(4)
        self.p1_mean, self.p1_std = None,None
        self.p1_mean, self.p1_std = self.get_p1_stats()

        self.start = timer()

    def train(self, iters):

        for iteration in range(iters):
            print('iteration:', iteration)

            gen_cost, real_p1 = self.generator_update()
            start_time = time.time()
            proj_cost = self.proj_update()
            stats = self.critic_update()
            add_stats = {'start': start_time,
                         'iteration': iteration,
                         'gen_cost': gen_cost,
                         'proj_cost': proj_cost}
            stats.update(add_stats)
            if iteration%5==0:
                stats['val_proj_err'], stats['val_critic_err'] = self.validation()
                self.log(stats)
            if iteration % 50 == 0:
                self.save(stats)

    def generator_update(self):
        start=timer()
        for p in self.D.parameters():
            p.requires_grad_(False)

        real_data= self.sample()
        real_images=real_data.to(self.device)
        with torch.no_grad():
            real_lengths=self.real_p1(real_images)
        real_p1=real_lengths.to(self.device)
        mone = torch.FloatTensor([1]) * -1
        mone=mone.to(self.device)

        for i in range(1):
            self.G.zero_grad()
            noise = self.gen_rand_noise(self.batch_size).to(self.device)
            noise.requires_grad_(True)
            fake_data = self.G(noise, real_p1)
            fake_data = self.format_data(fake_data)
            gen_cost = self.D(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost = gen_cost.view((1))
            gen_cost.backward(mone)
            gen_cost = -gen_cost

            self.optim_g.step()

        end=timer()
        # print('--generator update elapsed time:',end-start)
        return gen_cost, real_p1

    def critic_update(self):
        for p in self.D.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        start = timer()
        for i in range(self.critic_iters):
            self.D.zero_grad()
            real_images = self.sample().to(self.device)
            # gen fake data and load real data
            noise = self.gen_rand_noise(self.batch_size).to(self.device)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
                real_lengths= self.real_p1(real_images)
                real_p1 = real_lengths.to(self.device)
            fake_data = self.G(noisev, real_p1).detach()
            # train with real data
            disc_real = self.D(real_images)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = self.D(fake_data)
            disc_fake = disc_fake.mean()

            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(self.D, real_images, fake_data, self.batch_size, self.lambda_gp,int(math.sqrt(self.output_dim)))

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake - disc_real

            self.optim_d.step()
        end = timer()
        # print('---train D elapsed time:', end - start)
        stats={'w_dist': w_dist,
               'disc_cost':disc_cost,
               'fake_data':fake_data[:100],
               'real_data':real_images[:4],
               'disc_real':disc_real,
               'disc_fake':disc_fake,
               'gradient_penalty':gradient_penalty,
               'real_p1_avg':real_p1.mean(),
                'real_p1_std':real_p1.std()}
        return stats

    def proj_update(self):
        if not (self.proj_lambda and self.proj_iters):
            return 0
        start=timer()
        real_data = self.sample()
        total_pj_loss=torch.tensor([0.])
        with torch.no_grad():
            images = real_data.to(self.device)
            real_lengths = self.real_p1(images).view(-1, 1)
        for iteration in range(self.proj_iters):
            self.G.zero_grad()
            noise=self.gen_rand_noise(self.batch_size).to(self.device)
            noise.requires_grad=True
            fake_data = self.G(noise, real_lengths).view((self.batch_size,self.max_i,self.max_j))
            pj_loss=self.proj_lambda*self.proj_loss(fake_data,real_lengths)
            pj_loss.backward()
            total_pj_loss+=pj_loss.cpu()
            self.optim_pj.step()

        end=timer()
        return total_pj_loss/self.proj_iters

    def validation(self):
        proj_errors = []
        dev_disc_costs = []
        for batch in range(3):
            images=self.sample(train=False)
            if isinstance(images,list):
                images=images[0]
            imgs = torch.Tensor(images)
            imgs = imgs.to(self.device).squeeze()
            with torch.no_grad():
                imgs_v = imgs
                real_lengths = self.real_p1(imgs_v)
                noise = self.gen_rand_noise(real_lengths.shape[0]).to(self.device)
                fake_data = self.G(noise, real_lengths.to(self.device)).detach()
                _proj_err = self.proj_loss(fake_data, real_lengths).detach()
                D = self.D(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
            proj_errors.append(_proj_err)
            dev_disc_costs.append(_dev_disc_cost)
        dev_disc_cost = np.mean(dev_disc_costs)
        proj_error = sum(proj_errors)/(len(proj_errors)*self.batch_size)
        return proj_error, dev_disc_cost

    def log(self,stats):
        # ------------------VISUALIZATION----------
        self.writer.add_scalar('data/gen_cost', stats['gen_cost'], stats['iteration'])
        self.writer.add_scalar('data/disc_cost', stats['disc_cost'], stats['iteration'])
        self.writer.add_scalar('data/disc_fake', stats['disc_fake'], stats['iteration'])
        self.writer.add_scalar('data/disc_real', stats['disc_real'], stats['iteration'])
        self.writer.add_scalar('data/gradient_pen', stats['gradient_penalty'], stats['iteration'])
        self.writer.add_scalar('data/proj_error',stats['val_proj_err'],stats['iteration'])

    def gen_rand_noise(self,batch_size=None):
        if batch_size is None:
            batch_size=self.batch_size
        noise = torch.randn((batch_size, 128))
        noise = noise.to(self.device)
        return noise

    def format_data(self,data):
        return data

    def sample(self,train=True):
        if train:
            try:
                real_data = next(self.dataiter)
            except:
                self.dataiter = iter(self.train_loader)
                real_data = self.dataiter.next()
            if isinstance(real_data, list):
                real_data = real_data[0]

            if real_data.shape[0] < self.batch_size:
                real_data = self.sample()
        else:
            try:
                real_data = next(self.val_iter)
            except:
                self.val_iter = iter(self.val_loader)
                real_data = self.val_iter.next()
            if isinstance(real_data, list):
                real_data = real_data[0]
            if real_data.shape[0] < self.batch_size:
                real_data = self.sample(train=False)
        return real_data.squeeze()

    def get_p1_stats(self):
        p1_values=[]
        for _ in range(10):
            batch=self.sample()
            with torch.no_grad():
                p1=self.real_p1(batch)
            p1_values+=list(p1)
        values=np.array(p1_values)
        return values.mean(),values.std()

    def normalize_p1(self,p1):
        return (p1-self.p1_mean)/self.p1_std

    @abstractmethod
    def save(self, stats):
        pass

    @abstractmethod
    def proj_loss(self,fake_data,real_p1):
        pass

    @abstractmethod
    def real_p1(self,images):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def norm_data(self,data):
        pass

