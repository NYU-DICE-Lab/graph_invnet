import math
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from models.wgan import *
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import time
import os
from graph.utils import calc_gradient_penalty,gen_rand_noise,\
                        weights_init,generate_image
from config import InvNetConfig
import libs as lib
import libs.plot
import numpy as np

class InvNet:

    def __init__(self,batch_size,output_path,lr,critic_iters,proj_iters,output_dim,device,lambda_gp,restore_mode=False):

        self.writer = SummaryWriter()
        self.device=device

        self.output_path=output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.batch_size=batch_size
        self.output_dim=output_dim
        self.lambda_gp=lambda_gp

        self.train_loader,self.val_loader=self.load_data()
        self.dataiter,self.val_iter=iter(self.train_loader),iter(self.val_loader)

        self.critic_iters=critic_iters
        self.proj_iters=proj_iters

        if restore_mode:
            self.D = torch.load(output_path + "generator.pt").to(device)
            self.G = torch.load(output_path + "discriminator.pt").to(device)
        else:
            self.G = GoodGenerator(64, self.output_dim, ctrl_dim=10).to(device)
            self.D = GoodDiscriminator(64).to(device)
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.optim_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0, 0.9))
        self.optim_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0, 0.9))

        self.fixed_noise=gen_rand_noise(self.batch_size)

    def load_data(self):
        data_transform = transforms.Compose([
            transforms.Resize(128),
            # transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        mnist_data = datasets.MNIST('/Users/kellymarshall/PycharmProjects/graph_invnet/files/', download=True,
                                    transform=data_transform)
        train_data,val_data=torch.utils.data.random_split(mnist_data, [55000,5000])

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader=torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        images, _ = next(iter(train_loader))
        return train_loader,val_loader

    def generator_update(self):
        for p in self.D.parameters():
            p.requires_grad_(False)

        real_data= self.sample()
        real_class = F.one_hot(torch.tensor(real_data[1]), num_classes=10)
        real_class = real_class.float()
        real_p1 = real_class.to(self.device)
        mone = torch.FloatTensor([1]) * -1
        mone=mone.to(self.device)
        for i in range(1):
            print("Generator iters: " + str(i))
            self.G.zero_grad()
            noise = gen_rand_noise(self.batch_size).to(device)
            noise.requires_grad_(True)
            fake_data = self.G(noise, real_p1)
            gen_cost = self.D(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost = gen_cost.view((1))
            gen_cost.backward(mone)
            gen_cost = -gen_cost

        self.optim_g.step()

        return gen_cost, real_p1

    def critic_update(self):
        for p in self.D.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(self.critic_iters):
            print("Critic iter: " + str(i))

            start = timer()
            self.D.zero_grad()
            real_data = self.sample()
            # gen fake data and load real data
            noise = gen_rand_noise(self.batch_size).to(self.device)

            # batch = batch[0] #batch[1] contains labels
            real_images = real_data[0].to(self.device)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
                real_class = F.one_hot(torch.tensor(real_data[1]), num_classes=10).to(self.device)
                real_class = real_class.float()
                real_p1 = real_class.to(self.device)
            end = timer()
            print(f'---gen G elapsed time: {end - start}')
            start = timer()
            fake_data = self.G(noisev, real_p1).detach()
            end = timer()
            print(f'---load real imgs elapsed time: {end - start}')
            start = timer()

            # train with real data
            disc_real = self.D(real_images)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = self.D(fake_data)
            disc_fake = disc_fake.mean()

            # train with interpolates data
            gradient_penalty = calc_gradient_penalty(self.D, real_images, fake_data, self.batch_size, self.lambda_gp)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake - disc_real

            self.optim_d.step()
        end = timer()
        print(f'---train D elapsed time: {end - start}')
        stats={'w_dist': w_dist,
               'disc_cost':disc_cost,
               'fake_data':fake_data,
               'disc_real':disc_real,
               'disc_fake':disc_fake,
               'gradient_penalty':gradient_penalty}
        return stats

    def sample(self):
        try:
            real_data = next(self.dataiter)
        except:
            self.dataiter = iter(self.train_loader)
            real_data = self.dataiter.next()
        return real_data

    def log(self,stats):
        # ------------------VISUALIZATION----------
        self.writer.add_scalar('data/gen_cost', stats['gen_cost'], stats['iteration'])
        self.writer.add_scalar('data/disc_cost', stats['disc_cost'], stats['iteration'])
        self.writer.add_scalar('data/disc_fake', stats['disc_fake'], stats['iteration'])
        self.writer.add_scalar('data/disc_real', stats['disc_real'], stats['iteration'])
        self.writer.add_scalar('data/gradient_pen', stats['gradient_penalty'], stats['iteration'])

        lib.plot.plot(self.output_path + 'time', time.time() - stats['start'])
        lib.plot.plot(self.output_path + 'train_disc_cost', stats['disc_cost'].cpu().data.numpy())
        lib.plot.plot(self.output_path + 'train_gen_cost', stats['gen_cost'].cpu().data.numpy())
        lib.plot.plot(self.output_path + 'wasserstein_distance', stats['w_dist'].cpu().data.numpy())

    def save(self,stats):
        size=int(math.sqrt(self.output_dim))
        fake_2 = torch.argmax(stats['fake_data'].view(self.batch_size, 1, size, size), dim=1).unsqueeze(1)
        fake_2 = fake_2.int()
        fake_2 = fake_2.cpu().detach().clone()
        fake_2 = torchvision.utils.make_grid(fake_2, nrow=8, padding=2)
        self.writer.add_image('G/images', fake_2, stats['iteration'])

        dev_disc_costs=[]
        for _, images in enumerate(self.val_iter):
            imgs = torch.Tensor(images[0])
            imgs = imgs.to(device)
            with torch.no_grad():
                imgs_v = imgs

            D = self.D(imgs_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)

        lib.plot.plot(self.output_path + 'dev_disc_cost.png', np.mean(dev_disc_costs))
        lib.plot.flush()
        gen_images = generate_image(self.G, self.fixed_noise)
        torchvision.utils.save_image(gen_images, self.output_path + 'samples_{}.png'.format(stats['iteration']), nrow=8,
                                     padding=2)
        grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
        self.writer.add_image('images', grid_images, stats['iteration'])

        torch.save(self.G, self.output_path + 'generator.pt')
        torch.save(self.D, self.output_path + 'discriminator.pt')

    def train(self,iters):
        for iteration in range(iters):
            start_time=time.time()

            gen_cost,real_p1,=self.generator_update()

            stats=self.critic_update()
            stats['start'],stats['iteration'],stats['gen_cost']=start_time,iteration,gen_cost

            self.log(stats)
            if iteration%10==0:
                self.save(stats)
            lib.plot.tick()
if __name__=='__main__':
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    config=InvNetConfig()
    invnet=InvNet(config.batch_size,config.output_path,config.lr,config.critic_iter,config.proj_iter,128*128,device,config.lambda_gp)
    invnet.train(1)