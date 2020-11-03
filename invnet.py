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
from didyprog.image_generation.sp_layer import SPLayer
from didyprog.image_generation.mnist_digit import make_graph,compute_distances
from config import *
import libs as lib
import libs.plot
import numpy as np
import sys

class InvNet:

    def __init__(self,batch_size,output_path,data_dir,lr,critic_iters,proj_iters,output_dim,hidden_size,device,lambda_gp,restore_mode=False):
        self.writer = SummaryWriter()
        self.device=device

        self.data_dir=data_dir
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
            self.G = GoodGenerator(hidden_size, self.output_dim, ctrl_dim=11).to(device)
            self.D = GoodDiscriminator(hidden_size).to(device)
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.optim_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0, 0.9))
        self.optim_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0, 0.9))
        self.optim_pj = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0, 0.9))

        self.fixed_noise=gen_rand_noise(4)


        self.dp_layer=SPLayer()

    def load_data(self):
        data_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3801])
        ])
        data_dir = self.data_dir
        mnist_data = datasets.MNIST(data_dir, download=True,
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
        real_class = F.one_hot(real_data[1], num_classes=10)
        real_lengths=self.real_lengths(real_data[0])
        real_class = real_class.float()
        print('real_class:',real_class.shape)
        print('real_lengths:',real_lengths.shape)
        real_p1 = torch.cat([real_class,real_lengths],dim=1).to(self.device)
        mone = torch.FloatTensor([1]) * -1
        mone=mone.to(self.device)

        for i in range(1):
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
            # print("Critic iter: " + str(i))

            start = timer()
            self.D.zero_grad()
            real_data = self.sample()
            # gen fake data and load real data
            noise = gen_rand_noise(self.batch_size).to(self.device)

            # batch = batch[0] #batch[1] contains labels
            real_images = real_data[0].to(self.device)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
                real_class = F.one_hot(real_data[1], num_classes=10).to(self.device)
                real_class = real_class.float()
                real_lengths= self.real_lengths(real_images)
                real_p1 = torch.cat([real_class,real_lengths],dim=1).to(self.device)
            end = timer()
            # print('---gen G elapsed time:', end - start)
            start = timer()
            fake_data = self.G(noisev, real_p1).detach()
            end = timer()
            # print('---load real imgs elapsed time:', end - start)
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
        print('---train D elapsed time:', end - start)
        stats={'w_dist': w_dist,
               'disc_cost':disc_cost,
               'fake_data':fake_data,
               'real_data':real_data,
               'disc_real':disc_real,
               'disc_fake':disc_fake,
               'gradient_penalty':gradient_penalty}
        return stats

    def proj_update(self):

        real_data=self.sample()
        images= real_data[0].detach().to(self.device)
        real_lengths=self.real_lengths(images).view(-1,1).to(device)
        real_class = F.one_hot(real_data[1], num_classes=10).float().to(self.device)
        for iteration in range(self.proj_iters):
            self.G.zero_grad()
            noise=gen_rand_noise(self.batch_size).to(self.device)
            noise.requires_grad=True

            specs=torch.cat([real_class,real_lengths],dim=1)
            fake_data = self.G(noise, specs)
            pj_grad,pj_err=self.proj_loss(fake_data,real_lengths)
            pj_grad.backward()
            self.optim_pj.step()

        return pj_err

    def proj_loss(self,fake_data,real_lengths):
        #TODO parallelize this

        grads=torch.zeros((self.batch_size,32*32)).to(self.device)
        fake_data=fake_data.view((self.batch_size,32,32))
        fake_data_copy=fake_data.to(self.device).detach().numpy()
        hard_vs=torch.zeros((self.batch_size)).to(self.device)
        for i in range(fake_data.shape[0]):
            v,E,v_hard=self.dp_layer.forward(fake_data_copy[i])
            grad=self.dp_layer.backward(fake_data_copy[i],E)

            grads[i]=torch.tensor(grad).view(-1)
            real_lengths[i]=v_hard
        real_lengths=real_lengths.squeeze()
        grads.requires_grad=False
        fake_data=fake_data.view((self.batch_size,-1))
        coeff=2*(hard_vs-real_lengths)
        coeff=coeff.squeeze()
        summed=(grads*fake_data).sum(dim=1)
        proj_loss=summed.dot(coeff)

        proj_err=(hard_vs-real_lengths)**2
        proj_err=proj_err.sum().item()

        return proj_loss,proj_err



    def real_lengths(self,images):
        #TODO Find a way to parallelize this
        #https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html

        real_lengths = []
        for i in range(images.shape[0]):
            _,_,v_hard=self.dp_layer.forward(images[i])
            real_lengths.append(v_hard)
        real_lengths=torch.tensor(real_lengths)
        return real_lengths.view(-1,1)

    def sample(self):
        try:
            real_data = next(self.dataiter)
        except:
            self.dataiter = iter(self.train_loader)
            real_data = self.dataiter.next()
        if real_data[0].shape[0]<self.batch_size:
            real_data=self.sample()
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
            imgs = imgs.to(self.device)
            with torch.no_grad():
                imgs_v = imgs

            D = self.D(imgs_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            print(_dev_disc_cost)
            dev_disc_costs.append(_dev_disc_cost)
            print(dev_disc_costs)
        lib.plot.plot(self.output_path + 'dev_disc_cost.png', np.mean(dev_disc_costs))
        lib.plot.flush()
        gen_images = generate_image(self.G, 4,noise=self.fixed_noise,device=self.device)
        real_images=stats['real_data'][0]
        real_grid_images = torchvision.utils.make_grid(real_images[:4], nrow=8, padding=2)
        fake_grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
        real_grid_images=real_grid_images.long()
        fake_grid_images = fake_grid_images.long()
        self.writer.add_image('real images', real_grid_images, stats['iteration'])
        self.writer.add_image('fake images',fake_grid_images,stats['iteration'])
        torch.save(self.G, self.output_path + 'generator.pt')
        torch.save(self.D, self.output_path + 'discriminator.pt')

    def train(self,iters):
        for iteration in range(iters):
            print('iteration:',iteration)
            start_time=time.time()


            gen_cost,real_p1=self.generator_update()

            proj_cost=self.proj_update()
            stats=self.critic_update()
            stats['start'],stats['iteration'],stats['gen_cost']=start_time,iteration,gen_cost

            self.log(stats)
            if iteration%10==0:
                self.save(stats)
            lib.plot.tick()

if __name__=='__main__':
    config=TestConfig()


    # torch.cuda.set_device(config.gpu)
    cuda_available = torch.cuda.is_available()
    device = torch.device(config.gpu if cuda_available else "cpu")

    print('training on:',device)
    sys.stdout.flush()
    invnet=InvNet(config.batch_size,config.output_path,config.data_dir,config.lr,config.critic_iter,config.proj_iter,32*32,config.hidden_size,device,config.lambda_gp)
    invnet.train(100000)