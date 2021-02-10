import math
import numpy as np

from invnet import BaseInvNet
import torch
from micro_invnet import MicrostructureDataset
from layers import DPLayer
import torch.nn as nn
import torchvision

#proj ablation: /home/km3888/invnet/runs/Feb05_15-49-30_hegde-lambda-1 and the next four after
#running w/ shortest paths instead of longest ; Feb06_20-55-29_hegde-lambda-1 - /Feb06_20-55-57_hegde-lambda-1
#Fixed shortest paths: Feb09_11-53-16 -
class MicroInvnet(BaseInvNet):

    def __init__(self,batch_size,output_path,data_dir,lr,critic_iters,\
                 proj_iters,hidden_size,device,lambda_gp,edge_fn,max_op,restore_mode=False):

        super().__init__(batch_size,output_path,data_dir,lr,critic_iters,proj_iters,64**2,hidden_size,device,lambda_gp,1,restore_mode)

        print(self.proj_iters)
        self.edge_fn=edge_fn
        self.max_op=max_op
        self.DPLayer=DPLayer('v1_only',max_op,64,64,make_pos=False)

        self.p1_mean,self.p1_std=self.get_p1_stats()

    def proj_loss(self,fake_data,real_p1):
        fake_lengths=self.real_p1(fake_data)
        loss_f=nn.MSELoss()
        return loss_f(fake_lengths,real_p1)

    def real_p1(self,images):
        '''
        Images: [b,64,64,6]
        p1: [b,6]
        '''
        lengths=self.DPLayer(images)
        return lengths.view((-1,1))

    def sample(self,train=True):
        if train:
            try:
                real_data = next(self.dataiter)
            except:
                self.dataiter = iter(self.train_loader)
                real_data = self.dataiter.next()
            if real_data.shape[0] < self.batch_size:
                real_data = self.sample()
        else:
            try:
                real_data = next(self.val_iter)
            except:
                self.val_iter = iter(self.val_loader)
                real_data = self.val_iter.next()
            if real_data.shape[0] < self.batch_size:
                real_data = self.sample(train=False)
        return real_data.squeeze()

    def load_data(self):
        train_dir = self.data_dir + 'morph_global_64_train_255.h5'
        test_dir = self.data_dir + 'morph_global_64_valid_255.h5'
        #Returns train_loader and val_loader, both of pytorch DataLoader type
        train_data=MicrostructureDataset(train_dir)
        test_data=MicrostructureDataset(test_dir)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        return train_loader,test_loader

    def norm_data(self,data):
        #only get used for saving images
        data=data.view(-1,64,64)
        mean=data.mean(dim=0)
        deviation=data.std(dim=0)
        return (data-mean)/deviation

    def format_data(self,data):
        return data.view((self.batch_size,64,64))

    def get_p1_stats(self):
        p1_values=[]
        for _ in range(10):
            batch=self.sample()
            p1=self.DPLayer(batch)
            p1_values+=list(p1)
        values=np.array(p1_values)
        return values.mean(),values.std()

    def save(self,stats):
        # TODO split this into base saving actions and MNIST/DP specific saving stuff
        size = int(math.sqrt(self.output_dim))
        fake_2 = stats['fake_data'].view(self.batch_size, -1, size, size)
        fake_2 = fake_2.int()
        fake_2 = fake_2.cpu().detach().clone()
        fake_2 = torchvision.utils.make_grid(fake_2, nrow=8, padding=2)#todo this isn't working
        self.writer.add_image('G/images', fake_2, stats['iteration'])

        dev_proj_err, dev_disc_cost = stats['val_proj_err'],stats['val_critic_err']
        # Generating images for tensorboard display
        #284 is mean
        #std is 18.5
        mean,std=self.p1_mean,self.p1_std
        lv = torch.tensor([mean-std, mean, mean+std, mean+2*std]).view(-1, 1).float().to(self.device)
        with torch.no_grad():
            noisev = self.fixed_noise
            lv_v = lv
        noisev = noisev.float()
        gen_images = self.G(noisev, lv_v)
        normed_gen_images = self.norm_data(gen_images).view((4, -1, size, size))
        real_images =  stats['real_data']
        normed_real_images = self.norm_data(real_images).view((4,-1,size,size))
        real_grid_images = torchvision.utils.make_grid(normed_real_images[:4], nrow=8, padding=2)
        fake_grid_images = torchvision.utils.make_grid(normed_gen_images, nrow=8, padding=2)
        real_grid_images = real_grid_images.long()
        fake_grid_images = fake_grid_images.long()
        self.writer.add_image('real images', real_grid_images, stats['iteration'])
        self.writer.add_image('fake images', fake_grid_images, stats['iteration'])
        torch.save(self.G, self.output_path + 'generator.pt')
        torch.save(self.D, self.output_path + 'discriminator.pt')

        metric_dict = {'generator_cost': stats['gen_cost'],
                       'dev_discriminator_cost': dev_disc_cost, 'dev_validation_projection_error': dev_proj_err}
        self.writer.add_hparams(self.hparams, metric_dict, global_step=stats['iteration'])

