import math

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets

from invnet import BaseInvNet
from layers.dp_layer.DPLayer import DPLayer


class InvNet(BaseInvNet):

    def __init__(self,batch_size,output_path,data_dir,lr,critic_iters,\
                 proj_iters,hidden_size,device,lambda_gp,edge_fn,max_op,proj_lambda,restore_mode=False):

        self.max_op=max_op
        make_pos=True
        if edge_fn=='diff_exp':
            make_pos=False
        self.dp_layer=DPLayer(edge_fn,max_op,32,32,make_pos=make_pos)
        new_hparams = {'max_op': str(max_op), 'edge_fn': edge_fn}
        super().__init__(batch_size,output_path,data_dir,lr,critic_iters,proj_iters,32**2,hidden_size,device,lambda_gp,1,proj_lambda,restore_mode=restore_mode,hparams=new_hparams)

    def proj_loss(self,fake_data,real_lengths):
        #TODO Experiment with normalization
        fake_data = fake_data.view((self.batch_size, 32, 32))
        real_lengths=real_lengths.view((-1,1))

        fake_lengths=self.real_p1(fake_data)
        proj_loss=F.mse_loss(fake_lengths,real_lengths)
        return proj_loss

    def real_p1(self,images):
        images=torch.squeeze(images)
        images=self.norm_data(images)
        real_lengths=self.dp_layer(images).view(-1,1)
        if self.p1_mean is not None:
            real_lengths=self.normalize_p1(real_lengths)
        return real_lengths

    def load_data(self):
        data_transform = transforms.Compose([
            transforms.Resize(32),
            # transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3801])
        ])
        data_dir = self.data_dir
        print('data_dir:',data_dir)
        mnist_data = datasets.MNIST(data_dir, download=True,
                                    transform=data_transform)
        train_data,val_data=torch.utils.data.random_split(mnist_data, [55000,5000])

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader=torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        images, _ = next(iter(train_loader))
        return train_loader,val_loader

    def norm_data(self,data):
        avg = data.mean()
        std = data.std()
        normed = (data - avg) / (std / 0.8)
        return normed

    def save(self,stats):
        #TODO split this into base saving actions and MNIST/DP specific saving stuff
        size = int(math.sqrt(self.output_dim))
        fake_2 = stats['fake_data'].view(self.batch_size, -1, size, size)
        fake_2 = fake_2.int()
        fake_2 = fake_2.cpu().detach().clone()
        fake_2 = torchvision.utils.make_grid(fake_2, nrow=8, padding=2)
        self.writer.add_image('G/images', fake_2, stats['iteration'])

        dev_proj_err, dev_disc_cost=self.validation()
        #Generating images for tensorboard display
        mean,std=self.p1_mean,self.p1_std
        lv=torch.tensor([mean-std,mean,mean+std,mean+2*std]).view(-1,1).float().to(self.device)
        with torch.no_grad():
            noisev=self.fixed_noise
            lv_v=lv
        noisev=noisev.float()
        gen_images=self.G(noisev,lv_v).view((4,-1,size,size))
        gen_images = self.norm_data(gen_images)
        real_images = stats['real_data']
        real_grid_images = torchvision.utils.make_grid(real_images[:4], nrow=8, padding=2)
        fake_grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
        real_grid_images = real_grid_images.long()
        fake_grid_images = fake_grid_images.long()
        self.writer.add_image('real images', real_grid_images, stats['iteration'])
        self.writer.add_image('fake images', fake_grid_images, stats['iteration'])
        torch.save(self.G, self.output_path + 'generator.pt')
        torch.save(self.D, self.output_path + 'discriminator.pt')


        metric_dict = {'generator_cost': stats['gen_cost'],
                       'discriminator_cost': dev_disc_cost ,'validation_projection_error': dev_proj_err}
        self.writer.add_hparams(self.hparams, metric_dict,global_step=stats['iteration'])