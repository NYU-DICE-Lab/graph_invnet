#from micro_invnet import MicroConfig,MicroInvnet
import torch

from microstructure import Config, InvNet

if __name__=="__main__":
    config = Config()

    cuda_available = torch.cuda.is_available()
    device = torch.device(config.gpu if cuda_available else "cpu")
    device = "cuda:0"
    if cuda_available:
        torch.cuda.set_device(device)
    print('training on:', device)

    invnet = InvNet(config.batch_size, config.output_path, config.data_dir,
                         config.lr, config.critic_iter, config.proj_iter,
                         config.hidden_size, device, config.lambda_gp, config.edge_fn, config.max_op,config.proj_lambda)
    invnet.train(config.end_iter)