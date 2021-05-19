import torch

# Toggle this to change experiment type
from config import MicroStructureConfig as Config
from invnet import GraphInvNet

if __name__=="__main__":
    config = Config()

    cuda_available = torch.cuda.is_available()
    device = torch.device(config.gpu if cuda_available else "cpu")
    if cuda_available:
        torch.cuda.set_device(device)
    print('training on:', device)

    invnet = GraphInvNet(config.batch_size, config.output_path, config.data_dir,
                         config.lr, config.critic_iter, config.proj_iter, config.data_size, config.data_size,
                         config.hidden_size, device, config.lambda_gp,1, config.edge_fn, config.max_op,config.make_pos,
                         config.proj_lambda,config.top2bottom)
    invnet.train(config.end_iter)