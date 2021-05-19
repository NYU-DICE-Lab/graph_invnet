""" Config class for training the InvNet """

import argparse

from dp_layer.graph_layer.edge_functions import edge_f_dict as d


def get_parser(name):
    """

    :param name: String for Config Name
    :return: parser
    """

    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser

class MNISTConfig(argparse.Namespace):
    def build_parser(self):
        parser = get_parser("InvNet config")
        parser.add_argument('--dataset', default='mnist', help='circle / polycrystalline')
        parser.add_argument('--lr',default=01e-04)
        parser.add_argument('--output_path', default='./output_dir', help='output directory')
        parser.add_argument('--data_dir', default='/data/MNIST')
        parser.add_argument('--gpu', default=1, help='Selecting the gpu')
        parser.add_argument('--data_size', default=64, type=int)
        parser.add_argument('--batch_size', default=32,type=int, help='Batch size for training')
        parser.add_argument('--hidden_size', default=32, type=int,help='Hidden size used for generator and discriminator')
        parser.add_argument('--critic_iter', default=5, type=int,help='Number of iter for descriminator')
        parser.add_argument('--proj_iter', default=3, type=int, help='Number of iteration for projection update.')
        parser.add_argument('--end_iter', default=30000, help='How many iterations to train for.')
        parser.add_argument('--lambda_gp', default=10, help='gradient penalty hyperparameter')
        parser.add_argument('--restore_mode', default=False,
                            help='If True, it will load saved model from OUT_PATH and continue to train')
        parser.add_argument('--max_op', default=False)
        parser.add_argument('--edge_fn', default='diff_exp')
        parser.add_argument('--make_pos', type=bool,default=True)
        parser.add_argument('--proj_lambda',type=float,default=1)
        parser.add_argument('--top2bottom', dest='top2bottom', action='store_true')
        parser.add_argument('--no-top2bottom', dest='top2bottom', action='store_false')
        parser.set_defaults(top2bottom=False)
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))


class MicroStructureConfig(argparse.Namespace):
    def build_parser(self):
        parser = get_parser("MicroConfig config")
        parser.add_argument('--lr',default=01e-04)
        parser.add_argument('--output_path', default='./output_dir', help='output directory')
        parser.add_argument('--data_dir', default='/data/datasets/two_phase_morph/')
        parser.add_argument('--gpu', default=1, help='Selecting the gpu')
        parser.add_argument('--data_size',default=64,type=int)
        parser.add_argument('--batch_size', default=32,type=int, help='Batch size for training')
        parser.add_argument('--hidden_size', default=32, type=int,help='Hidden size used for generator and discriminator')
        parser.add_argument('--critic_iter', default=5, type=int,help='Number of iter for descriminator')
        parser.add_argument('--proj_iter', default=1, type=int, help='Number of iteration for projection update.')
        parser.add_argument('--end_iter', default=50000, help='How many iterations to train for.')
        parser.add_argument('--lambda_gp', default=10, help='gradient penalty hyperparameter')
        parser.add_argument('--restore_mode', default=False,
                            help='If True, it will load saved model from OUT_PATH and continue to train')

        parser.add_argument('--max_op', default=False)
        parser.add_argument('--edge_fn', choices=list(d.keys()),default='diff_exp')
        parser.add_argument('--make_pos', type=bool, default=False)
        parser.add_argument('--proj_lambda', type=float, default=1)
        parser.add_argument('--top2bottom', dest='top2bottom', action='store_true')
        parser.add_argument('--no-top2bottom', dest='top2bottom', action='store_false')
        parser.set_defaults(top2bottom=False)

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
