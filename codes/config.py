import argparse

def show_namespace(name_space, msg='Namespace:'):

    print('\n%s\n%s\n%s'%('-'*30, msg, '-'*30))

    for name in dir(name_space):
        if name[0] != '_':
            print(name, '\t=', getattr(name_space, name))

    print('%s\n'%('-'*30))


def config_options():
    parser = argparse.ArgumentParser()

    # -----------------------
    # Computational settings
    # -----------------------

    # paths
    parser.add_argument("--datapath", type=str, default='../dataset/', help="path of the dataset")
    parser.add_argument("--workpath", type=str, default='../results/', help="path to store the results")

    # parallelization
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu processes for distributed training without GPU, only effective when n_gpu = 0")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of gpu devices to use for training, 0 for pure CPU training")

    # process control
    parser.add_argument("--epochs", type=int, default=2000,   help="number of epochs of training")
    parser.add_argument("--resume", type=int, default=-1,     help="resume training from last epoch")
    parser.add_argument("--check_every", type=int, default=1, help="interval between image sampling")

    # -----------------
    # Hyper-parameters
    # -----------------

    # input info
    parser.add_argument("--latent_dim", type=int, default=256, help="dimension of the latent space")
    parser.add_argument("--img_size",   type=int, default=192, help="size of each image dimension")
    parser.add_argument("--img_chan",   type=int, default=3,   help="number of image channels")
    
    # optimizer options
    parser.add_argument("--batch_size", type=int,   default=16,      help="size of the mini-batch on each GPU")
    parser.add_argument("--lr",         type=float, default=1e-3,    help="adam: learning rate")
    parser.add_argument("--lr_decay",   type=str,   default='const', help="scheduler for learning rate decay")
    parser.add_argument("--beta1", type=float, default=0.9,   help="adam: exp decay rate for the first moment estimation")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: exp decay rate for the second moment estimation")
    
    # GAN options
    parser.add_argument("--n_critic", type=int, default=1, help="multiple of D-training iterations w.r.t G-training iterations (to train D better)")
    parser.add_argument("--lambda_gp", type=float, default=100, help="gradient penalty coefficient for WGAN-GP loss")
    parser.add_argument("--lambda_d1", type=float, default=100, help="penalty coefficient for labelling loss")
    parser.add_argument("--lambda_d2", type=float, default=1000, help="penalty coefficient for statistical constraint")

    # -----------------
    # Print
    # -----------------

    options = parser.parse_args()

    if options.n_gpu:
        options.n_cpu = options.n_gpu
    
    show_namespace(options, 'Options are:')
    
    return options
