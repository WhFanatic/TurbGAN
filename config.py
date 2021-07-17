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

    # input info
    parser.add_argument("--latent_dim", type=int, default=256, help="dimension of the latent space")
    parser.add_argument("--img_size",   type=int, default=192, help="size of each image dimension")
    parser.add_argument("--channels",   type=int, default=3, help="number of image channels")
    
    # training options
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    
    # output options
    parser.add_argument("--draw_every", type=int, default=100, help="interval between image sampling")

    # -----------------
    # Hyper-parameters
    # -----------------

    # optimizer options
    parser.add_argument("--batch_size", type=int, default=16, help="size of the mini-batch") # as large as GPU can fit
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9,   help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    
    # GAN options
    parser.add_argument("--n_critic", type=int, default=1, help="multiple of D-training iterations w.r.t G-training iterations (to train D better)")

    # -----------------
    # Print
    # -----------------

    options = parser.parse_args()
    show_namespace(options, 'Options are:')
    
    return options
