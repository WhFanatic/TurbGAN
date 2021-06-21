import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

from config import config_options
from generator import Generator
from discriminator import Discriminator


def weights_init_normal(m):
    # initiate parameters in all Conv and BN layers for an arbitrary model m
    classname = m.__class__.__name__

    if classname.find("Conv") >= 0:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif classname.find("BatchNorm2d") >= 0:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':

    options = config_options()

    os.makedirs(options.datapath, exist_ok=True)
    os.makedirs(options.workpath, exist_ok=True)
    os.makedirs(options.workpath+'images/', exist_ok=True)
    os.makedirs(options.workpath+'models/', exist_ok=True)

    # use CUDA whenever available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loss function
    adv_loss = nn.BCELoss().to(device)

    # Initialize generator and discriminator
    gennet = Generator(options).to(device)
    disnet = Discriminator(options).to(device)

    gennet.apply(weights_init_normal) # the function applies to all submodels in the model following LRD of the model tree
    disnet.apply(weights_init_normal)

    # Configure data loader
    dataloader = DataLoader(
        datasets.MNIST(
            options.datapath,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(options.img_size),
                transforms.ToTensor(),
                transforms.Normalize([.5], [.5])
            ]),
        ),
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.n_cpu,
    )

    # Optimizers
    opt_G = torch.optim.Adam(gennet.parameters(), lr=options.lr, betas=(options.b1, options.b2))
    opt_D = torch.optim.Adam(disnet.parameters(), lr=options.lr, betas=(options.b1, options.b2))

    # ----------
    #  Training
    # ----------

    gennet.train()
    disnet.train()

    for epoch in range(options.n_epochs):

        for i, (imgs, _) in enumerate(dataloader):

            # put loaded batch into CUDA
            real_imgs = imgs.to(device)

            authentic = torch.ones(len(imgs), device=device).view(-1,1)
            counterfeit = torch.zeros(len(imgs), device=device).view(-1,1)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for _ in range(options.k):
                opt_D.zero_grad()

                # Sample noise as gennet input
                z = torch.randn((len(imgs), options.latent_dim), device=device)

                # Generate a batch of images
                with torch.no_grad():
                    gen_imgs = gennet(z) # detach means the gradient of loss_D will not affect G through gen_imgs

                # Measure disnet's ability to classify real from generated samples
                real_loss = adv_loss(disnet(real_imgs), authentic)
                fake_loss = adv_loss(disnet(gen_imgs), counterfeit)
                loss_D = (real_loss + fake_loss) / 2

                loss_D.backward()
                opt_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            opt_G.zero_grad()

            # Sample noise as gennet input
            z = torch.randn((len(imgs), options.latent_dim), device=device)

            # Generate a batch of images
            gen_imgs = gennet(z)

            # Measure the distance of the generated images to authentic
            loss_G = adv_loss(disnet(gen_imgs), authentic)

            loss_G.backward()
            opt_G.step()

            # ---------
            #  Monitor
            # ---------

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
                epoch, options.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item() ))

            with open(options.workpath + 'log.dat', 'a') as fp:
                fp.write('%i\t%.8e\t%.8e\n'%(epoch, loss_D, loss_G))

            iters = epoch * len(dataloader) + i

            if iters % options.draw_every == 0:
                save_image(gen_imgs.data[:25], options.workpath+'images/%d.png'%iters, nrow=5, normalize=True) # put 5*5 generated figs together

                if iters > 0:
                    plt.semilogy(np.loadtxt(options.workpath+'log.dat')[:,1:], '.-')
                    plt.legend(['D loss', 'G loss'])
                    plt.savefig(options.workpath+'log.png', dpi=300)
                    plt.close()

        # save the model every epoch for resuming training
        torch.save({
            'epoch': epoch,
            'G_state_dict': gennet.state_dict(),
            'D_state_dict': disnet.state_dict(),
            'G_optimizer_state_dict': opt_G.state_dict(),
            'D_optimizer_state_dict': opt_D.state_dict(),
            'G_loss': loss_G,
            'D_loss': loss_D,
            }, options.workpath+'models/for_resume.pt')

        torch.save(gennet.state_dict(), options.workpath+'models/model_G.pt')
        torch.save(disnet.state_dict(), options.workpath+'models/model_D.pt')





