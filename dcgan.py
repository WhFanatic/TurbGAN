import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import config_options
from generator import Generator
from discriminator import Discriminator
from reader import Reader
from fid import FID, wrapped_dl_gen
from plots import draw_vel, draw_log, draw_fid


def weights_init_normal(m):
    # initiate parameters in all Conv, Linear and BN layers for an arbitrary model m
    # when called by model.apply(), this function applies to all submodels in the model following LRD of the model tree
    classname = m.__class__.__name__

    if classname.find("Conv") >= 0 or classname.find("Linear") >= 0:
        nn.init.kaiming_normal_(m.weight.data, a=.2)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm2d") >= 0:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)


def loss_WGAN_GP(disnet, real_imgs, fake_imgs, lamb):
    ## compute the WGAN-GP loss proposed by Gulrajani et al. 2017

    # # another realization: requires less memory
    # # D loss for WGAN
    # loss_r = - torch.mean(disnet(real_imgs))
    # loss_r.backward()

    # loss_f = torch.mean(disnet(fake_imgs))
    # loss_f.backward()

    # # GP: gradient penalty
    # alpha = torch.rand(len(real_imgs), 1, 1, 1, device=real_imgs.device)
    # inter_imgs = alpha * real_imgs + (1-alpha) * fake_imgs

    # loss_gp = 0
    # local_bs = 16

    # for imgs in DataLoader(inter_imgs, batch_size=local_bs):
    #     imgs.requires_grad = True
    #     grads, = torch.autograd.grad(disnet(imgs).sum(), imgs, retain_graph=True, create_graph=True)
    #     loss = lamb/len(inter_imgs) * torch.square(torch.sum(grads**2, dim=(1,2,3))**.5 - 1).sum()
    #     loss.backward()
    #     loss_gp += loss

    # return loss_r + loss_f + loss_gp
    # # remember to comment out the loss_D.backward() in the main loop


    # D loss for WGAN
    loss_WGAN = torch.mean(disnet(fake_imgs)) - torch.mean(disnet(real_imgs))

    # GP: gradient penalty
    alpha = torch.rand(len(real_imgs), 1, 1, 1, device=real_imgs.device)
    inter_imgs = alpha * real_imgs + (1-alpha) * fake_imgs
    inter_imgs.requires_grad=True

    grads, = torch.autograd.grad(disnet(inter_imgs).sum(), inter_imgs, create_graph=True) # takes much memory
    loss_GP = ((grads.view(len(grads), -1).norm(2, dim=1) - 1)**2).mean()

    # get loss for D
    loss_D = loss_WGAN + lamb * loss_GP

    return loss_D

def save_for_resume(path, gennet, disnet, opt_G, opt_D, loss_G, loss_D, epoch):
    torch.save({
        'epoch': epoch,
        'G_state_dict': gennet.state_dict(),
        'D_state_dict': disnet.state_dict(),
        'G_optimizer_state_dict': opt_G.state_dict(),
        'D_optimizer_state_dict': opt_D.state_dict(),
        'G_loss': loss_G,
        'D_loss': loss_D,
        }, path + 'for_resume.pt')

def load_for_resume(path, gennet, disnet, opt_G, opt_D):
    try:
        for_resume = torch.load(path + 'for_resume.pt')
    except:
        print('\nTrain from scratch.\n')
        return -1
    
    gennet.load_state_dict(for_resume['G_state_dict'])
    disnet.load_state_dict(for_resume['D_state_dict'])
    opt_G.load_state_dict(for_resume['G_optimizer_state_dict'])
    opt_D.load_state_dict(for_resume['D_optimizer_state_dict'])
    epoch = for_resume['epoch']

    print('\nResume training from epoch %i.\n'%(epoch+1))

    return epoch



if __name__ == '__main__':

    options = config_options()

    os.makedirs(options.datapath, exist_ok=True)
    os.makedirs(options.workpath, exist_ok=True)
    os.makedirs(options.workpath+'images/', exist_ok=True)
    os.makedirs(options.workpath+'models/', exist_ok=True)

    # use CUDA whenever available
    device = ('cpu', 'cuda')[torch.cuda.is_available()]
    tensor = (torch.FloatTensor, torch.cuda.FloatTensor)[torch.cuda.is_available()]

    # Initialize generator and discriminator
    gennet = Generator(options).to(device)
    disnet = Discriminator(options).to(device)

    gennet.apply(weights_init_normal)
    disnet.apply(weights_init_normal)

    dataloader = DataLoader(
        Reader('/mnt/disk2/whn/etbl/TBL_1420_big/test/', options.datapath, options.img_size),
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.n_cpu,
    )

    # Optimizers
    opt_G = torch.optim.Adam(gennet.parameters(), lr=options.lr, betas=(options.b1, options.b2))
    opt_D = torch.optim.Adam(disnet.parameters(), lr=options.lr, betas=(options.b1, options.b2))

    # Judge for measuring how well the generator is learning
    fid = FID(
        dataloader,
        wrapped_dl_gen(gennet, options.latent_dim, options.batch_size),
        options.workpath + 'models/inception_v3.pt',
    )

    # ----------
    #  Training
    # ----------

    epoch = load_for_resume(options.workpath+'models/', gennet, disnet, opt_G, opt_D)

    gennet.train()
    disnet.train()

    for epoch in range(epoch+1, options.n_epochs):

        for i, imgs in enumerate(dataloader):

            iters = epoch * len(dataloader) + i

            bs = len(imgs) # size of the current batch, may be different for the last batch

            # put loaded batch into CUDA
            real_imgs = imgs.to(device).type(tensor)

            # Sample noise as gennet input
            z = torch.randn((bs, options.latent_dim), device=device)

            # tool arrays for computing BCE loss
            authentic = torch.ones(bs, device=device).view(-1,1)
            counterfeit = torch.zeros(bs, device=device).view(-1,1)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            disnet.requires_grad_(True)
            opt_D.zero_grad()

            # Generate a batch of images
            with torch.no_grad():
                fake_imgs = gennet(z) # detach means the gradient of loss_D will not affect G through fake_imgs

            loss_D = loss_WGAN_GP(disnet, real_imgs, fake_imgs, 100)

            loss_D.backward()
            opt_D.step()

            if iters % options.n_critic: continue

            # -----------------
            #  Train Generator
            # -----------------

            disnet.requires_grad_(False)
            opt_G.zero_grad()

            # Generate a batch of images
            fake_imgs = gennet(z)

            loss_G = -disnet(fake_imgs).mean() + 100 * torch.sum(fake_imgs.mean(dim=-1)**2)

            # print(torch.sum(fake_imgs.mean(dim=-1)**2).item())
            # print(torch.sum(real_imgs.mean(dim=-1)**2).item())

            loss_G.backward()
            opt_G.step()

            # ---------
            #  Monitor
            # ---------

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]" % (
                epoch, options.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item() ))

            with open(options.workpath + 'log.dat', 'aw'[iters==0]) as fp:
                fp.write('%i\t%.8e\t%.8e\n'%(iters, loss_D, loss_G))

            if (iters//options.n_critic) % options.draw_every == 0:
                vel = fake_imgs[0].detach().cpu()
                ds = dataloader.dataset
                ys = ds.gengrid(ds.para.Ly, vel.shape[-2])
                zs = ds.para.Lz * np.arange(vel.shape[-1]) / vel.shape[-1]

                gennet.eval()
                with open(options.workpath + 'fid.dat', 'aw'[iters==0]) as fp:
                    fidr, fid1, fid0 = fid.calc(relative=True)
                    fp.write('%i\t%.8e\t%.8e\t%.8e\n'%(iters, fidr, fid1, fid0))
                gennet.train()

                draw_vel(options.workpath + 'images/%d.png'%iters, vel, ys, zs)
                draw_log(options.workpath + 'log.png', options.workpath + 'log.dat')
                draw_fid(options.workpath + 'fid.png', options.workpath + 'fid.dat')

        # save the model every epoch for resuming training
        save_for_resume(options.workpath+'models/', gennet, disnet, opt_G, opt_D, loss_G, loss_D, epoch)
        torch.save(gennet.state_dict(), options.workpath+'models/model_G.pt')
        torch.save(disnet.state_dict(), options.workpath+'models/model_D.pt')





