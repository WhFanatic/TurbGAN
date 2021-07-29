import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import config_options
from models import Generator, Discriminator
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

    # D loss for WGAN
    # loss_WGAN = torch.mean(disnet(fake_imgs)) - torch.mean(disnet(real_imgs))
    loss_WGAN = disnet(torch.cat((fake_imgs, real_imgs))).view(2,-1).mean(dim=-1)
    loss_WGAN = loss_WGAN[0] - loss_WGAN[1]

    # GP: gradient penalty
    alpha = torch.rand(len(real_imgs), device=real_imgs.device).view(-1,1,1,1)
    inter_imgs = alpha * real_imgs + (1-alpha) * fake_imgs
    inter_imgs.requires_grad=True

    grads, = torch.autograd.grad(disnet(inter_imgs).sum(), inter_imgs, create_graph=True) # takes much memory
    loss_GP = ((grads.view(len(grads), -1).norm(2, dim=1) - 1)**2).mean()

    # get loss for D
    loss_D = loss_WGAN + lamb * loss_GP

    return loss_D

def save_for_resume(path, epoch, gennet, disnet, opt_G, opt_D, loss_G, loss_D):
    torch.save({
        'epoch': epoch,
        'G_state_dict': gennet.state_dict(),
        'D_state_dict': disnet.state_dict(),
        'G_optimizer_state_dict': opt_G.state_dict(),
        'D_optimizer_state_dict': opt_D.state_dict(),
        'G_loss': loss_G,
        'D_loss': loss_D,
        }, path + 'for_resume_ep%i.pt'%epoch)

def load_for_resume(path, epoch, gennet, disnet, opt_G, opt_D):
    for_resume = torch.load(path + 'for_resume_ep%i.pt'%epoch, map_location='cpu') # load to cpu in case GPU is full

    assert epoch == for_resume['epoch'], "\nResume file error !\n"

    gennet.load_state_dict(for_resume['G_state_dict'])
    disnet.load_state_dict(for_resume['D_state_dict'])
    opt_G.load_state_dict(for_resume['G_optimizer_state_dict'])
    opt_D.load_state_dict(for_resume['D_optimizer_state_dict'])

    print('\nResume training from epoch %i.\n'%epoch)



if __name__ == '__main__':

    options = config_options()

    os.makedirs(options.datapath, exist_ok=True)
    os.makedirs(options.workpath, exist_ok=True)
    os.makedirs(options.workpath+'images/', exist_ok=True)
    os.makedirs(options.workpath+'models/', exist_ok=True)

    # use CUDA whenever available
    cuda_on = torch.cuda.is_available()
    device = ('cpu',             'cuda'                )[cuda_on]
    tensor = (torch.FloatTensor, torch.cuda.FloatTensor)[cuda_on]

    # Initialize generator and discriminator
    gennet = Generator(options.latent_dim, options.img_size).to(device)
    disnet = Discriminator(options.img_size).to(device)

    # gennet.apply(weights_init_normal)
    # disnet.apply(weights_init_normal)

    dataloader = DataLoader(
        Reader('/mnt/disk2/whn/etbl/TBL_1420/test/', options.datapath, options.img_size),
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

    epoch = options.resume

    if epoch < 0: print('\nTrain from scratch.\n')
    else: load_for_resume(options.workpath + 'models/', epoch, gennet, disnet, opt_G, opt_D)

    # # mannually adjust lr in resuming training
    # for g in opt_G.param_groups: g['lr'] *= .5
    # for g in opt_D.param_groups: g['lr'] *= .5

    # scheduler2_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, factor=.5, patience=5, verbose=True)
    # scheduler2_D = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_D, factor=.5, patience=5, verbose=True)

    scheduler_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=1*options.n_critic, gamma=.9, last_epoch=epoch, verbose=True)
    scheduler_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=1*options.n_critic, gamma=.9, last_epoch=epoch, verbose=True)

    # ----------
    #  Training
    # ----------

    gennet.train()
    disnet.train()

    while epoch < options.n_epochs:
        epoch += 1
        total_losses = []

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

            loss_D = loss_WGAN_GP(disnet, real_imgs, fake_imgs, options.lambda_gp)

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

            # imprecise statistical constraints (Yang, Wu & Xiao 2021; Wu et al. 2020)
            # assuming the Re of each sample in fake batch is in accordance with that in real batch
            real_ave = real_imgs.mean(dim=-1)
            fake_ave = fake_imgs.mean(dim=-1)
            real_std = real_imgs.std(dim=-1)
            fake_std = fake_imgs.std(dim=-1)

            distF_d1 = (fake_ave - real_ave)**2
            distF_d2 = (fake_std - real_std)**2
            thres_d1 = (real_std * .1)**2
            thres_d2 = (real_std * .3)**2

            # d = < \Sigma{ max[ (S(x)-S(y))^2 - \epsilon^2 , 0 ] }^2 >
            d1 = torch.mean(torch.maximum(distF_d1 - thres_d1, torch.zeros_like(distF_d1)).sum(dim=(1,2))**2)
            d2 = torch.mean(torch.maximum(distF_d2 - thres_d2, torch.zeros_like(distF_d1)).sum(dim=(1,2))**2)

            loss_G = -disnet(fake_imgs).mean() + options.lambda_d1 * d1 + options.lambda_d2 * d2

            loss_G.backward()
            opt_G.step()

            np.savetxt(options.workpath + 'real_varu.dat', real_imgs[0,0].var(dim=-1).detach().cpu().numpy())
            np.savetxt(options.workpath + 'fake_varu.dat', fake_imgs[0,0].var(dim=-1).detach().cpu().numpy())
            np.savetxt(options.workpath + 'real_varv.dat', real_imgs[0,1].var(dim=-1).detach().cpu().numpy())
            np.savetxt(options.workpath + 'fake_varv.dat', fake_imgs[0,1].var(dim=-1).detach().cpu().numpy())
            np.savetxt(options.workpath + 'real_varw.dat', real_imgs[0,2].var(dim=-1).detach().cpu().numpy())
            np.savetxt(options.workpath + 'fake_varw.dat', fake_imgs[0,2].var(dim=-1).detach().cpu().numpy())

            # ---------
            #  Monitor
            # ---------

            total_losses.append((loss_G + loss_D).item())

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]" % (
                epoch, options.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item() ))

            with open(options.workpath + 'log.dat', 'aw'[iters==0]) as fp:
                fp.write(
                    '%i\t%.8e\t%.8e\t%.8e\t%.8e\n'%(
                    iters,
                    loss_D.item(),
                    loss_G.item(),
                    options.lambda_d1 * d1.item(),
                    options.lambda_d2 * d2.item(),
                    ))

            if (iters//options.n_critic) % options.draw_every == 0:
                vel = fake_imgs[0].detach().cpu().numpy()
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
        save_for_resume(options.workpath+'models/', epoch, gennet, disnet, opt_G, opt_D, loss_G, loss_D)
        torch.save(gennet.state_dict(), options.workpath+'models/model_G.pt')
        torch.save(disnet.state_dict(), options.workpath+'models/model_D.pt')
        print('Resume file saved for epoch %i.'%epoch)

        # learning rate decay
        scheduler_G.step()
        scheduler_D.step()

        # scheduler2_G.step(np.mean(total_losses))
        # scheduler2_D.step(np.mean(total_losses))
        # total_losses = []




