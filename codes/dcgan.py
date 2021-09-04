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

    if not lamb: return loss_WGAN

    # GP: gradient penalty
    alpha = torch.rand(len(real_imgs), device=real_imgs.device).view(-1,1,1,1)
    inter_imgs = alpha * real_imgs + (1-alpha) * fake_imgs
    inter_imgs.requires_grad=True

    grads, = torch.autograd.grad(disnet(inter_imgs).sum(), inter_imgs, create_graph=True) # takes much memory
    loss_GP = ((grads.view(len(grads), -1).norm(2, dim=1) - 1)**2).mean()

    # get loss for D
    loss_D = loss_WGAN + lamb * loss_GP

    return loss_D

def save_for_resume(path, epoch, gennet, disnet, optimizer_G, optimizer_D, loss_G, loss_D):
    torch.save({
        'epoch': epoch,
        'G_state_dict': gennet.state_dict(),
        'D_state_dict': disnet.state_dict(),
        'G_optimizer_state_dict': optimizer_G.state_dict(),
        'D_optimizer_state_dict': optimizer_D.state_dict(),
        'G_loss': loss_G,
        'D_loss': loss_D,
        }, path + 'for_resume_ep%i.pt'%epoch)

def load_for_resume(path, epoch, gennet, disnet, optimizer_G, optimizer_D):
    for_resume = torch.load(path + 'for_resume_ep%i.pt'%epoch, map_location='cpu') # load to cpu in case GPU is full

    assert epoch == for_resume['epoch'], "\nResume file error !\n"

    gennet.load_state_dict(for_resume['G_state_dict'])
    disnet.load_state_dict(for_resume['D_state_dict'])
    optimizer_G.load_state_dict(for_resume['G_optimizer_state_dict'])
    optimizer_D.load_state_dict(for_resume['D_optimizer_state_dict'])

    print('\nResume training from epoch %i.\n'%epoch)



if __name__ == '__main__':

    opt = config_options()

    os.makedirs(opt.datapath, exist_ok=True)
    os.makedirs(opt.workpath, exist_ok=True)
    os.makedirs(opt.workpath+'images/', exist_ok=True)
    os.makedirs(opt.workpath+'models/', exist_ok=True)

    # use CUDA whenever available
    cuda_on = torch.cuda.is_available()
    device = ('cpu',             'cuda'                )[cuda_on]
    tensor = (torch.FloatTensor, torch.cuda.FloatTensor)[cuda_on]

    # Initialize generator and discriminator
    gennet = Generator(opt.latent_dim, opt.img_size).to(device)
    disnet = Discriminator(opt.img_size).to(device)

    # gennet.apply(weights_init_normal)
    # disnet.apply(weights_init_normal)

    dataloader = DataLoader(
        Reader('/mnt/disk2/whn/etbl/TBLs/TBL_1420/test/', opt.datapath, opt.img_size),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(gennet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(disnet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Judge for measuring how well the generator is learning
    fid = FID(
        dataloader,
        wrapped_dl_gen(gennet, opt.latent_dim, opt.batch_size),
        opt.workpath + 'models/inception_v3.pt',
    )

    epoch = opt.resume

    if epoch < 0:
        print('\nTrain from scratch.\n')
    else:
        load_for_resume(opt.workpath + 'models/', epoch, gennet, disnet, optimizer_G, optimizer_D)

        if opt.lr != 1:
            print('\nMannually adjust lr by %f for resuming training.\n'%opt.lr)
            for g in optimizer_G.param_groups: g['lr'] *= opt.lr
            for g in optimizer_D.param_groups: g['lr'] *= opt.lr

    # scheduler2_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, factor=.5, patience=5, verbose=True)
    # scheduler2_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, factor=.5, patience=5, verbose=True)

    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=1 * int(opt.n_critic**.5+.5), gamma=.9, last_epoch=epoch, verbose=True)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=1 * int(opt.n_critic**.5+.5), gamma=.9, last_epoch=epoch, verbose=True)

    # ----------
    #  Training
    # ----------

    gennet.train()
    disnet.train()

    while epoch < opt.n_epochs * opt.n_critic:
        epoch += 1
        total_losses = []

        for i, imgs in enumerate(dataloader):

            iters = epoch * len(dataloader) + i

            bs = len(imgs) # size of the current batch, may be different for the last batch

            # put loaded batch into CUDA
            real_imgs = imgs.to(device).type(tensor)

            # Sample noise as gennet input
            z = torch.randn((bs, opt.latent_dim), device=device)

            # tool arrays for computing BCE loss
            authentic = torch.ones(bs, device=device).view(-1,1)
            counterfeit = torch.zeros(bs, device=device).view(-1,1)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            disnet.requires_grad_(True)
            optimizer_D.zero_grad()

            # Generate a batch of images
            with torch.no_grad():
                fake_imgs = gennet(z) # detach means the gradient of loss_D will not affect G through fake_imgs

            loss_D = loss_WGAN_GP(disnet, real_imgs, fake_imgs, opt.lambda_gp)

            loss_D.backward()
            optimizer_D.step()

            if iters % opt.n_critic: continue

            # -----------------
            #  Train Generator
            # -----------------

            disnet.requires_grad_(False)
            optimizer_G.zero_grad()

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

            loss_G = -disnet(fake_imgs).mean() + .5**epoch * (opt.lambda_d1 * d1 + opt.lambda_d2 * d2)

            loss_G.backward()
            optimizer_G.step()

            np.savetxt(opt.workpath + 'real_varu.dat', real_imgs[0,0].var(dim=-1).detach().cpu().numpy())
            np.savetxt(opt.workpath + 'fake_varu.dat', fake_imgs[0,0].var(dim=-1).detach().cpu().numpy())
            np.savetxt(opt.workpath + 'real_varv.dat', real_imgs[0,1].var(dim=-1).detach().cpu().numpy())
            np.savetxt(opt.workpath + 'fake_varv.dat', fake_imgs[0,1].var(dim=-1).detach().cpu().numpy())
            np.savetxt(opt.workpath + 'real_varw.dat', real_imgs[0,2].var(dim=-1).detach().cpu().numpy())
            np.savetxt(opt.workpath + 'fake_varw.dat', fake_imgs[0,2].var(dim=-1).detach().cpu().numpy())

            # ---------
            #  Monitor
            # ---------

            total_losses.append((loss_G + loss_D).item())

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]" % (
                epoch, opt.n_epochs * opt.n_critic, i, len(dataloader), loss_D.item(), loss_G.item() ))

            with open(opt.workpath + 'log.dat', 'aw'[iters==0]) as fp:
                fp.write(
                    '%i\t%.8e\t%.8e\t%.8e\t%.8e\n'%(
                    iters,
                    loss_D.item(),
                    loss_G.item(),
                    opt.lambda_d1 * d1.item(),
                    opt.lambda_d2 * d2.item(),
                    ))

            if (iters//opt.n_critic) % opt.draw_every == 0:
                vel = fake_imgs[0].detach().cpu().numpy()
                ds = dataloader.dataset
                ys = ds.gengrid(ds.para.Ly, vel.shape[-2])
                zs = ds.para.Lz * np.arange(vel.shape[-1]) / vel.shape[-1]

                gennet.eval()
                with open(opt.workpath + 'fid.dat', 'aw'[iters==0]) as fp:
                    fidr, fid1, fid0 = fid.calc(relative=True)
                    fp.write('%i\t%.8e\t%.8e\t%.8e\n'%(iters, fidr, fid1, fid0))
                gennet.train()

                draw_vel(opt.workpath + 'images/%d.png'%iters, vel, ys, zs)
                draw_log(opt.workpath + 'log.png', opt.workpath + 'log.dat')
                draw_fid(opt.workpath + 'fid.png', opt.workpath + 'fid.dat')

        # save the model every epoch for resuming training
        save_for_resume(opt.workpath+'models/', epoch, gennet, disnet, optimizer_G, optimizer_D, loss_G, loss_D)
        torch.save(gennet.state_dict(), opt.workpath+'models/model_G.pt')
        torch.save(disnet.state_dict(), opt.workpath+'models/model_D.pt')
        print('Resume file saved for epoch %i.'%epoch)

        # learning rate decay
        scheduler_G.step()
        scheduler_D.step()

        # scheduler2_G.step(np.mean(total_losses))
        # scheduler2_D.step(np.mean(total_losses))
        # total_losses = []




