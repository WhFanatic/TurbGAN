import os
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler

from config import config_options
from models import Generator, Discriminator
from reader import MyDataset
from fid import FID, wrapped_dl_gen
from plots import draw_vel, draw_log, draw_fid
from recorder import save_for_resume, load_for_resume, save_current
from losses import loss_D_CcGAN_WGAN, loss_G_CcGAN_WGAN


def statis_constraint(real_imgs, fake_imgs):
    ''' imprecise statistical constraints (Yang, Wu & Xiao 2021; Wu et al. 2020) '''
    # assuming the Re of each sample in fake batch is in accordance with that in real batch
    real_ave = real_imgs.mean(dim=-1)
    fake_ave = fake_imgs.mean(dim=-1)
    real_std = real_imgs.std(dim=-1)
    fake_std = fake_imgs.std(dim=-1)

    distF_d1 = (fake_ave - real_ave)**2
    distF_d2 = (fake_std - real_std)**2
    thres_d1 = (real_std * .1)**2
    thres_d2 = (real_std * .3)**2

    # d = < ||thres[ S(x)-S(y), eps ]||_F > = < \Sigma{max[ (S(x)-S(y))^2 - eps^2 , 0 ]}^.5 >
    d1 = torch.mean(torch.maximum(distF_d1 - thres_d1, torch.zeros_like(distF_d1)).sum(dim=(1,2))**.5)
    d2 = torch.mean(torch.maximum(distF_d2 - thres_d2, torch.zeros_like(distF_d1)).sum(dim=(1,2))**.5)

    return d1, d2

def check_vars(path, real_imgs, fake_imgs):
    i = np.random.randint(len(real_imgs))
    for j, v in enumerate('uvw'):
        np.savetxt(path + 'real_var%s.dat'%v, real_imgs[i,j].detach().var(dim=-1).cpu().numpy())
        np.savetxt(path + 'fake_var%s.dat'%v, fake_imgs[i,j].detach().var(dim=-1).cpu().numpy())


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
    disnet = Discriminator(opt.img_size, opt.latent_dim).to(device)

    # Define Dataset & Dataloader
    dataset = MyDataset(opt.datapath, img_size=opt.img_size)
    dataloader = DataLoader(
        dataset,
        batch_sampler=BatchSampler(
            sampler=WeightedRandomSampler(dataset.label_balancer(), num_samples=len(dataset)),
            batch_size=opt.batch_size,
            drop_last=False,
            ),
        num_workers=opt.n_cpu,
        prefetch_factor=8,
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



    sigma = 1e-20 # (4/3/len(dataset))**.2 * np.std(np.unique(dataset.get_labels()))
    nu = 1e-20 # (20 * np.diff(dataset.get_labels()).max())**-2 # 20 means at least neighbouring 40 labels can contribute to the conditional distribution
    thres_w = 1e-3
    lab_gap = (-np.log(thres_w)/nu)**.5

    print('\nCcGAN parameters:')
    print('sigma =', sigma)
    print('nu =', nu)
    print('thres_w =', thres_w)
    print('lab_gap =', lab_gap)
    print()


    # ----------
    #  Training
    # ----------

    while epoch < opt.n_epochs * opt.n_critic:
        epoch += 1

        gennet.train()
        disnet.train()

        for i, (imgs, labs) in enumerate(dataloader):

            iters = epoch * len(dataloader) + i

            bs = len(imgs) # size of the current batch, may be different for the last batch

            # put loaded batch into CUDA
            imgs = imgs.type(tensor)
            labs = labs.type(tensor)

            # draw labels in neighbourhoods of samples
            eps = torch.randn_like(labs) * sigma
            ids = [dataset.get_labelled(lb, eps=lab_gap) for lb in (labs - eps)]
            
            real_imgs = imgs
            real_labs = tensor([dataset[_][1] for _ in ids]) + eps
            fake_labs = real_labs + (torch.rand_like(real_labs) * 2 - 1) * lab_gap
            
            wrs = torch.exp(-nu * (real_labs - labs)**2)
            wfs = torch.exp(-nu * (real_labs - fake_labs)**2)

            wrs /= wrs.mean()
            wfs /= wfs.mean()

            # Sample noise as gennet input
            zs = torch.randn(bs, opt.latent_dim, device=device).type(tensor)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            disnet.requires_grad_(True)
            optimizer_D.zero_grad()

            # Generate a batch of images
            with torch.no_grad():
                fake_imgs = gennet(zs, fake_labs)

            loss_D = loss_D_CcGAN_WGAN(
                disnet,
                real_imgs, real_labs, wrs,
                fake_imgs, fake_labs, wfs, opt.lambda_gp)

            loss_D.backward()
            optimizer_D.step()

            if iters % opt.n_critic: continue

            # -----------------
            #  Train Generator
            # -----------------

            disnet.requires_grad_(False)
            optimizer_G.zero_grad()

            # Generate a batch of images
            fake_labs = sigma* torch.randn_like(labs) + labs # fake labels should correspond to real images
            fake_imgs = gennet(torch.randn_like(zs), fake_labs)

            # physical constraints
            d1, d2 = statis_constraint(real_imgs, fake_imgs)

            loss_G = loss_G_CcGAN_WGAN(disnet, fake_imgs, fake_labs) + .5**epoch * (opt.lambda_d1 * d1 + opt.lambda_d2 * d2)

            loss_G.backward()
            optimizer_G.step()

            # ---------
            #  Monitor
            # ---------

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

            check_vars(opt.workpath, real_imgs, fake_imgs)

        # save the model every epoch for resuming training
        save_for_resume(opt.workpath+'models/', epoch, gennet, disnet, optimizer_G, optimizer_D, loss_G, loss_D)
        save_current(opt.workpath + 'models/', gennet, disnet)

        # learning rate decay
        scheduler_G.step()
        scheduler_D.step()

        # scheduler2_G.step(np.mean(total_losses))
        # scheduler2_D.step(np.mean(total_losses))
        # total_losses = []

        # ---------
        # Visualize
        # ---------

        if epoch % opt.draw_every == 0:
            vel = np.concatenate((
                fake_imgs[0].detach().cpu().numpy(),
                real_imgs[0].detach().cpu().numpy()))

            with h5py.File(opt.datapath + os.listdir(opt.datapath)[0], "r") as f:
                ys = f['ys'][:]
                zs = f['zs'][:]

            gennet.eval()
            with open(opt.workpath + 'fid.dat', 'aw'[epoch==0]) as fp:
                fidr, fid1, fid0 = fid.calc(relative=True)
                fp.write('%i\t%.8e\t%.8e\t%.8e\n'%(epoch, fidr, fid1, fid0))

            draw_vel(opt.workpath + 'images/%d.png'%epoch, vel, ys, zs)
            draw_log(opt.workpath + 'log.png', opt.workpath + 'log.dat', epoch+1)
            draw_fid(opt.workpath + 'fid.png', opt.workpath + 'fid.dat')



