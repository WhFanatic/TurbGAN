import numpy as np
import torch
import torch.nn as nn


def gradient_penalty(disnet, real_imgs, fake_imgs):
    # compute GP (gradient penalty) for WGAN-GP (Gulrajani et al. 2017)
    alpha = torch.rand(len(real_imgs)).type(real_imgs.type()).view(-1,1,1,1)

    inter_imgs = alpha * real_imgs + (1-alpha) * fake_imgs
    inter_imgs.requires_grad = True

    grads, = torch.autograd.grad(disnet(inter_imgs).sum(), inter_imgs, create_graph=True) # takes much memory
    loss_gp = ((grads.norm(2, dim=[*range(1, grads.dim())]) - 1)**2).mean()

    return loss_gp


def loss_D_WGAN(disnet, real_imgs, fake_imgs, lamb=0):
    ''' compute the WGAN-GP loss proposed by Gulrajani et al. (2017) '''

    # D loss for WGAN (Arjovsky et al. 2017)
    loss_WGAN = disnet(fake_imgs).mean() - disnet(real_imgs).mean()

    if not lamb: return loss_WGAN

    loss_GP = gradient_penalty(disnet, real_imgs, fake_imgs)

    return loss_WGAN + lamb * loss_GP, \
           loss_WGAN, \
           loss_GP

def loss_G_WGAN(disnet, fake_imgs):
    return - disnet(fake_imgs).mean(),


def loss_D_WACGAN(disnet, real_imgs,      fake_imgs,
                          real_labs=None, fake_labs=None,
                          wrs=1,          wfs=1,    lamb_aux=0, lamb_gp=0):
    # output format depend on whether label arguments are None
    if real_labs is None and fake_labs is None:
        return loss_D_WGAN(disnet, real_imgs, fake_imgs, lamb_gp)

    # D loss for WGAN
    fake_diss            = disnet(fake_imgs)
    real_diss, real_auxs = disnet(real_imgs, real_labs, supervised=True)

    loss_adv = torch.mean(torch.tensor(wfs).view(-1) * fake_diss.view(-1)) \
             - torch.mean(torch.tensor(wrs).view(-1) * real_diss.view(-1))
    loss_aux = 0
    loss_gp  = 0

    if lamb_aux: # auxiliary loss
        loss_aux = torch.mean((real_labs - real_auxs)**2)

    if lamb_gp: # GP loss (label by AC is not involved in GP, because loss_aux is only a regularizer and has no part in the competition of D and G)
        loss_gp = gradient_penalty(disnet, real_imgs, fake_imgs)

    return loss_adv + lamb_aux * loss_aux + lamb_gp * loss_gp, \
           loss_adv, \
           loss_aux, \
           loss_gp

def loss_G_WACGAN(disnet, fake_imgs, fake_labs=None, lamb_aux=0):
    # output format depend on whether fake_labs is None
    if fake_labs is None:
        return loss_G_WGAN(disnet, fake_imgs)

    if lamb_aux:
        fake_diss, fake_auxs = disnet(fake_imgs, fake_labs)
    else:
        fake_diss, fake_auxs = disnet(fake_imgs),fake_labs

    loss_adv = torch.mean(-fake_diss)
    loss_aux = torch.mean((fake_labs - fake_auxs)**2)

    return loss_adv + lamb_aux * loss_aux, \
           loss_adv, \
           loss_aux




def loss_D_CcGAN(disnet, real_imgs, real_labs, wrs, fake_imgs, fake_labs, wfs):
    return - torch.mean(wrs.view(-1) * torch.log(    disnet(real_imgs, real_labs).view(-1) + 1e-20)) \
           - torch.mean(wfs.view(-1) * torch.log(1 - disnet(fake_imgs, fake_labs).view(-1) + 1e-20))

def loss_G_CcGAN(disnet, fake_imgs, fake_labs):
    return - torch.log(disnet(fake_imgs, fake_labs) + 1e-20).mean()


def loss_Hinge(disnet, real_imgs, fake_imgs):
    zero = torch.tensor(0, device=real_imgs.device)
    return torch.maximum(zero, 1 - disnet(real_imgs)).mean() + torch.maximum(zero, 1 + disnet(fake_imgs)).mean()



