import numpy as np
import torch
import torch.nn as nn


def loss_D_WGAN(disnet, real_imgs, fake_imgs, lamb=0):
    ''' compute the WGAN (Arjovsky et al. 2017) loss for D '''

    # D loss for WGAN
    loss_WGAN = disnet(fake_imgs).mean() - disnet(real_imgs).mean()

    if not lamb: return loss_WGAN

    # GP: gradient penalty (Gulrajani et al. 2017)
    alpha = torch.rand(len(real_imgs), device=real_imgs.device).view(-1,1,1,1)
    inter_imgs = alpha * real_imgs + (1-alpha) * fake_imgs
    inter_imgs.requires_grad=True

    grads, = torch.autograd.grad(disnet(inter_imgs).sum(), inter_imgs, create_graph=True) # takes much memory
    loss_GP = ((grads.norm(2, dim=[*range(1, grads.dim())]) - 1)**2).mean()

    # get loss for D
    loss_D = loss_WGAN + lamb * loss_GP

    return loss_D

def loss_G_WGAN(disnet, fake_imgs):
    return - disnet(fake_imgs).mean()


def loss_D_CcGAN(disnet, real_imgs, real_labs, wrs, fake_imgs, fake_labs, wfs):
    return - torch.mean(wrs.view(-1) * torch.log(    disnet(real_imgs, real_labs).view(-1) + 1e-20)) \
           - torch.mean(wfs.view(-1) * torch.log(1 - disnet(fake_imgs, fake_labs).view(-1) + 1e-20))

def loss_G_CcGAN(disnet, fake_imgs, fake_labs):
    return - torch.log(disnet(fake_imgs, fake_labs) + 1e-20).mean()


def loss_D_CcGAN_WGAN(disnet, real_imgs, real_labs, wrs, fake_imgs, fake_labs, wfs, lamb=0):
    ## compute the WGAN-GP loss proposed by Gulrajani et al. 2017

    # D loss for WGAN
    loss_WGAN = torch.mean(wfs.view(-1) * disnet(fake_imgs, fake_labs).view(-1)) \
              - torch.mean(wrs.view(-1) * disnet(real_imgs, real_labs).view(-1))

    if not lamb: return loss_WGAN

    # GP: gradient penalty
    alpha = torch.rand(len(real_imgs)).type(real_imgs.type()).view(-1,1,1,1)

    inter_imgs = alpha * real_imgs + (1-alpha) * fake_imgs
    inter_labs = torch.ones_like(real_labs)

    inter_imgs.requires_grad = True
    inter_labs.requires_grad = False

    grads, = torch.autograd.grad(disnet(inter_imgs, inter_labs).sum(), inter_imgs, create_graph=True) # takes much memory
    loss_GP = ((grads.norm(2, dim=[*range(1, grads.dim())]) - 1)**2).mean()

    # get loss for D
    loss_D = loss_WGAN + lamb * loss_GP

    return loss_D

def loss_G_CcGAN_WGAN(disnet, fake_imgs, fake_labs):
    return - disnet(fake_imgs, fake_labs).mean()


def loss_Hinge(disnet, real_imgs, fake_imgs):
    zero = torch.tensor(0, device=real_imgs.device)
    return torch.maximum(zero, 1 - disnet(real_imgs)).mean() + torch.maximum(zero, 1 + disnet(fake_imgs)).mean()



