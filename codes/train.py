import numpy as np
import h5py
import torch

from fid import FID, wrapped_dl_gen
from plots import draw_vel, draw_log, draw_fid
from recorder import save_for_resume, load_for_resume, save_current
from losses import loss_D_WGAN, loss_G_WGAN, loss_D_WACGAN, loss_G_WACGAN


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

class LabelRandomizer:
    def __init__(self):
        self.sigma = 0
        self.nu = 1
        self.lab_gap = 0

    def on(self, dataset):
        self.thres_w = 1e-3
        self.sigma = (4/3/len(dataset))**.2 * np.std(np.unique(dataset.get_labels()))
        self.nu = (20 * np.diff(dataset.get_labels()).max())**-2 # 20 means at least neighbouring 40 labels can contribute to the conditional distribution
        self.lab_gap = (-np.log(self.thres_w)/self.nu)**.5
        self.show()

    def show(self):
        print('\nCcGAN parameters:')
        print('sigma =', self.sigma)
        print('nu =', self.nu)
        print('thres_w =', self.thres_w)
        print('lab_gap =', self.lab_gap)
        print()


class Trainer:
    def __init__(self, gennet, disnet, dataloader, opt):
        self.gennet = gennet
        self.disnet = disnet
        self.datldr = dataloader
        self.opt = opt

        self.config()

    def config(self):
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.gennet.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
        self.optimizer_D = torch.optim.Adam(self.disnet.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))

        # load in resume files
        epoch = self.opt.resume
        if epoch < 0:
            print('\nTrain from scratch.\n')
        else:
            load_for_resume(self.opt.workpath+'models/', epoch, self.gennet, self.disnet, self.optimizer_G, self.optimizer_D)


        # Judge for measuring how well the generator is learning
        self.fid = FID(
            self.datldr,
            wrapped_dl_gen(self.gennet.module, self.opt.latent_dim, self.opt.batch_size),
            self.opt.workpath + 'models/inception_v3.pt',
        )
        # Decay of learning rate
        sched = lambda ep: 2**-int(np.log2(1+ep//10))
        # sched = lambda ep: 1/(1+(ep//10))
        # sched = lambda ep: .9**-(ep//(10 * int(self.opt.n_critic**.5+.5))) # equivalent to StepLR

        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, sched, last_epoch=self.opt.resume, verbose=True)
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, sched, last_epoch=self.opt.resume, verbose=True)

        # stochastic perturbation on regression labels
        self.labrdm = LabelRandomizer()

    def train(self, train_epoch):

        epoch = self.opt.resume

        while epoch < self.opt.epochs:
            epoch += 1

            # ----------
            #  Training
            # ----------

            self.gennet.train()
            self.disnet.train()
            self.datldr.batch_sampler.sampler.set_epoch(epoch)

            train_epoch(epoch, self.gennet, self.disnet, self.datldr)

            # save the model every epoch for resuming training
            save_for_resume(self.opt.workpath+'models/', epoch, self.gennet, self.disnet, self.optimizer_G, self.optimizer_D)
            save_current   (self.opt.workpath+'models/',        self.gennet, self.disnet)

            # learning rate decay
            self.scheduler_G.step()
            self.scheduler_D.step()

            # ---------
            # Visualize
            # ---------

            if epoch % self.opt.check_every == 0 == torch.distributed.get_rank():
                self.gennet.eval()

                with open(self.opt.workpath + 'fid.dat', 'aw'[epoch==0]) as fp:
                    fidr, fid1, fid0 = self.fid.calc(relative=True)
                    fp.write('%i\t%.8e\t%.8e\t%.8e\n'%(epoch, fidr, fid1, fid0))

                vel = np.concatenate((
                    self.gennet.module.getone(self.opt.latent_dim).detach().cpu().numpy(),
                    next(iter(self.datldr))[0][0].detach().cpu().numpy(),
                    ))
                draw_vel(self.opt.workpath + 'images/%d.png'%epoch, vel)
                draw_log(self.opt.workpath + 'log.png', self.opt.workpath + 'log.dat', epoch+1)
                draw_fid(self.opt.workpath + 'fid.png', self.opt.workpath + 'fid.dat')


    def train_unsuper(self, epoch, gennet, disnet, dataloader):
        # train gennet and disnet for one epoch using the dataset provided by dataloader
        for itera, (imgs, labs) in enumerate(dataloader):

            niters = epoch * len(dataloader) + itera

            bs = len(imgs) # size of the current batch, may be different for the last batch

            real_imgs = imgs

            # Sample noise as gennet input
            zs = torch.randn(bs, self.opt.latent_dim).type(imgs.type()).to(imgs.device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            disnet.requires_grad_(True)
            self.optimizer_D.zero_grad()

            # Generate a batch of images
            with torch.no_grad():
                fake_imgs = gennet(zs)

            loss_D, \
            loss_D_adv, \
            loss_D_gp = loss_D_WGAN(disnet, real_imgs, fake_imgs, self.opt.lambda_gp)

            loss_D.backward()
            self.optimizer_D.step()

            if niters % self.opt.n_critic: continue

            # -----------------
            #  Train Generator
            # -----------------

            disnet.requires_grad_(False)
            self.optimizer_G.zero_grad()

            # Generate a batch of images
            fake_imgs = gennet(torch.randn_like(zs))

            # physical constraints
            d1, d2 = statis_constraint(real_imgs, fake_imgs)

            loss_G_adv, = loss_G_WGAN(disnet, fake_imgs)
            loss_G = loss_G_adv + .5**epoch * self.opt.lambda_d2 * (d1 + d2)

            loss_G.backward()
            self.optimizer_G.step()

            # ---------
            #  Monitor
            # ---------

            if torch.distributed.get_rank(): continue

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"%(
                epoch,
                self.opt.epochs,
                itera,
                len(dataloader),
                loss_D.item(),
                loss_G.item()
                ))

            with open(self.opt.workpath + 'log.dat', 'aw'[niters==0]) as fp:
                fp.write(
                    '%i\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\n'%(
                    niters,
                    loss_D.item(),
                    loss_G.item(),
                    d1.item(),
                    d2.item(),
                    loss_D_adv.item(), loss_D_gp.item(),
                    loss_G_adv.item(),
                    ))

            # check_vars(self.opt.workpath, real_imgs, fake_imgs)

    def train_super(self, epoch, gennet, disnet, dataloader):

        for itera, (imgs, labs) in enumerate(dataloader):

            niters = epoch * len(dataloader) + itera

            bs = len(imgs) # size of the current batch, may be different for the last batch

            real_imgs = imgs
            real_labs = labs

            # Sample noise as gennet input
            zs = torch.randn(bs, self.opt.latent_dim).type(imgs.type()).to(imgs.device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            disnet.requires_grad_(True)
            self.optimizer_D.zero_grad()

            # Generate a batch of images
            with torch.no_grad():
                fake_labs = real_labs # torch.rand_like(real_labs)
                fake_imgs = gennet(zs, fake_labs, bool(self.opt.lambda_d1))

            loss_D, loss_D_adv, loss_D_aux, loss_D_gp = loss_D_WACGAN(
                disnet, real_imgs, fake_imgs,
                        real_labs, fake_labs, lamb_aux=self.opt.lambda_d1, lamb_gp=self.opt.lambda_gp)

            loss_D.backward()
            self.optimizer_D.step()

            if niters % self.opt.n_critic: continue

            # -----------------
            #  Train Generator
            # -----------------

            disnet.requires_grad_(False)
            self.optimizer_G.zero_grad()

            # Generate a batch of images
            # fake_labs = torch.randn_like(fake_labs) # fake labels should correspond to real images
            fake_imgs = gennet(torch.randn_like(zs), fake_labs, bool(self.opt.lambda_d1))

            # physical constraints
            d1, d2 = statis_constraint(real_imgs, fake_imgs)

            loss_G, loss_G_adv, loss_G_aux = loss_G_WACGAN(disnet, fake_imgs, fake_labs, lamb_aux=self.opt.lambda_d1)
            loss_G = loss_G + .5**epoch * self.opt.lambda_d2 * (d1 + d2)

            loss_G.backward()
            self.optimizer_G.step()

            # ---------
            #  Monitor
            # ---------

            if torch.distributed.get_rank(): continue

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]" % (
                epoch,
                self.opt.epochs,
                itera,
                len(dataloader),
                loss_D.item(),
                loss_G.item()
                ))

            with open(opt.workpath + 'log.dat', 'aw'[niters==0]) as fp:
                fp.write(
                    '%i\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\n'%(
                    niters,
                    loss_D.item(),
                    loss_G.item(),
                    d1.item(),
                    d2.item(),
                    loss_D_adv.item(), loss_D_aux.item(), loss_D_gp.item(),
                    loss_G_adv.item(), loss_G_aux.item(),
                    ))

            # check_vars(self.opt.workpath, real_imgs, fake_imgs)



