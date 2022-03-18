import numpy as np
import torch
import torch.nn as nn
from torch.fft import fft, rfft


class EqualLR:
    ''' Apply equalized learning rate to a nn.Module,
        proposed by Karras et al. (2018),
        explained here: https://personal-record.onrender.com/post/equalized-lr/ '''
    def __init__(self, name):
        self.name = name

    def __call__(self, module, input):
        self.compute_weight(module)

    @staticmethod
    def apply(module, name):
        w = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(w.data))
        module.register_forward_pre_hook(EqualLR(name))

    def compute_weight(self, module):
        w = getattr(module, self.name + '_orig')
        fan_out, fan_in = w.size()[:2]
        for d in w.size()[2:]:
            fan_in *= d
            fan_out *= d
        w = w * (2 / fan_in)**.5
        setattr(module, self.name, w)

class SpecNorm:
    def __init__(self, name):
        self.name = name

    def __call__(self, module, input):
        self.compute_weight(module)

    @staticmethod
    def apply(module, name):
        w = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(w.data))
        module.register_buffer(name + '_u', torch.randn(w.size(0), 1, device=w.device) * .1)

        fn = SpecNorm(name)
        fn.compute_weight(module)
        module.register_forward_pre_hook(fn)

    def compute_weight(self, module):
        w = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')

        # spectral normalization of the weight matrix
        w_sn = w.contiguous().view(w.size(0), -1)
        v = w_sn.T @ u
        v = v / (v.norm() + 1e-12)
        u = w_sn @ v
        u = u / (u.norm() + 1e-12)
        w_sn = w_sn / (u.T @ w_sn @ v)
        w_sn = w_sn.view(*w.size())

        # # equalized lr: rescale the weight by size
        # # remember to use normal init if eqlr is activated
        # fan_out, fan_in = w.size()[:2]
        # for d in w.size()[2:]:
        #     fan_in *= d
        #     fan_out *= d
        # w_sn = w_sn * (2 / fan_in)**.5

        setattr(module, self.name, w_sn)
        setattr(module, self.name + '_u', u.data)

def setModule(module, eqlr=False, specnorm=False):
    def func(m):
        if hasattr(m, 'bias') and getattr(m, 'bias') is not None:
            m.bias.data.zero_()

        if not hasattr(m, 'weight'): return

        if eqlr:
            m.weight.data.normal_(mean=0, std=1)
        else:
            nn.init.kaiming_normal_(m.weight)

        if eqlr:
            EqualLR.apply(m, 'weight')
        elif specnorm:
            SpecNorm.apply(m, 'weight')

    module.apply(func)
    return module


class Lambda(nn.Module):
    ''' define custom non-parametric layers '''
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, input):
        return self.func(input)

class ResNet(nn.Module):
    # add shortcut connection to the two ends of a convolutional block
    def __init__(self, module):#, in_features=None, out_features=None):
        super().__init__()

        # automatically infer in/out_features of the conv block
        # use identity mapping whenever possible
        for i, p in enumerate(module.parameters()):
            in_features = p.data.size(1) if i == 0 else in_features
            out_features= p.data.size(0)

        self.module = module
        self.bypass = nn.Identity() \
            if in_features == out_features else \
            nn.Conv2d(in_features, out_features, 1, bias=False)

    def forward(self, input):
        out = self.module(input)
        bps = self.bypass(input)
        resize = nn.AdaptiveAvgPool2d(out.size()[-2:])
        return out + resize(bps)

def conv1x1(in_features, out_features):
    return nn.Conv2d(in_features, out_features, 1)

def conv3x3(in_features, out_features, periodic=(False, False), **kwargs):
    # implement padding by hand: periodic for (spanwise, wall-normal)
    # note: (l, r, t, b) for tensor, (l, r, b, t) for flow field
    return nn.Sequential(
        nn.ReplicationPad2d((1,1,0,0)) if periodic[0] else nn.ZeroPad2d((1,1,0,0)), # spanwise
        nn.ReplicationPad2d((0,0,1,1)) if periodic[1] else nn.ZeroPad2d((0,0,1,1)), # wall-normal
        nn.Conv2d(in_features, out_features, 3, **kwargs),
    )

def defReLU():
    # default leaky ReLU activation
    return nn.LeakyReLU(.2, inplace=True)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, img_chan=3):
        super().__init__()

        def pixel_norm(x):
            return x / (torch.mean(x**2, dim=1, keepdim=True) + 1e-12)**.5

        def pnLReLU():
            # leaky ReLu with pixelwise normalization, negative slope 0.2
            return nn.Sequential(Lambda(pixel_norm), defReLU())

        def FCBlock(latent_dim, start_chan, start_size):
            # inlet block to handle the latent vector input
            return nn.Sequential(
                nn.Linear(latent_dim, start_chan * start_size**2),
                nn.Unflatten(1, (start_chan, start_size, start_size)),
                )

        def InBlock(in_features, out_features):
            module = nn.Sequential(
                pnLReLU(),
                conv3x3(in_features, out_features),
                )
            return ResNet(module)

        def MyBlock(in_features, out_features):
            # repeated block, each block enlarges the image by 2
            module = nn.Sequential(
                pnLReLU(),
                nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
                conv3x3(in_features, out_features),
                pnLReLU(),
                conv3x3(out_features,out_features),
                )
            return ResNet(module)

        def ExBlock(in_features):
            return nn.Sequential(
                pnLReLU(),
                conv1x1(in_features, img_chan)
                )

        nblocks    = int(np.log2(img_size//4)) # at least 4x4 for start size
        start_size = img_size//2**nblocks
        start_chan = 4 * latent_dim

        # ## version 3, structures adapted from Kim & Lee (JCP 2020) and PG-GAN of Karras et al. (ICLR, 2018)
        # self.inlet = \
        #     FCBlock(latent_dim, start_chan, start_size) # channels 128->512, size 1x1->4x4

        # self.model = nn.Sequential(
        #     InBlock(start_chan, start_chan), *[  # channels 512->512, size 4x4->4x4
        #     MyBlock(start_chan//2**max(n-1,0),   # channels 512->64,  size 4x4->64x64
        #             start_chan//2**n) for n in range(nblocks)],
        #     ExBlock(start_chan//2**(nblocks-1)), # channels 64->3,    size 64x64->64x64, the generated images
        #     )

        ## version 4, structure as in Ding et al. (2021) https://github.com/UBCDingXin/improved_CcGAN
        start_chan = 1024
        assert nblocks >= 4, 'figure size too small'

        self.inlet = \
            FCBlock(latent_dim, start_chan, start_size) # channels 128->1024, size 1x1->4x4
        self.model = nn.Sequential(*[
            MyBlock(start_chan//2**max(3-n,0),          # channels 1024->64, size 4x4->64x64
                    start_chan//2**max(4-n,0)) for n in range(nblocks)[::-1]],
            InBlock(start_chan//2**4, img_chan),        # channels 64->3, size 64x64
            )

        # initialize and apply equalized learning rate
        setModule(self.inlet, eqlr=True)
        setModule(self.model, eqlr=True)

    def forward(self, zs, ys=None, supervised=None):
        if supervised is None:
            supervised = (ys is not None)

        # determining what to return by the argument 'supervised' in stead of by the existence of argument 'ys'
        # allows more formats of supervision with different outputs
        if not supervised:
            return self.model(self.inlet(zs))

        # conditionalize as InfoGAN (Chen et al. 2016)
        return self.model(self.inlet(torch.cat([ys.view(1,-1), zs.T[1:]]).T))

        # # conditionalize as CcGAN (Ding et al. 2021)
        # out = self.inlet(zs)
        # out = out + ys.view(-1, *([1] * (out.dim()-1)))
        # out = self.model(out)
        # return out

    def getone(self, latent_dim):
        # generate one image without label, latent vector automatically generated
        with torch.no_grad():
            return self(torch.randn(latent_dim).view(1,-1).to(next(self.parameters()).device))[0]


class Discriminator(nn.Module):
    def __init__(self, img_size, latent_dim, img_chan=3):
        super().__init__()

        def batch_std(x):
            # minibatch standard deviation to improve variety (Karras et al. 2018)
            std_map = torch.ones_like(x[:,:1]) * torch.std(x, dim=0).mean()
            return torch.cat((x, std_map), dim=1)

        def InBlock(out_features):
            return conv1x1(img_chan, out_features)

        def MyBlock(in_features, out_features, order='iiio', in_act=True, pool=2):
            # repeated block, each block shrinks the image by 1/2
            features = {
                'i': in_features,
                'o': out_features,
                }
            module = nn.Sequential(
                defReLU() if in_act else nn.Identity(),
                conv3x3(features[order[0]], features[order[1]]),
                defReLU(),
                conv3x3(features[order[2]], features[order[3]]),
                nn.AvgPool2d(pool) if pool>1 else nn.Identity(),
                )
            return ResNet(module)

        def ExBlock(final_chan, final_size):
            return nn.Sequential(
                defReLU(),
                # Lambda(batch_std),
                conv3x3(final_chan, final_chan),
                defReLU(),
                nn.Conv2d(final_chan, final_chan, final_size),
                defReLU(),
                nn.Flatten(),
                )

        def FCBlock1(in_features):
            return nn.Linear(in_features, 1)

        def FCBlock2(out_features):
            return nn.Linear(1, out_features, bias=False)

        def FinalAct1():
            return nn.Sequential(
                nn.Identity(),
                # nn.Sigmoid(), # not needed in WGAN, where the discrimintator becomes a critic
                )
        def FinalAct2():
            return nn.Sequential(
                nn.Identity(),
                # nn.Sigmoid(), # output label that have been normalized to 0~1
                )

        nblocks    = int(np.log2(img_size//4)) # at least 4x4 for start size
        final_size = img_size//2**nblocks
        final_chan = 4 * latent_dim

        # ## version 3, structures adapted from Kim & Lee (JCP 2020) and PG-GAN of Karras et al. (ICLR, 2018)
        # self.model = nn.Sequential(
        #     InBlock(final_chan//2**(nblocks-1)), *[ # channels 3->64, size 64x64->64x64
        #     MyBlock(final_chan//2**n,               # channels 64->512, size 64x64->4x4
        #             final_chan//2**max(n-1,0)) for n in range(nblocks)[::-1]],
        #     ExBlock(final_chan, final_size), # channels 512->128, size 4x4->1x1
        #     )
        # self.linear1 = FCBlock1(final_chan)
        # self.linear2 = FCBlock1(final_chan) # FCBlock1 for ACGAN (Odena et al. 2017), FCBlock2 for Projection D (Miyato & Koyama 2018)
        # self.outlet1 = FinalAct1()
        # self.outlet2 = FinalAct2()
        # self.outlet = self.outlet1
        # self.linear = self.linear1

        ## version 4, structure as in Ding et al. (2021) https://github.com/UBCDingXin/improved_CcGAN
        final_chan = 1024
        assert nblocks >= 4, 'figure size too small'

        self.model = nn.Sequential(
            MyBlock(img_chan,                                            # channels 3->64, size 64x64->32x32
                    final_chan//2**4,          order='iooo', in_act=False), *[
            MyBlock(final_chan//2**max(4-n,0),                           # channels 64->1024, size 32x32->4x4
                    final_chan//2**max(3-n,0), order='iooo', pool=2*(n+1<nblocks)) for n in range(nblocks)],
            defReLU(),
            nn.AvgPool2d(final_size), Lambda(lambda a: final_size**2*a), # channels 1024, size 4x4->1x1
            nn.Flatten(),
            )
        self.linear = FCBlock1(final_chan)
        self.outlet = FinalAct1()
        # self.outlet1 = self.outlet
        # self.linear1 = self.linear
        # self.linear2 = FCBlock1(final_chan) # FCBlock1 for ACGAN (Odena et al. 2017), FCBlock2 for Projection D (Miyato & Koyama 2018)
        # self.outlet2 = FinalAct2()

        setModule(self.model, eqlr=True)
        setModule(self.linear, eqlr=True)
        setModule(self.outlet, eqlr=True)
        # setModule(self.linear2, eqlr=True)
        # setModule(self.outlet2, eqlr=True)

    def forward(self, xs, ys=None, supervised=None):
        if supervised is None:
            supervised = (ys is not None)

        # determining what to return by the argument 'supervised' in stead of by the existence of argument 'ys'
        # allows more formats of supervision with different outputs
        if not supervised:
            return self.outlet(self.linear(self.model(xs)))

        # conditionalize as ACGAN (Odena et al. 2017)
        out = self.model(xs)
        dis = self.outlet1(self.linear1(out))
        aux = self.outlet2(self.linear2(out))
        return dis, aux

        # # conditionalize as Projection D (Miyato & Koyama 2018; Ding et al. 2021)
        # out = self.model(xs)
        # out_y = self.linear2(ys.view(-1,1)) # note the GitHub code of Ding et al. 2021 puts an '+1' here
        # out = self.outlet(self.linear1(out) + (out * out_y).sum(dim=-1, keepdim=True))
        # return out

    def spec(self, imgs):
        ''' get the spectral representation of the images,
            Chebyshev in y direction, Fourier in z direction,
            use rfft and separate amplitudes and phases '''
        if not hasattr(self, 'chebmat'):
            PI = torch.arccos(torch.tensor(-1.))
            ny = imgs.size(-2)
            indexs = torch.arange(ny, device=imgs.device)
            points = 1 - torch.cos(PI/2 * indexs/(ny-1))
            self.chebmat = torch.cos(indexs * torch.arccos(points).view(-1,1))
            self.chebmat = torch.inverse(self.chebmat)
            self.chebmat.requires_grad = False

        fimgs = rfft(self.chebmat @ imgs)
        ampls = (fimgs.real**2 + fimgs.imag**2).add(1e-12).log()
        angls = torch.arctan(fimgs.imag / fimgs.real.add(1e-12)) # torch.angle(fimgs)
        fimgs = torch.cat((ampls, angls.T[1:1+imgs.size(-1)-ampls.size(-1)].T), dim=-1)
        return fimgs




