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

        fn = SpecNorm(name)
        fn.compute_weight(module)
        module.register_forward_pre_hook(fn)

    def compute_weight(self, module):
        w = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u').to(w.device) if \
            hasattr(module, self.name + '_u') else \
            torch.randn(w.size(0), 1, requires_grad=False, device=w.device) * .1

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
        if not hasattr(m, 'weight'): return

        if eqlr:
            m.weight.data.normal_(mean=0, std=1)
            m.bias.data.zero_()
        else:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()

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

def PadConv(in_features, out_features):
    # implement padding by hand: periodic for spanwise, zero for top & bottom
    # note: (l, r, t, b) for tensor, (l, r, b, t) for flow field
    return nn.Sequential(
        nn.ReplicationPad2d((1, 1, 0, 0)),
        nn.ZeroPad2d((0, 0, 1, 1)),
        nn.Conv2d(in_features, out_features, 3),
    )


class Generator(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()

        def pixel_norm(x):
            return x / (torch.mean(x**2, dim=1, keepdim=True) + 1e-12)**.5

        def PixNormLRelu():
            # leaky ReLu with pixelwise normalization, negative slope 0.2
            return nn.Sequential(
                Lambda(pixel_norm),
                nn.LeakyReLU(.2),
                )

        def MyBlock(in_features, out_features):
            # repeated block, each block enlarges the image by 2
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                PadConv(in_features, out_features), PixNormLRelu(),
                PadConv(out_features,out_features), PixNormLRelu(),
                )

        def InBlock(latent_dim, start_size):
            # inlet block to handle the latent vector input
            return nn.Sequential(
                nn.Linear(latent_dim, latent_dim * start_size**2),
                nn.Unflatten(1, (latent_dim, start_size, start_size)),
                PixNormLRelu(),
                PadConv(latent_dim, latent_dim),
                PixNormLRelu(),
                )

        self.model = nn.Sequential(
            InBlock(latent_dim,    img_size//2**5),
            MyBlock(latent_dim,    latent_dim),    # channels 256 -> 256
            MyBlock(latent_dim,    latent_dim),    # channels 256 -> 256
            MyBlock(latent_dim,    latent_dim),    # channels 256 -> 256
            MyBlock(latent_dim,    latent_dim//2), # channels 256 -> 128
            MyBlock(latent_dim//2, latent_dim//4), # channels 128 -> 64
            nn.Conv2d(latent_dim//4, 3, 1),
            )

        # initialize and apply equalized learning rate
        setModule(self.model, eqlr=True)

    def forward(self, zs):
        # return the generated images
        return self.model(zs)


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        def batch_std(x):
            # minibatch standard deviation to improve variety (Karras et al. 2018)
            std_map = torch.ones_like(x[:,:1]) * torch.std(x, dim=0).mean()
            return torch.cat((x, std_map), dim=1)

        def MyBlock(in_features, out_features):
            # repeated block, each block shrinks the image by 1/2
            return nn.Sequential(
                PadConv(in_features, in_features), nn.LeakyReLU(.2),
                PadConv(in_features,out_features), nn.LeakyReLU(.2),
                nn.AvgPool2d(2),
                )

        self.iphys = nn.Sequential(
            nn.Conv2d(3, 64, 1), nn.LeakyReLU(.2),
            MyBlock(64, 128),
            MyBlock(128,256),
            MyBlock(256,256),
            MyBlock(256,256),
            MyBlock(256,256),
            # Lambda(batch_std),
            PadConv(256, 256), nn.LeakyReLU(.2),
            nn.Flatten(),
            )

        # self.ispec = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear( img_size**2 * 3, (img_size//8)**2), nn.LeakyReLU(.2),
        #     nn.Linear((img_size//8)**2, (img_size//32)**2),nn.LeakyReLU(.2),
        #     )

        self.model = nn.Sequential(
            nn.Linear(256 * (img_size//32)**2, 256),
            nn.LeakyReLU(.2),
            nn.Linear(256, 1),
            # nn.Sigmoid(), # not needed in WGAN, where the discrimintator becomes a critic
            )

        setModule(self.iphys, eqlr=True)
        # setModule(self.ispec, eqlr=True)
        setModule(self.model, eqlr=True)

    def forward(self, imgs):
        ophys = self.iphys(imgs)
        return self.model(ophys)
        # ospec = self.ispec(self.spec(imgs))
        # return self.model(torch.cat((ophys, ospec), dim=-1))

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




