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
            nn.Conv2d(in_features, out_features, 1)

    def forward(self, input):
        out = self.module(input)
        bps = self.bypass(input)
        resize = nn.AdaptiveAvgPool2d(out.size()[-2:])
        return out + resize(bps)

def conv1x1(in_features, out_features):
    return nn.Conv2d(in_features, out_features, 1)

def conv3x3(in_features, out_features):
    # implement padding by hand: periodic for spanwise, zero for top & bottom
    # note: (l, r, t, b) for tensor, (l, r, b, t) for flow field
    return nn.Sequential(
        nn.ReplicationPad2d((1, 1, 0, 0)),
        nn.ZeroPad2d((0, 0, 1, 1)),
        nn.Conv2d(in_features, out_features, 3),
    )

def defReLU():
    # default leaky ReLU activation
    return nn.LeakyReLU(.2, inplace=True)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()

        def pixel_norm(x):
            return x / (torch.mean(x**2, dim=1, keepdim=True) + 1e-12)**.5

        def pnLReLU():
            # leaky ReLu with pixelwise normalization, negative slope 0.2
            return nn.Sequential(Lambda(pixel_norm), defReLU())

        def FCBlock(latent_dim, start_size):
            # inlet block to handle the latent vector input
            return nn.Sequential(
                nn.Linear(latent_dim, latent_dim * start_size**2),
                nn.Unflatten(1, (latent_dim, start_size, start_size)),
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
                nn.Upsample(scale_factor=2),
                conv3x3(in_features, out_features),
                pnLReLU(),
                conv3x3(out_features,out_features),
                )
            return ResNet(module)

        def ExBlock(in_features):
            return nn.Sequential(
                pnLReLU(),
                conv1x1(in_features, 3)
                )

        self.inlet = \
            FCBlock(latent_dim,    img_size//2**5)

        self.model = nn.Sequential(
            InBlock(latent_dim,    latent_dim),
            MyBlock(latent_dim,    latent_dim),    # channels 256 -> 256
            MyBlock(latent_dim,    latent_dim),    # channels 256 -> 256
            MyBlock(latent_dim,    latent_dim),    # channels 256 -> 256
            MyBlock(latent_dim,    latent_dim//2), # channels 256 -> 128
            MyBlock(latent_dim//2, latent_dim//4), # channels 128 -> 64
            ExBlock(latent_dim//4),                # return the generated images
            )

        # initialize and apply equalized learning rate
        setModule(self.inlet, eqlr=True)
        setModule(self.model, eqlr=True)

    def forward(self, zs, ys=None):
        if ys is None: return self.model(self.inlet(zs))

    # def forward(self, zs, ys):
    #     out = self.inlet(zs)
    #     out = out + ys.view(-1, *([1] * (out.dim()-1)))
    #     out = self.model(out)
    #     return out


class Discriminator(nn.Module):
    def __init__(self, img_size, final_dim):
        super().__init__()

        def batch_std(x):
            # minibatch standard deviation to improve variety (Karras et al. 2018)
            std_map = torch.ones_like(x[:,:1]) * torch.std(x, dim=0).mean()
            return torch.cat((x, std_map), dim=1)

        def InBlock(out_features):
            return conv1x1(3, out_features)

        def MyBlock(in_features, out_features):
            # repeated block, each block shrinks the image by 1/2
            module = nn.Sequential(
                defReLU(),
                conv3x3(in_features, in_features),
                defReLU(),
                conv3x3(in_features,out_features),
                nn.AvgPool2d(2),
                )
            return ResNet(module)

        def ExBlock(in_features, final_size):
            return nn.Sequential(
                defReLU(),
                # Lambda(batch_std),
                conv3x3(in_features, in_features),
                defReLU(),
                nn.Conv2d(in_features, in_features, final_size), # Equal to nn.Linear(in_features * final_size**2, in_features),
                defReLU(),
                nn.Flatten(),
                )

        def FCBlock1(in_features):
            return nn.Linear(in_features, 1)

        def FCBlock2(out_features):
            return nn.Linear(1, out_features, bias=False)

        def FinalAct():
            return nn.Identity() # nn.Sigmoid() # not needed in WGAN, where the discrimintator becomes a critic

        self.model = nn.Sequential(
            InBlock(final_dim//4),
            MyBlock(final_dim//4,final_dim//2),
            MyBlock(final_dim//2,final_dim),
            MyBlock(final_dim,   final_dim),
            MyBlock(final_dim,   final_dim),
            MyBlock(final_dim,   final_dim),
            ExBlock(final_dim,   img_size//2**5),
            )

        self.linear1 = FCBlock1(final_dim)
        self.linear2 = FCBlock2(final_dim)
        self.outlet  = FinalAct()

        setModule(self.model, eqlr=True)
        setModule(self.linear1, eqlr=True)
        setModule(self.linear2, eqlr=True)
        setModule(self.outlet, eqlr=True)

    def forward(self, xs, ys=None):
        if ys is None: return self.outlet(self.linear1(self.model(xs)))


    # def forward(self, xs, ys):
    #     out = self.model(xs)
    #     out_y = self.linear2(ys + 1) # why +1 ? (following GitHub code of Ding et al. 2021)
    #     out = self.outlet(self.linear1(out) + (out * out_y).sum(dim=-1))
    #     return out

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




