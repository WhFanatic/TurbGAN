import torch
import torch.nn as nn


class EqualLR:
    ''' Apply equalized learning rate to a nn.Module,
        proposed by Karras et al. (2018),
        explained here: https://personal-record.onrender.com/post/equalized-lr/ '''
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * (2 / fan_in)**.5

    @staticmethod
    def apply(module, name):
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(EqualLR(name))

        return module

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


class Equalized(nn.Module):
    def __init__(self, module):
        super().__init__()

        module.weight.data.normal_(mean=0, std=1)
        module.bias.data.zero_()

        self.module = EqualLR.apply(module, 'weight')

    def forward(self, x):
        return self.module(x)


class Lambda(nn.Module):
    ''' define custom non-parametric layers '''
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def conv_with_padding(in_features, out_features):
    return [
        # implement padding by hand: periodic for spanwise, reflect for top, zero for bottom
        nn.ReplicationPad2d((1, 1, 0, 0)),
        nn.ZeroPad2d((0, 0, 1, 1)),
        # nn.ReflectionPad2d((0, 0, 0, 1)), # top for flow field, bottom for tensor
        # nn.ZeroPad2d((0, 0, 1, 0)), # bottom for flow field, top for tensor
        Equalized(nn.Conv2d(in_features, out_features, 3)),
    ]

class Generator(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()

        def pixel_norm(x):
            return x / (torch.mean(x**2, dim=-3, keepdim=True) + 1e-8)**.5

        def relu_with_pixnorm():
            return [
                Lambda(pixel_norm),
                nn.LeakyReLU(.2),
            ]

        def repeat_block(in_features, out_features):
            return [
                nn.Upsample(scale_factor=2) ] + \
                conv_with_padding(in_features, out_features) + \
                relu_with_pixnorm() + \
                conv_with_padding(out_features, out_features) + \
                relu_with_pixnorm()

        # the starting size of feature maps at the entrance of conv blocks
        start_size = img_size // 2**3 # 2**5 # the start_size turns into img_size through 5 Upsample

        self.model = nn.Sequential(
            # inlet block to handle the latent vector input
            Equalized(nn.Linear(latent_dim, 256 * start_size**2)),
            nn.Unflatten(1, (256, start_size, start_size)),
            *relu_with_pixnorm(),
            *conv_with_padding(256, 256),
            *relu_with_pixnorm(), # should be a relu layer here?

            # repeated blocks
            *repeat_block(256, 256),
            # *repeat_block(256, 256),
            # *repeat_block(256, 256),
            *repeat_block(256, 128),
            *repeat_block(128, 64),

            # outlet block to finalize the output
            Equalized(nn.Conv2d(64, 3, 1)),
        )

    def forward(self, z):
        # return the generated image
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        def batch_std(x):
            # minibatch standard deviation to improve variety (Karras et al. 2018)
            std_map = torch.ones_like(x[:,:1]) * torch.std(x, dim=0).mean()
            return torch.cat((x, std_map), dim=1)

        def repeat_block(in_features, out_features):
            return \
                conv_with_padding(in_features, in_features) + [
                nn.LeakyReLU(.2) ] + \
                conv_with_padding(in_features, out_features) + [
                nn.LeakyReLU(.2),
                nn.AvgPool2d(2),
            ]

        # The height and width of downsampled image
        ds_size = img_size // 2**3 # 2**5 # each repeat_block results in a factor-2 downsample
        
        self.model = nn.Sequential(
            # inlet block to take the image as input
            Equalized(nn.Conv2d(3, 64, 1)),

            # repeated blocks
            *repeat_block(64, 128),
            *repeat_block(128, 256),
            # *repeat_block(256, 256),
            # *repeat_block(256, 256),
            *repeat_block(256, 256),

            # outlet block to finalize the output
            # Lambda(batch_std),
            # *conv_with_padding(257, 256),
            *conv_with_padding(256, 256),
            nn.LeakyReLU(.2),
            nn.Flatten(),
            Equalized(nn.Linear(256 * ds_size**2, 256)),
            nn.LeakyReLU(.2),
            Equalized(nn.Linear(256, 1)),
            # nn.Sigmoid(), # not needed in WGAN, where the discrimintator becomes a critic
        )

    def forward(self, img):
        return self.model(img)


