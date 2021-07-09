import torch
import torch.nn as nn


# define custom non-parametric layers
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Discriminator(nn.Module):
    def __init__(self, options):
        super().__init__()

        def conv_with_padding(in_features, out_features):
            return [
                # implement padding by hand: periodic for spanwise, reflect for top, zero for bottom
                nn.ReplicationPad2d((1, 1, 0, 0)),
                nn.ReflectionPad2d((0, 0, 1, 0)),
                nn.ZeroPad2d((0, 0, 0, 1)),
                nn.Conv2d(in_features, out_features, 3),
            ]

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
        ds_size = options.img_size // 2**5 # each repeat_block results in a factor-2 downsample
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            *repeat_block(64, 128),
            *repeat_block(128, 256),
            *repeat_block(256, 256),
            *repeat_block(256, 256),
            *repeat_block(256, 256),
            Lambda(batch_std),
            *conv_with_padding(257, 256),
            # *conv_with_padding(256, 256),
            nn.LeakyReLU(.2),
            nn.Flatten(),
            nn.Linear(256 * ds_size**2, 256),
            nn.LeakyReLU(.2),
            nn.Linear(256, 1),
            # nn.Sigmoid(), # not needed in WGAN, where the discrimintator becomes a critic
        )

    def forward(self, img):
        return self.model(img)



if __name__ == '__main__':
    pass
### test for WGAN-GP loss
# import torch

# class A:
#     def __init__(self):
#         self.b = torch.tensor([1,2,3.], requires_grad=True)
#     def func(self, x):
#         return self.b**2 * x

# a = A()
# x = torch.tensor([3,2,1.])

# x.requires_grad=True
# if x.grad: x.grad.zero_()

# y = a.func(x)

# g, = torch.autograd.grad(y.sum(), x, retain_graph=False, create_graph=True)
# print(x.grad)
# print(g)

# g.sum().backward()

# print(x.grad)
# print(a.b.grad)






