import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from config import config_options
from models import Generator


if __name__ == '__main__':

    options = config_options()

    generator = Generator(options.latent_dim, options.img_size)
    generator.load_state_dict(torch.load(options.workpath+'models/model_G.pt'))
    generator.eval()

    # Generate one image
    with torch.no_grad():
        z = torch.randn((1, options.latent_dim)) # Sample noise as generator input
        img, = generator(z)
    
    plt.imshow(np.squeeze(img.numpy()), cmap='binary')
    plt.show()






    # ## test for WGAN-GP loss

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



## equalized learning rate test
# class EqualLR:
#     ''' Apply equalized learning rate to a nn.Module,
#         proposed by Karras et al. (2018),
#         explained here: https://personal-record.onrender.com/post/equalized-lr/ '''
#     def __init__(self, name):
#         self.name = name

#     def compute_weight(self, module):
#         weight = getattr(module, self.name + '_orig')
#         fan_in = weight.data.size(1) * weight.data[0][0].numel()

#         return weight * (2 / fan_in)**.5

#     @staticmethod
#     def apply(module, name):
#         weight = getattr(module, name)
#         del module._parameters[name]
#         module.register_parameter(name + '_orig', nn.Parameter(weight.data))
#         module.register_forward_pre_hook(EqualLR(name))

#     def __call__(self, module, input):
#         weight = self.compute_weight(module)
#         setattr(module, self.name, weight)

# if __name__ == '__main__':
#     # a = nn.Conv2d(16,16,3)
#     a = nn.Linear(16,16)
#     EqualLR.apply(a, 'weight')
#     for k in a.state_dict():
#         print(k,a.state_dict()[k].shape)





# # learning rate decay test
# import torch
# import torch.nn as nn
# import torch.optim as optim

# model = [nn.Parameter(torch.randn(2, 2, requires_grad=True))]
# optimizer = optim.SGD(model, 0.1)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# optimizer.step()
# scheduler.step()
# optimizer.step()
# scheduler.step()

# torch.save({'opt':optimizer.state_dict()}, '__opt.pt')
# del optimizer

# optimizer = optim.SGD(model, 0.1)

# print(optimizer)

# optimizer.load_state_dict(torch.load('__opt.pt')['opt'])

# print(optimizer)

# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=0)

# print(optimizer)

# optimizer.step()
# scheduler.step()

# print(optimizer)

# exit()
