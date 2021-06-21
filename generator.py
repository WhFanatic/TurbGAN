import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,options):
        super().__init__()

        self.init_size = options.img_size // 4 # the init_size turns into img_size going through 2 Upsample

        self.l1 = nn.Sequential(
            nn.Linear(options.latent_dim, 128 * self.init_size**2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), # 128 feature maps
            nn.Upsample(scale_factor=2), # nearest interpolation (repeat every row and col)
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, 0.8), # eps = 0.8 ???
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, options.channels, 3, padding=1),
            nn.Tanh(), # output is img_size
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(len(out), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


if __name__ == '__main__':

    from config import config_options

    options = config_options()
    generator = Generator(options)
    generator.load_state_dict(torch.load(options.workpath+'models/model_G.pt'))
    generator.eval()

    # Generate one image
    with torch.no_grad():
        z = torch.randn((1, options.latent_dim)) # Sample noise as generator input
        img, = generator(z)


    import numpy as np
    import matplotlib.pyplot as plt

    plt.imshow(np.squeeze(img.numpy()), cmap='binary')
    plt.show()






