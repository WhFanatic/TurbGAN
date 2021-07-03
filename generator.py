import torch
import torch.nn as nn


# define custom non-parametric layers
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Generator(nn.Module):
    def __init__(self, options):
        super().__init__()

        def pixel_norm(x):
            return x / (torch.mean(x**2, dim=-3, keepdim=True) + 1e-8)**.5

        def conv_with_padding(in_features, out_features):
            return [
                # implement padding by hand: periodic for spanwise, reflect for top, zero for bottom
                nn.ReplicationPad2d((1, 1, 0, 0)),
                nn.ReflectionPad2d((0, 0, 1, 0)),
                nn.ZeroPad2d((0, 0, 0, 1)),
                nn.Conv2d(in_features, out_features, 3),
            ]

        def relu_with_pixnorm():
            return [
                nn.LeakyReLU(.2),
                Lambda(pixel_norm),
            ]

        def repeat_block(in_features, out_features):
            return [
                nn.Upsample(scale_factor=2) ] + \
                conv_with_padding(in_features, out_features) + \
                relu_with_pixnorm() + \
                conv_with_padding(out_features, out_features) + \
                relu_with_pixnorm()

        # the starting size of feature maps at the entrance the conv blocks
        start_size = options.img_size // 2**5 # the start_size turns into img_size through 5 Upsample

        self.model = nn.Sequential(
            nn.Linear(options.latent_dim, 256 * start_size**2),
            nn.Unflatten(1, (256, start_size, start_size)),
            *relu_with_pixnorm(),
            *conv_with_padding(256, 256),
            *repeat_block(256, 256),
            *repeat_block(256, 256),
            *repeat_block(256, 256),
            *repeat_block(256, 128),
            *repeat_block(128, 64),
            nn.Conv2d(64, 3, 1),
        )

    def forward(self, z):
        # return the generated image
        return self.model(z)


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






