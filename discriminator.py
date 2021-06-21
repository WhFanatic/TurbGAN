import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,options):
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            return [
                nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(p=0.25) # randomly zero out each feature map with 25% chance, to promote independence between feature maps
                ] + ([] if not bn else [
                nn.BatchNorm2d(out_filters, 0.8)
                ])

        self.model = nn.Sequential(
            *discriminator_block(options.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = options.img_size // 2**4 # each D-block results in a factor-2 downsample
        
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size**2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
