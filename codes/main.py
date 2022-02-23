import os
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler

from config import config_options
from models import Generator, Discriminator
from reader import MyDataset
from train  import Trainer


if __name__ == '__main__':

    opt = config_options()

    os.makedirs(opt.workpath+'images/', exist_ok=True)
    os.makedirs(opt.workpath+'models/', exist_ok=True)

    # Initialize generator and discriminator
    gennet = Generator(opt.latent_dim, opt.img_size)
    disnet = Discriminator(opt.img_size, opt.latent_dim)

    # Define Dataset & Dataloader
    dataset = MyDataset(opt.datapath, img_size=opt.img_size)
    dataloader = DataLoader(
        dataset,
        batch_sampler=BatchSampler(
            sampler=WeightedRandomSampler(dataset.label_balancer(), num_samples=len(dataset)),
            batch_size=opt.batch_size,
            drop_last=False,
            ),
        num_workers=opt.n_cpu,
        prefetch_factor=8,
    )

    trainer = Trainer(gennet, disnet, dataloader, opt)

    trainer.train(trainer.train_once)

