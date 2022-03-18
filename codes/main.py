import os
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from config import config_options
from models import Generator, Discriminator
from reader import MyDataset
from train  import Trainer


def wrap_dataset(dataset, sampler=None):
    # wrap a sampler into dataset so that it can be distributed
    class SamplerWrappedDataset(torch.utils.data.Dataset):
        # wrap a sampler with a dataset
        def __init__(self, dataset, sampler):
            self.dataset = dataset
            self.sampler = sampler
            self.it = iter(self.sampler)

        def __len__(self):
            return len(self.sampler)

        def __getitem__(self, _):
            try: return self.dataset[next(self.it)]
            except StopIteration: self.it = iter(self.sampler)
            return self.dataset[next(self.it)]

    return dataset if sampler is None else SamplerWrappedDataset(dataset, sampler)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL' # for debug

    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # set default GPU corresponding to the current process
    if rank < torch.cuda.device_count():
        print('set GPU %i for process %i'%(rank, rank))
        torch.cuda.set_device(rank)

    torch.distributed.barrier()

def cleanup():
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

def runner(rank, opt):
    setup(rank, opt.n_cpu)

    # set default gpu for the current process
    dev = 'cuda' if opt.n_gpu else 'cpu'

    # Initialize generator and discriminator
    gennet = Generator    (opt.latent_dim, opt.img_size, opt.img_chan).to(dev)
    disnet = Discriminator(opt.img_size, opt.latent_dim, opt.img_chan).to(dev)

    # Define Dataset & Dataloader
    dataset = MyDataset(
        opt.datapath,
        img_size=opt.img_size,
        device=dev,
        )
    dataloader = DataLoader(
        wrap_dataset(
            dataset,
            sampler=WeightedRandomSampler(dataset.label_balancer(), num_samples=opt.batch_size)
            ),
        batch_sampler=BatchSampler(
            sampler=torch.utils.data.distributed.DistributedSampler(dataset),
            batch_size=opt.batch_size, # batch size on each GPU, gradients are averaged among GPUs before backward prop
            drop_last=False,
            ),
        # pin_memory=True,
    )

    trainer = Trainer(
        DDP(gennet),
        DDP(disnet),
        dataloader,
        opt,
        )
    trainer.train(trainer.train_unsuper)
        
    cleanup()



if __name__ == '__main__':

    opt = config_options()
    os.makedirs(opt.workpath+'images/', exist_ok=True)
    os.makedirs(opt.workpath+'models/', exist_ok=True)

    assert opt.n_gpu <= torch.cuda.device_count(), 'Max %i GPUs available!'%torch.cuda.device_count()

    torch.multiprocessing.spawn(runner,
        args=(opt,),
        nprocs=opt.n_cpu,
        join=True)





