import os
import numpy as np
import numpy.fft as ft
from scipy.interpolate import interp1d

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import basic


class Reader_raw(Dataset):
    def __init__(self, path, img_size):
        self.para = basic.DataSetInfo(path)
        self.img_size = img_size

    def __len__(self):
        return len(self.para.tsteps) * (self.para.Nx//2)

    def __getitem__(self, idx):
        return self.preproc(self.read(*self.indexing(idx)))

    def indexing(self, idx):
        tsteps = self.para.tsteps
        Nx = self.para.Nx
        t = tsteps[idx//(Nx//2)]
        i = idx%(Nx//2)
        return t, i

    def read(self, t, i):
        i += self.para.Nx//4
        vel = self.interp(self.readcut(t, i), self.img_size)
        vel = torch.tensor(vel, dtype=torch.float32)
        return vel

    def preproc(self, vel):
        vel -= vel.mean(dim=-1, keepdim=True)
        vel *= torch.tensor([10., 20., 15.]).view(3, 1, 1)
        vel = self.augment(vel, 2)
        return vel

    def readcut(self, t, i):
        # read a cut of velocity vector (u,v,w) aligned to the i-th cell-center in streamwise direction
        # y and z grids preserved (still staggered)
        Nz = self.para.Nz
        Ny = self.para.Ny

        u = np.empty((Ny+1, Nz-1))
        v = np.empty((Ny+1, Nz-1))
        w = np.empty((Ny+1, Nz-1))

        filename = self.para.fieldpath+'U%08i.bin'%t
        nx, nz, ny = np.fromfile(filename, np.int32, 3)

        for j in range(Ny+1):
            for k in range(Nz-1):
                u[j,k] = np.fromfile(self.para.fieldpath+'U%08i.bin'%t, np.float64, 2, offset=8*(((j+1)*nz+k+1)*nx+i)).mean()
                v[j,k] = np.fromfile(self.para.fieldpath+'V%08i.bin'%t, np.float64, 1, offset=8*(((j+1)*nz+k+1)*nx+i))
                w[j,k] = np.fromfile(self.para.fieldpath+'W%08i.bin'%t, np.float64, 1, offset=8*(((j+1)*nz+k+1)*nx+i))

        return np.array([u,v,w])

    def interp(self, vel, img_size):
        # interpolate/filter the cut (y-z plane) of velocity
        # from the original staggered grid to the collocated image grid

        Nz = self.para.Nz

        # vel[1,0] = vel[1,1]
        # vel[1,1:-1] = .5 * (vel[1,1:-1] + vel[1,2:])
        # vel[2] = .5 * (vel[2] + vel[2,:,range(2-Nz,1)])

        u, v, w, = vel # (Ny+1) * (Nz-1) * ...

        # spanwise filter
        shifter = np.exp(-np.pi*1j * ft.rfftfreq(Nz-1)) # shift staggered w to cell-center
        shifter = shifter.reshape([-1] + [1]*(u.ndim-2))

        u = ft.hfft(ft.ihfft(u, axis=1),           n=img_size, axis=1)
        v = ft.hfft(ft.ihfft(v, axis=1),           n=img_size, axis=1)
        w = ft.hfft(ft.ihfft(w, axis=1) * shifter, n=img_size, axis=1)

        # wall-normal interpolation
        new_ys = self.gengrid(self.para.Ly, img_size)

        u = interp1d(self.para.yc,    u,     axis=0, fill_value='extrapolate')(new_ys)
        v = interp1d(self.para.y[1:], v[1:], axis=0, fill_value='extrapolate')(new_ys)
        w = interp1d(self.para.yc,    w,     axis=0, fill_value='extrapolate')(new_ys)

        return np.array([u,v,w])

    def gengrid(self, ly, ny):
        # cos-distributed wall-normal grids
        return ly * (1 - np.cos(.5*np.pi * np.arange(ny)/(ny-1)))

    def augment(self, vel, flag):
        flip = torch.randint(2, (1,)).item()
        roll = torch.randint(self.img_size, (1,)).item()

        # flip in spanwise direction
        if flag == 1 and flip:
            vel = vel.flip(-1)
            vel[2] *= -1

        # random shift in spanwise direction
        elif flag == 2 and roll:
            vel = self.augment(vel, 1).roll(roll, dims=-1)

        return vel


class Reader(Reader_raw):

    def __init__(self, path1, path2, img_size):
        # path1 is where the raw data of numerical simulations are stored
        # path2 is where the dataset for training (restructured from raw data) are stored
        super().__init__(path1, img_size)
        self.datapath = path2
        os.makedirs(self.datapath, exist_ok=True)

    def read(self, t, i):
        if np.any(['%s%08i.bin'%(c,t) not in os.listdir(self.datapath) for c in 'UVW']):
            self.prepare(t, self.para.fieldpath, self.datapath)

        nx, nz, ny = np.fromfile(self.datapath+'U%08i.bin'%t, np.int32, 3)

        u = np.fromfile(self.datapath+'U%08i.bin'%t, np.float64, nz*nx, offset=8*nz*nx*(i+1)).reshape([nz,nx])
        v = np.fromfile(self.datapath+'V%08i.bin'%t, np.float64, nz*nx, offset=8*nz*nx*(i+1)).reshape([nz,nx])
        w = np.fromfile(self.datapath+'W%08i.bin'%t, np.float64, nz*nx, offset=8*nz*nx*(i+1)).reshape([nz,nx])

        vel = torch.tensor([u,v,w], dtype=torch.float32)

        return vel

    def prepare(self, t, in_path, out_path):
        Nx = self.para.Nx
        ys = self.para.yc

        print('preparing dataset, tstep', t)

        u = basic.read_channel(in_path+'U%08i.bin'%t)[:, 1:-1, Nx//4:Nx//4+Nx//2+1]
        v = basic.read_channel(in_path+'V%08i.bin'%t)[:, 1:-1, Nx//4:Nx//4+Nx//2]
        w = basic.read_channel(in_path+'W%08i.bin'%t)[:, 1:-1, Nx//4:Nx//4+Nx//2]

        u = .5 * (u[:,:,:-1] + u[:,:,1:])

        self.interp(np.array([u,v,w]), self.img_size)

        basic.write_channel(out_path+'U%08i.bin'%t, np.transpose(u, (2,0,1)))
        basic.write_channel(out_path+'V%08i.bin'%t, np.transpose(v, (2,0,1)))
        basic.write_channel(out_path+'W%08i.bin'%t, np.transpose(w, (2,0,1)))

        print('tstep %i added to dataset'%t)


if __name__ == '__main__':

    from config import config_options

    options = config_options()

    reader = Reader('/mnt/disk2/whn/etbl/TBL_1420_big/test/', options.datapath, options.img_size)

    dataloader = DataLoader(
        reader,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.n_cpu,
        prefetch_factor=1,
    )

    for vels in dataloader:
        
        vel = vels[0]

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

        c1 = axs[0,0].contourf(vel[0])
        c2 = axs[0,1].contourf(vel[1])
        c3 = axs[1,0].contourf(vel[2])

        plt.colorbar(c1, ax=axs[0,0])
        plt.colorbar(c2, ax=axs[0,1])
        plt.colorbar(c3, ax=axs[1,0])

        fig.tight_layout()
        fig.savefig('figname.png', dpi=300)
        plt.close()

        exit()





