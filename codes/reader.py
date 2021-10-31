import os
import numpy as np
import numpy.fft as ft
from scipy.interpolate import interp1d, UnivariateSpline, RectBivariateSpline
import h5py

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from rawdata import basic, statis


class MyDataset(Dataset):
    def __init__(self, path, raw_paths=None, img_size=192):
        self.cnt = 0 # count of samples (x-cut) altogether
        self.path = path

        self.filenames = sorted([name for name in os.listdir(self.path) if name[:3]+name[-5:] == 'vel.hdf5'])
        self.filenumbs = [int(name[3:11]) for name in self.filenames] # number of samples before (including) this file

        for num, name in zip(self.filenumbs, self.filenames):
            with h5py.File(self.path + name, "r") as f:
                self.cnt += f.attrs['len']
                assert num == self.cnt
                assert f['u'][:].shape[:2] == (img_size, img_size)

        if raw_paths:
            for raw_path in raw_paths:
                self.create_dataset(self.path, raw_path, img_size)
            # check again the newly created dataset
            self.__init__(path, raw_paths=None, img_size=img_size)
            return

        # make sure dataset is functioning correctly
        # and create some class attributes (they have to be created in initiation because this object might not be the one instantiate in dataloader)
        self.__getitem__(0)

    def __len__(self):
        return self.cnt

    def __getitem__(self, idx):
        x, y = self.get_sample(idx)
        x = self.augment(x, 1) # should not shift in Z when periodicity is not kept
        x = torch.tensor(x, dtype=torch.float32)
        return x, y

    def indexing(self, idx):
        ''' transform dataset index to file-relevant indices '''

        def locate(idx):
            n = np.searchsorted(self.filenumbs, idx+1) # file to access
            i = idx - self.filenumbs[n] # position in the file
            return n, i
           
        if not hasattr(self, 'label_sorter'):
            def access(name):
                with h5py.File(self.path + name, "r") as f:
                    return f['Re_theta'][:]
            self.label_sorter = np.argsort( # original indices of sorted labels
                np.concatenate([*map(access, self.filenames)], axis=None))

        return locate(self.label_sorter[idx])

    def get_sample(self, idx):
        ''' get one normalized sample by its index in sorted labels '''
        n, i = self.indexing(idx)
        with h5py.File(self.path + self.filenames[n], "r") as f:
            x = np.array([f[c][:,:,i] for c in 'uvw'])
            y = f['Re_theta'][i]
        return self.normalize(x, y)

    def get_label(self, idx):
        ''' get one normalized label by its index in sorted labels '''
        n, i = self.indexing(idx)
        with h5py.File(self.path + self.filenames[n], "r") as f:
            y = f['Re_theta'][i]
        return self.normalize(y=y)

    def get_labels(self, ids=None):
        ''' get labels in the range ids by the order determined by self.indexing '''
        if ids is None: ids = range(len(self))
        nis = [*map(self.indexing, ids)]
        myhash = {}
        for n, i in nis:
            if n in myhash: continue
            with h5py.File(self.path + self.filenames[n], "r") as f:
                myhash[n] = f['Re_theta'][:]
        return np.array([self.normalize(y=myhash[n][i]) for n, i in nis])

    def get_labelled(self, label, eps=0):
        ''' get the sample nearest to a given (normalized) label,
            randomly choose one if multiple candidates within the tolerance '''

        def biSearch(getx, i0, j0, target, side='left'):
            ''' Binary search for the  first element >= (left) or > (right) target '''
            i, j = i0, j0
            while i < j:
                mid = (i + j) >> 1
                if (side=='left'  and getx(mid) <  target) \
                or (side=='right' and getx(mid) <= target): i = mid + 1
                else: j = mid
            return j

        i = biSearch(self.get_label, 0, len(self), label - eps, side='left')
        j = biSearch(self.get_label, i, len(self), label + eps, side='left')

        if i < j:
            idx = np.random.randint(i, j)
        else:
            idx = sorted([i-1, i%len(self)], key=lambda i: abs(self.get_label(i) - label))[0]

        return idx

    def normalize(self, x=None, y=None):
        ''' normalize sample or label or both to [0~1] '''
        if x is not None:
            if not hasattr(self, 'norm_x'):
                self.norm_x = (np.array([10, 20, 15]).reshape([-1,1,1]), 0)
            x = x * self.norm_x[0] + self.norm_x[1]
        if y is not None:
            if not hasattr(self, 'norm_y'):
                self.norm_y = (1/2000, -0.5)
            y = y * self.norm_y[0] + self.norm_y[1]
        return x if y is None else y if x is None else (x, y)

    def denormalize(self, x=None, y=None):
        if x is not None:
            x = (x - self.norm_x[1]) / self.norm_x[0]
        if y is not None:
            y = (y - self.norm_y[1]) / self.norm_y[0]
        return x if y is None else y if x is None else (x, y)

    def augment(self, vel, flag):
        flip = np.random.randint(2)
        roll = np.random.randint(vel.shape[-1])

        # flip in spanwise direction
        if flag == 1 and flip:
            vel = np.flip(vel, axis=-1)
            vel[2] *= -1

        # random shift and flip in spanwise direction
        elif flag == 2 and roll:
            vel = np.roll(self.augment(vel, 1), roll, axis=-1)

        return np.array(vel)

    def label_balancer(self):
        unique, unique_inverse, unique_counts = np.unique(self.get_labels(), return_inverse=True, return_counts=True)
        return 1. / unique_counts[unique_inverse]

    def create_dataset(self, workpath, datapath, img_size):

        def gengrid(ly, ny):
            return ly * (1 - np.cos(.5*np.pi * np.arange(ny)/(ny-1)))

        para = basic.DataSetInfo(datapath)
        feld = basic.Field(para)
        stas = statis.Statis_x(para)

        stas.calc_umean(para.datapath + '../umean.h5')
        stas.calc_develops()

        Nx = para.Nx
        Ny = para.Ny
        Nz = para.Nz

        # only the middle half of TBL simulation is used
        mask = range(Nx//4, Nx//4 + Nx//2)

        # target coordinates normalized by local delta
        new_ly = 2.
        new_lz = 2.5
        zs_new = new_lz / img_size * np.arange(img_size)
        ys_new = gengrid(new_ly, img_size)

        for tstep in para.tsteps:
            print('Processing step %i...'%tstep)

            u = feld.read('U%08i.bin'%tstep) - np.expand_dims(stas.Um, 1)
            v = feld.read('V%08i.bin'%tstep) - np.expand_dims(stas.Vm, 1)
            w = feld.read('W%08i.bin'%tstep) - np.expand_dims(stas.Wm, 1)

            u = self.interp(u[:,:,mask], stas.dlt[mask], *(para.zc[1:-1], para.yc), *(zs_new, ys_new))
            v = self.interp(v[:,:,mask], stas.dlt[mask], *(para.zc[1:-1], para.yc), *(zs_new, ys_new))
            w = self.interp(w[:,:,mask], stas.dlt[mask], *(para.zc[1:-1], para.yc), *(zs_new, ys_new))

            self.cnt += Nx//2

            # all quantities are normalized by local U_inf and delta
            with h5py.File(workpath + "vel%08i.hdf5"%self.cnt, "w") as f:
                f.create_dataset('u', data=u)
                f.create_dataset('v', data=v)
                f.create_dataset('w', data=w)
                f.create_dataset('um', data=stas.Um[:,mask])
                f.create_dataset('vm', data=stas.Vm[:,mask])
                f.create_dataset('wm', data=stas.Wm[:,mask])

                f.create_dataset('zs', data=zs_new)
                f.create_dataset('ys', data=ys_new)

                f.create_dataset('dlt1', data=(stas.dlt1/stas.dlt)[mask])
                f.create_dataset('dlt2', data=(stas.dlt2/stas.dlt)[mask])
                f.create_dataset('dlt3', data=(stas.dlt3/stas.dlt)[mask])
                f.create_dataset('Re',   data=(stas.dlt/stas.dlt2 * stas.Re_the)[mask])
                f.create_dataset('Re_theta', data=stas.Re_the[mask])
                f.create_dataset('Re_tau',   data=stas.Re_tau[mask])
                f.create_dataset('Cf',       data=stas.Cf[mask])

                f.attrs['len'] = Nx//2

    @staticmethod
    def interp(us, dlts, zs, ys, zs_new, ys_new, periodize=False):
        ''' interpolate a scalar bulk to required grid normalized by BL thickness '''
        assert (len(ys), len(zs), len(dlts)) == us.shape

        def within(arr1, arr2):
            return arr2[0] <= arr1[0] < arr1[-1] <= arr2[-1] # arr1 within arr2

        us_new = np.empty([len(ys_new), len(zs_new), len(dlts)])

        ## simple interpolation, losses periodicity in Z direction
        if not periodize:
            for i, dlt in enumerate(dlts):
                zs_old = (zs - zs[0])/dlt + zs_new[0] # rescale and align the origin
                ys_old = (ys - ys[0])/dlt + ys_new[0]
                us_old = us[:,:,i]

                assert within(ys_new, ys_old), 'Interpolation for y out of range'
                if not within(zs_new, zs_old):
                    print('Interpolation for z out of range (%.2f < %.2f), periodically extended'%(zs_old[-1], zs_new[-1]))
                    zs_old = np.hstack((zs_old, zs_old + zs_old[[1,-1]].sum() - 2*zs_old[0]))
                    us_old = np.hstack((us_old, us_old))

                us_new[:,:,i] = RectBivariateSpline(
                    ys_old, zs_old, us_old)(
                    ys_new, zs_new)

            return us_new

        else:
            print('Periodic z interpolation not developed yet.')
            exit()

        ## interpolation in y-kz space to make the output periodic
        ### 这里插值有问题啊  没考虑 dlt 的变化
        ks     = ft.rfftfreq(len(zs),     .5/np.pi * (zs[1] - zs[0]))
        new_ks = ft.rfftfreq(len(zs_new), .5/np.pi * (zs_new[1] - zs_new[0]))

        us = interp1d(ys, us, axis=0)(ys_new)
        us = ft.ihfft(us, axis=1)
        ampl = np.abs(us)**2 / ks[1] * ks.reshape([-1,1])
        angl = np.angle(us)

        ampl = interp1d(ks, ampl, axis=1)(new_ks)
        angl = interp1d(ks, angl, axis=1)(new_ks)
        ampl = np.maximum(0, ampl / (new_ks.reshape([-1,1]) / new_ks[1] + 1e-12))**.5

        ampl[:,0] = us[:,0].real
        angl[:,0] = 0
        ampl[0] = 0

        us_new = ft.hfft(ampl * np.exp(1j * angl), axis=1)

        return us_new



        # compute pre-multiplied energy spectrum and phase
        us = ft.ihfft(us, axis=-2)

        ampl = np.abs(us)**2 / ks[1] * ks.reshape([-1,1])
        angl = np.angle(us)  / ks[1] * ks.reshape([-1,1])

        # interpolate pre-multiplied spectrum and phase in log-log coordinate
        cut_new = lambda dlt, cut: RectBivariateSpline(
            np.log(ys/dlt+1e-12), np.log(ks*dlt+1e-12), cut)(
            np.log(ys_new+1e-12), np.log(new_ks+1e-12) )

        for i, dlt in enumerate(dlts):
            am = cut_new(dlt, ampl[:,:,i]) * new_ks[1] / (new_ks+1e-12)
            an = cut_new(dlt, angl[:,:,i]) * new_ks[1] / (new_ks+1e-12)
            am[0] = 0
            an[0] = 0

            um = UnivariateSpline(np.log(ys[1:]/dlt), us[1:,0,i].real)(ys_new[1:])
            am[1:,0] = um**2
            an[1:,0] = np.pi * (um < 0)

            us_new[:,:,i] = ft.hfft(am**.5 * np.exp(1j*an))

        return us_new


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
        ys_new = self.gengrid(self.para.Ly, img_size)

        u = interp1d(self.para.yc,    u,     axis=0, fill_value='extrapolate')(ys_new)
        v = interp1d(self.para.y[1:], v[1:], axis=0, fill_value='extrapolate')(ys_new)
        w = interp1d(self.para.yc,    w,     axis=0, fill_value='extrapolate')(ys_new)

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

    datapaths = (
        # '/mnt/disk2/whn/etbl/TBLs/TBL_1000/test/',
        # '/mnt/disk2/whn/etbl/TBLs/TBL_1420/test/',
        # '/mnt/disk2/whn/etbl/TBLs/TBL_2000/test/',
        )

    dataset = MyDataset('../dataset/', datapaths, 64)
    # dataset = MyDataset('../dataset/', datapaths, 192)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=1,
        prefetch_factor=4,
        )

    # sweep through the entire dataset
    for samples in dataloader:
        vels, Re_thes = samples
        vel0, Re_the0 = vels[0], Re_thes[0]
        vel1, Re_the1 = vels[1], Re_thes[1]


    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(6.4,6.4))

    c1 = axs[0,0].contourf(vel0[0])
    c2 = axs[1,0].contourf(vel0[1])
    c3 = axs[2,0].contourf(vel0[2])

    c1 = axs[0,1].contourf(vel1[0])
    c2 = axs[1,1].contourf(vel1[1])
    c3 = axs[2,1].contourf(vel1[2])

    axs[0,0].set_title(r'label = %.2f, $Re_\theta = $%i'%(Re_the0, dataset.denormalize(y=Re_the0)))
    axs[0,1].set_title(r'label = %.2f, $Re_\theta = $%i'%(Re_the1, dataset.denormalize(y=Re_the1)))

    fig.align_labels()
    fig.tight_layout()

    plt.colorbar(c1, ax=axs[0])
    plt.colorbar(c2, ax=axs[1])
    plt.colorbar(c3, ax=axs[2])

    fig.savefig('../figname.png', dpi=300)
    plt.close()






