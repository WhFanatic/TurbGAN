import os
import numpy as np
from scipy.stats import skew, kurtosis
from matplotlib import pyplot as plt
from matplotlib import ticker
import h5py


class Scrutinize:
    def __init__(self, workpath):
        self.workpath = workpath
        os.makedirs(workpath, exist_ok=True)

    def __call__(self, dl_r, dl_f):
        real_imgs = self.get_imgs(dl_r, num=2)#10000)
        fake_imgs = self.get_imgs(dl_f, num=2)#10000)

        self.plot_prof(real_imgs, fake_imgs, self.workpath)
        self.plot_jpdf(real_imgs, fake_imgs, self.workpath)
        self.plot_corr(real_imgs, fake_imgs, self.workpath)

    @staticmethod
    def plot_prof(real_imgs, fake_imgs, workpath):
        mr1, mr2, mr3, mr4, rsr = Scrutinize.get_prof(real_imgs, filename=workpath+'prof_r.h5')
        mf1, mf2, mf3, mf4, rsf = Scrutinize.get_prof(fake_imgs, filename=workpath+'prof_f.h5')
        
        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(6.4,4.8))

        axs[0,0].plot(mr1[0], 'k-')
        axs[0,0].plot(mr1[1], 'k-')
        axs[0,0].plot(mr1[2], 'k-')
        axs[0,1].plot(mr2[0], 'k-')
        axs[0,1].plot(mr2[1], 'k-')
        axs[0,1].plot(mr2[2], 'k-')
        axs[0,1].plot(rsr[0], 'k-')
        axs[1,0].plot(mr3[0], 'k-')
        axs[1,0].plot(mr3[1], 'k-')
        axs[1,0].plot(mr3[2], 'k-')
        axs[1,1].plot(mr4[0], 'k-')
        axs[1,1].plot(mr4[1], 'k-')
        axs[1,1].plot(mr4[2], 'k-')

        axs[0,0].plot(mf1[0], label=r"$U$")
        axs[0,0].plot(mf1[1], label=r"$V$")
        axs[0,0].plot(mf1[2], label=r"$W$")
        axs[0,1].plot(mf2[0], label=r"$<u'u'>$")
        axs[0,1].plot(mf2[1], label=r"$<v'v'>$")
        axs[0,1].plot(mf2[2], label=r"$<w'w'>$")
        axs[0,1].plot(rsf[0], label=r"$<u'v'>$")
        axs[1,0].plot(mf3[0])
        axs[1,0].plot(mf3[1])
        axs[1,0].plot(mf3[2])
        axs[1,1].plot(mf4[0])
        axs[1,1].plot(mf4[1])
        axs[1,1].plot(mf4[2])

        axs[0,0].legend()
        axs[0,1].legend()

        axs[0,0].set_ylabel(r"$U_i$")
        axs[0,1].set_ylabel(r"$<u'_iu'_i>$")
        axs[1,0].set_ylabel(r"$<u'_i^3>/<u'_i^2>^{1.5}$")
        axs[1,1].set_ylabel(r"$<u'_i^4>/<u'_i^2>^2 - 3$")

        axs[0,0].set_title('Mean velocity')
        axs[0,1].set_title('Reynolds stress')
        axs[1,0].set_title('Skewness')
        axs[1,1].set_title('Kurtosis')

        for ax in axs[-1]:
            ax.set_xlabel(r"$\frac{N_y-1}{\pi/2} \arccos(1 - \frac{y}{L_y})$")

        fig.align_labels()
        fig.tight_layout()

        plt.show()
        fig.savefig(workpath+'figname1.png', dpi=300)
        plt.close()

    @staticmethod
    def plot_jpdf(real_imgs, fake_imgs, workpath):
        get_real = lambda j: Scrutinize.get_jpdf(real_imgs, j, filename=workpath+'jpdf_r.h5')
        get_fake = lambda j: Scrutinize.get_jpdf(fake_imgs, j, filename=workpath+'jpdf_f.h5')

        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.4,4.8))

        c1 = axs[0,0].contour(*get_real( 7)[0], colors='black', locator=ticker.LogLocator())
        c2 = axs[0,1].contour(*get_real(32)[0], colors='black', locator=ticker.LogLocator())
        c3 = axs[1,0].contour(*get_real( 7)[1], colors='black', locator=ticker.LogLocator())
        c4 = axs[1,1].contour(*get_real(32)[1], colors='black', locator=ticker.LogLocator())

        axs[0,0].contour(*get_fake( 7)[0], colors='red', levels=c1.levels)
        axs[0,1].contour(*get_fake(32)[0], colors='red', levels=c2.levels)
        axs[1,0].contour(*get_fake( 7)[1], colors='red', levels=c3.levels)
        axs[1,1].contour(*get_fake(32)[1], colors='red', levels=c4.levels)

        axs[0,0].set_ylabel(r"$v'$")
        axs[1,0].set_ylabel(r"$w'$")
        axs[1,0].set_xlabel(r"$u'$")
        axs[1,1].set_xlabel(r"$u'$")
        axs[0,0].set_title("j = %i"%7)
        axs[0,1].set_title("j = %i"%32)

        fig.align_labels()
        fig.tight_layout()

        plt.show()
        fig.savefig(workpath+'figname2.png', dpi=300)
        plt.close()

    @staticmethod
    def plot_corr(real_imgs, fake_imgs, workpath):
        get_real = lambda j: Scrutinize.get_corr(real_imgs, j, filename=workpath+'corr_r.h5')
        get_fake = lambda j: Scrutinize.get_corr(fake_imgs, j, filename=workpath+'corr_f.h5')

        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(7.2,4.8))

        c1 = axs[0,0].contour(get_real( 7)[0], colors='black', linewidths=.8, levels=[-.05,.1,.2,.4,.8])#, locator=ticker.LogLocator())
        c2 = axs[0,1].contour(get_real( 7)[1], colors='black', linewidths=.8, levels=c1.levels)#, locator=ticker.LogLocator())
        c3 = axs[0,2].contour(get_real( 7)[2], colors='black', linewidths=.8, levels=c1.levels)#, locator=ticker.LogLocator())
        c4 = axs[1,0].contour(get_real(32)[0], colors='black', linewidths=.8, levels=c1.levels)#, locator=ticker.LogLocator())
        c5 = axs[1,1].contour(get_real(32)[1], colors='black', linewidths=.8, levels=c1.levels)#, locator=ticker.LogLocator())
        c6 = axs[1,2].contour(get_real(32)[2], colors='black', linewidths=.8, levels=c1.levels)#, locator=ticker.LogLocator())

        axs[0,0].contour(get_fake( 7)[0], colors='red', linewidths=.8, levels=c1.levels)
        axs[0,1].contour(get_fake( 7)[1], colors='red', linewidths=.8, levels=c2.levels)
        axs[0,2].contour(get_fake( 7)[2], colors='red', linewidths=.8, levels=c3.levels)
        axs[1,0].contour(get_fake(32)[0], colors='red', linewidths=.8, levels=c4.levels)
        axs[1,1].contour(get_fake(32)[1], colors='red', linewidths=.8, levels=c5.levels)
        axs[1,2].contour(get_fake(32)[2], colors='red', linewidths=.8, levels=c6.levels)

        for ax in axs[-1]: ax.set_xlabel(r"$N_z (\frac{\Delta z}{L_z} + \frac{1}{2})$")
        axs[0,0].set_ylabel(r"$\frac{N_y-1}{\pi/2} \arccos(1 - \frac{\Delta y + 0.03}{L_y})$")
        axs[1,0].set_ylabel(r"$\frac{N_y-1}{\pi/2} \arccos(1 - \frac{\Delta y + 0.6}{L_y})$")
        axs[0,0].set_title(r"$R_{u'u'}$")
        axs[0,1].set_title(r"$R_{v'v'}$")
        axs[0,2].set_title(r"$R_{w'w'}$")

        fig.align_labels()
        fig.tight_layout()

        plt.show()
        fig.savefig(workpath+'figname3.png', dpi=300)
        plt.close()


    @staticmethod
    def get_imgs(dl, num):
        cnt = 0
        res = []
        while cnt < num:
            for imgs, _ in dl:
                res.append(imgs.cpu().numpy())
                cnt += len(imgs)
                if cnt >= num: break
        return np.concatenate(res)[:num]

    @staticmethod
    def get_prof(imgs, filename=None):
        try:
            with h5py.File(filename, 'r') as f:
                # locals().update({'m'+c+m: f['m'+c+m][:] for m in '1234' for c in 'uvw'})
                mu1, mv1, mw1 = f['mu1'][:], f['mv1'][:], f['mw1'][:]
                mu2, mv2, mw2 = f['mu2'][:], f['mv2'][:], f['mw2'][:]
                mu3, mv3, mw3 = f['mu3'][:], f['mv3'][:], f['mw3'][:]
                mu4, mv4, mw4 = f['mu4'][:], f['mv4'][:], f['mw4'][:]
                ruv, rvw, ruw = f['ruv'][:], f['rvw'][:], f['ruw'][:]
            print('read from ' + filename)
            return (mu1, mv1, mw1), (mu2, mv2, mw2), (mu3, mv3, mw3), (mu4, mv4, mw4), (ruv, rvw, ruw)
        except: print('reading ' + filename + ' failed')

        from scipy.stats import skew, kurtosis
        mu1, mv1, mw1 = np.mean (imgs, axis=(0,-1))
        mu2, mv2, mw2 = np.var  (imgs, axis=(0,-1))
        mu3, mv3, mw3 = skew    (imgs.transpose([1,2,3,0]).reshape(list(imgs.shape[1:-1])+[-1]), axis=-1)
        mu4, mv4, mw4 = kurtosis(imgs.transpose([1,2,3,0]).reshape(list(imgs.shape[1:-1])+[-1]), axis=-1)
        ruv = np.mean(imgs[:,0]*imgs[:,1], axis=(0,-1)) - mu1 * mv1
        rvw = np.mean(imgs[:,1]*imgs[:,2], axis=(0,-1)) - mv1 * mw1
        ruw = np.mean(imgs[:,0]*imgs[:,2], axis=(0,-1)) - mu1 * mw1

        if filename is not None:
            with h5py.File(filename, 'w') as f:
                f.create_dataset('ruv', data=ruv)
                f.create_dataset('rvw', data=rvw)
                f.create_dataset('ruw', data=ruw)
                for m in '1234':
                    for c in 'uvw':
                        f.create_dataset('m'+c+m, data=locals()['m'+c+m])
            print(filename + ' written')

        return (mu1, mv1, mw1), (mu2, mv2, mw2), (mu3, mv3, mw3), (mu4, mv4, mw4), (ruv, rvw, ruw)

    @staticmethod
    def get_jpdf(imgs, j, filename=None, n=100):
        try:
            with h5py.File(filename, 'r') as f:
                us = f['%i/us'%j][:]
                vs = f['%i/vs'%j][:]
                ws = f['%i/ws'%j][:]
                pdfuv = f['%i/pdfuv'%j][:]
                pdfuw = f['%i/pdfuw'%j][:]
            print('read from ' + filename)
            return (us, vs, pdfuv), (us, ws, pdfuw)
        except: print('reading ' + filename + ' failed')

        from rawdata import jpdf

        us, vs, pdfuv = jpdf.calc_jpdf(imgs[:,0,j], imgs[:,1,j])
        us, ws, pdfuw = jpdf.calc_jpdf(imgs[:,0,j], imgs[:,2,j])

        if filename is not None:
            with h5py.File(filename, 'a') as f:
                f.create_dataset('%i/us'%j, data=us)
                f.create_dataset('%i/vs'%j, data=vs)
                f.create_dataset('%i/ws'%j, data=ws)
                f.create_dataset('%i/pdfuv'%j, data=pdfuv)
                f.create_dataset('%i/pdfuw'%j, data=pdfuw)
            print(filename + ' written')

        return (us, vs, pdfuv), (us, ws, pdfuw)

    @staticmethod
    def get_corr(imgs, j, filename=None):
        try:
            with h5py.File(filename, 'r') as f:
                ruu = f['%i/ruu'%j][:]
                rvv = f['%i/rvv'%j][:]
                rww = f['%i/rww'%j][:]
            print('read from ' + filename)
            return ruu, rvv, rww
        except: print('reading ' + filename + ' failed')

        ruu = np.zeros(imgs.shape[-2:])
        rvv = np.zeros(imgs.shape[-2:])
        rww = np.zeros(imgs.shape[-2:])

        def calc(line, plane):
            nz = line.shape[-1]
            cor = np.empty_like(plane)
            for k in range(-(nz//2), nz//2+nz%2):
                if k==0:  cor[:,k+nz//2] = np.mean(line * plane, axis=-1)
                elif k<0: cor[:,k+nz//2] = np.mean(line[-k:] * plane[:,:k], axis=-1)
                else:     cor[:,k+nz//2] = np.mean(line[:-k] * plane[:,k:], axis=-1)
            return cor

        aves = np.mean(imgs, axis=(0,-1), keepdims=True)
        stds = np.std (imgs, axis=(0,-1))

        for u, v, w in imgs - aves:
            ruu += calc(u[j], u) / len(imgs) / (stds[0,j] * stds[0] + 1e-8).reshape([-1,1])
            rvv += calc(v[j], v) / len(imgs) / (stds[1,j] * stds[1] + 1e-8).reshape([-1,1])
            rww += calc(w[j], w) / len(imgs) / (stds[2,j] * stds[2] + 1e-8).reshape([-1,1])

        if filename is not None:
            with h5py.File(filename, 'a') as f:
                f.create_dataset('%i/ruu'%j, data=ruu)
                f.create_dataset('%i/rvv'%j, data=rvv)
                f.create_dataset('%i/rww'%j, data=rww)
            print(filename + ' written')

        return ruu, rvv, rww


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader

    from reader import MyDataset
    from fid import wrapped_dl_gen
    from models import Generator

    gennet = Generator(128, 64).to('cuda')
    gennet.load_state_dict(torch.load('../results/models/model_G.pt'))

    dl_r = DataLoader(
        MyDataset('../dataset_64x64/', img_size=64),
        batch_size=64,
        shuffle=True,
        )

    dl_f = wrapped_dl_gen(gennet, 128, 64)

    Scrutinize('scrut/')(dl_r, dl_f)


