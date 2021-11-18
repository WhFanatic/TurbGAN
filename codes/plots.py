import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def biSearch(getx, n, target):
    ''' Binary search for the  first element >= target '''
    i, j = 0, n
    while i < j:
        mid = (i + j) >> 1
        if getx(mid) < target: i = mid + 1
        else: j = mid
    return j

def dataClean(data):
    lag = 0
    for i, line in enumerate(data[1:], 1):
        if data[i,0] <= data[i-lag-1,0]:
            lag = i - biSearch(lambda idx: data[idx,0], i-lag-1, data[i,0])
        if lag:
            data[i-lag] = data[i]
    return data[:len(data)-lag]


def draw_vel(figname, vel, ys=None, zs=None):

    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(6.4,7.2))

    if ys is not None and zs is not None:
        c1 = axs[0,1].contourf(zs, ys, vel[3])
        c2 = axs[1,1].contourf(zs, ys, vel[4])
        c3 = axs[2,1].contourf(zs, ys, vel[5])

        axs[0,0].contourf(zs, ys, vel[0], levels=c1.levels)
        axs[1,0].contourf(zs, ys, vel[1], levels=c2.levels)
        axs[2,0].contourf(zs, ys, vel[2], levels=c3.levels)
    else:
        c1 = axs[0,1].contourf(vel[3])
        c2 = axs[1,1].contourf(vel[4])
        c3 = axs[2,1].contourf(vel[5])

        axs[0,0].contourf(vel[0], levels=c1.levels)
        axs[1,0].contourf(vel[1], levels=c2.levels)
        axs[2,0].contourf(vel[2], levels=c3.levels)

    for ax in axs[:,0]: ax.set_ylabel(r'$y$')
    for ax in axs[-1]:  ax.set_xlabel(r'$z$')

    fig.align_labels()
    fig.tight_layout()

    plt.colorbar(c1, ax=axs[0])
    plt.colorbar(c2, ax=axs[1])
    plt.colorbar(c3, ax=axs[2])

    fig.savefig(figname, dpi=300)
    plt.close()

def draw_log(figname, filename, epochs):

    data = np.atleast_2d(np.loadtxt(filename))
    data = dataClean(data)
    if data[0,0] > 0:
        complementary = np.zeros([int(data[0,0]), len(data[0])])
        complementary[:,0] = np.arange(data[0,0])
        data = np.vstack((complementary, data))

    fig, ax = plt.subplots()

    # draw losses at every iteration
    ax1 = ax.twiny()
    ax1.plot(data[:,0], data[:,1], lw=.5, alpha=.5)
    ax1.plot(data[:,0], data[:,2], lw=.5, alpha=.5)

    # draw mean losses of every epoch
    data = data.reshape([epochs, -1, *data.shape[1:]]).mean(axis=1)
    
    ax.plot(np.arange(epochs), data[:,1], '.-', label='D loss')
    ax.plot(np.arange(epochs), data[:,2], '.-', label='G loss')

    ax.set_ylim(np.multiply([-4, 8], np.abs(data[-1,1:3]).max()))

    ax.legend()
    ax.set_xlabel('Epochs')
    ax1.set_xlabel('Iterations')

    fig.tight_layout()

    fig.savefig(figname, dpi=300)
    plt.close()

def draw_fid(figname, filename):

    data = np.atleast_2d(np.loadtxt(filename))
    data = dataClean(data)

    fig, ax = plt.subplots()

    ax.semilogy(data[:,0], data[:,1], '.-', label='FID/FID0')

    ax.set_ylabel('Relative FID')
    ax.set_xlabel('Epochs')

    fig.tight_layout()

    fig.savefig(figname, dpi=300)
    plt.close()





