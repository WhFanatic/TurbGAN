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


def draw_vel(figname, vel, ys, zs):

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(3.2,7.2))

    c1 = axs[0].contourf(zs, ys, vel[0])
    c2 = axs[1].contourf(zs, ys, vel[1])
    c3 = axs[2].contourf(zs, ys, vel[2])

    for ax in axs:
        ax.set_ylabel(r'$y$')
    axs[-1].set_xlabel(r'$z$')

    plt.colorbar(c1, ax=axs[0])
    plt.colorbar(c2, ax=axs[1])
    plt.colorbar(c3, ax=axs[2])

    fig.align_labels()
    fig.tight_layout()

    fig.savefig(figname, dpi=300)
    plt.close()

def draw_log(figname, filename):

    data = np.atleast_2d(np.loadtxt(filename))
    data = dataClean(data)

    fig, ax = plt.subplots()

    ax.plot(data[:,0], data[:,1], '.-', lw=.5, markersize=1, label='D loss')
    ax.plot(data[:,0], data[:,2], '.-', lw=.5, markersize=1, label='G loss')
    ax.plot(data[:,0], data[:,3], '.-', lw=.5, markersize=1, label=r'$\lambda_1 d_1$')
    ax.plot(data[:,0], data[:,4], '.-', lw=.5, markersize=1, label=r'$\lambda_2 d_2$')

    ax.legend()
    ax.set_xlabel('Iterations')

    ax.set_ylim(np.multiply([-5, 10], np.abs(np.median(data[:,1:], axis=0)).max()))

    fig.tight_layout()

    fig.savefig(figname, dpi=300)
    plt.close()

def draw_fid(figname, filename):

    data = np.atleast_2d(np.loadtxt(filename))
    data = dataClean(data)

    fig, ax = plt.subplots()

    ax.semilogy(data[:,0], data[:,1], '.-', lw=.5, markersize=1, label='FID/FID0')

    ax.set_ylabel('Relative FID')
    ax.set_xlabel('Iterations')

    fig.tight_layout()

    fig.savefig(figname, dpi=300)
    plt.close()





