import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image


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

    fig, ax = plt.subplots()

    ax.plot(data[:,0], data[:,1], '.-', lw=.5, markersize=1, label='D loss')
    ax.plot(data[:,0], data[:,2], '.-', lw=.5, markersize=1, label='G loss')

    ax.legend()
    ax.set_xlabel('Iterations')

    fig.tight_layout()

    fig.savefig(figname, dpi=300)
    plt.close()

def draw_fid(figname, filename):

    data = np.atleast_2d(np.loadtxt(filename))

    fig, ax = plt.subplots()

    ax.plot(data[:,0], data[:,1], '.-', lw=.5, markersize=1, label='FID/FID0')

    ax.set_ylabel('Relative FID')
    ax.set_xlabel('Iterations')

    fig.tight_layout()

    fig.savefig(figname, dpi=300)
    plt.close()





