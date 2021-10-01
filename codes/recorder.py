import numpy as np
import torch


def save_for_resume(path, epoch, gennet, disnet, optimizer_G, optimizer_D, loss_G, loss_D):
    def swap_dg(m):
        for p in m.parameters():
            p.data, p.grad = p.grad, p.data

    dic = {
        'epoch': epoch,
        'G_state_dict': gennet.state_dict(),
        'D_state_dict': disnet.state_dict(),
        'G_optimizer_state_dict': optimizer_G.state_dict(),
        'D_optimizer_state_dict': optimizer_D.state_dict(),
        'G_loss': loss_G,
        'D_loss': loss_D,
        }

    for m in (gennet, disnet): swap_dg(m)

    dic['G_grad_dict'] = gennet.state_dict()
    dic['D_grad_dict'] = disnet.state_dict()
    
    for m in (gennet, disnet): swap_dg(m)

    torch.save(dic, path + 'for_resume_ep%03i.pt'%epoch)

    print('Resume file saved for epoch %i.'%epoch)

def load_for_resume(path, epoch, gennet, disnet, optimizer_G, optimizer_D):
    for_resume = torch.load(path + 'for_resume_ep%03i.pt'%epoch, map_location='cpu') # load to cpu in case GPU is full

    assert epoch == for_resume['epoch'], "\nResume file error !\n"

    gennet.load_state_dict(for_resume['G_state_dict'])
    disnet.load_state_dict(for_resume['D_state_dict'])
    optimizer_G.load_state_dict(for_resume['G_optimizer_state_dict'])
    optimizer_D.load_state_dict(for_resume['D_optimizer_state_dict'])

    print('\nResume file loaded from epoch %i.\n'%epoch)

def save_current(path, gennet, disnet):
    torch.save(gennet.state_dict(), path + 'model_G.pt')
    torch.save(disnet.state_dict(), path + 'model_D.pt')





