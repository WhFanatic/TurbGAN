# Modified from: https://github.com/hukkelas/pytorch-frechet-inception-distance/blob/master/fid.py
import numpy as np
from scipy import linalg
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as Ftv
from torchvision.models import inception_v3


class PartialInceptionNetwork(nn.Module):

    def __init__(self, model_file=None):
        super().__init__()

        # download inception_v3 model or read it from local file
        if model_file is None:
            self.inception_network = inception_v3(pretrained=True)
        else:
            try:
                self.inception_network = torch.load(model_file)
            except:
                self.inception_network = inception_v3(pretrained=True)
                torch.save(self.inception_network, model_file)

        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)

    def output_hook(self, module, input, output):
        self.mixed_7c_output = output # N x 2048 x 8 x 8

    def forward(self, x):
        # Args: x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        # Returns: inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        
        assert -1 <= x.min() and x.max() <= 1, "Expect input value to be in range -1 ~ 1, but got %.2f ~ %.2f"%(x.min(), x.max())
        assert x.dtype is torch.float32,       "Except input type to be torch.float32, but got {}".format(x.dtype)
        assert x.shape[1:] == (3, 299, 299),   "Expect input shape to be (N,3,299,299), but got {}".format(x.shape)

        # Trigger output hook
        self.inception_network(x)

        assert self.mixed_7c_output.shape == (len(x),2048,8,8), "Expext output shape to be (N,2048,8,8), but got {}".format(self.mixed_7c_output.shape)        

        acts = F.adaptive_avg_pool2d(self.mixed_7c_output, (1,1)).view(len(x), 2048)
    
        return acts


class FID:
    def __init__(self, dl_r, dl_f, incep_net):
        self.dl_r = dl_r # dataloader for real images
        self.dl_f = dl_f # dataloader for fake images

        # incep_net can be a pre-trained model object, or the file name of the model to be loaded, or None for downloading the inception_v3 model online
        self.incep_net = PartialInceptionNetwork(incep_net) if (isinstance(incep_net, str) or incep_net is None) else incep_net

    def calc(self, num=1000, relative=False):
        ## calculate the FID: fid(P_r, P_f); relative: fid(P_r, P_f) / fid(P_r, P_r)

        mu1, sigma1 = self.calc_acts_statis(self.dl_r, self.incep_net, num)
        mu2, sigma2 = self.calc_acts_statis(self.dl_f, self.incep_net, num)

        fid = self.calc_frechet_dist(mu1, sigma1, mu2, sigma2)

        if not relative: return fid

        mu2, sigma2 = self.calc_acts_statis(self.dl_r, self.incep_net, num)

        fid0 = self.calc_frechet_dist(mu1, sigma1, mu2, sigma2)

        return fid/fid0, fid, fid0

        # if relative:
        #     mu1, sigma1 = self.calc_acts_statis(self.dl_r, self.incep_net, num)
        #     mu2, sigma2 = self.calc_acts_statis(self.dl_r, self.incep_net, num)
        #     mu3, sigma3 = self.calc_acts_statis(self.dl_f, self.incep_net, num)
        #     return self.calc_frechet_dist(mu1, sigma1, mu3, sigma3) \
        #          / self.calc_frechet_dist(mu1, sigma1, mu2, sigma2)

        # return self.calc_frechet_dist(
        #     *self.calc_acts_statis(self.dl_r, self.incep_net, num),
        #     *self.calc_acts_statis(self.dl_f, self.incep_net, num) )

    @staticmethod
    def calc_frechet_dist(mu1, sigma1, mu2, sigma2, eps=1e-6):
        # The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2)
        # d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
        # mu: The sample mean over activations of the pool_3 layer, precalcualted on an representive data set.
        # sigma: The covariance matrix over activations of the pool_3 layer, precalcualted on an representive data set.

        mu1, sigma1 = np.atleast_1d(mu1), np.atleast_2d(sigma1)
        mu2, sigma2 = np.atleast_1d(mu2), np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        def check_singular(cov):
            # product might be almost singular
            if not np.isfinite(cov).all():
                msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
                warnings.warn(msg)
                offset = np.eye(len(sigma1)) * eps
                cov = linalg.sqrtm((sigma1+offset) @ (sigma2+offset))
            return cov

        def check_complex(cov):
            # numerical error might give slight imaginary component
            if np.iscomplexobj(cov):
                if not np.allclose(np.diagonal(cov).imag, 0, atol=1e-3):
                    m = np.max(np.abs(cov.imag))
                    raise ValueError("Imaginary component {}".format(m))
                cov = cov.real
            return cov

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1@sigma2, disp=False)
        covmean = check_singular(covmean)
        covmean = check_complex(covmean)

        return diff@diff + np.trace(sigma1) + np.trace(sigma2) - 2*np.trace(covmean)

    @staticmethod
    def calc_acts_statis(dataloader, incep_net, num):
        ## compute statistics of activations over in total num images returned by dataloader batch by batch

        incep_acts = np.empty([num, 2048], dtype=np.float32)

        incep_net.eval()

        cnt = 0

        while cnt < num:
            for imgs, _ in dataloader:

                with torch.no_grad():
                    acts = incep_net.to(imgs.device)(FID.preproc_imgs(imgs))

                incep_acts[cnt:cnt+len(imgs)] = acts[:num-cnt].cpu().numpy()

                cnt += len(imgs)
                if cnt >= num: break

        return np.mean(incep_acts, axis=0), np.cov(incep_acts, rowvar=False)

    @staticmethod
    def preproc_imgs(imgs, scale=10):
        ## pre-process the images to make sure they have desired size and within the value range [-1,1]
        imgs = Ftv.resize(imgs, (299,299))
        imgs /= scale * torch.mean((imgs**2).sum(dim=-3)**.5)

        # # check if scale is appropriate
        # print((torch.sum(imgs<-1) + torch.sum(1<imgs)))
        # print(torch.prod(torch.tensor(imgs.shape)))
        # exit()

        imgs = torch.clamp(imgs, -1, 1)
        return imgs


def wrapped_dl_gen(gennet, latent_dim, batch_size=1):

    class DataSetGen(Dataset):
        def __init__(self, g, n):
            self.n = n
            self.g = g # just matain a reference of the generator net, do not change its status

        def __getitem__(self, i):
            return self.g.getone(self.n), 0

        def __len__(self):
            return 99999999

    dl = DataLoader(DataSetGen(gennet, latent_dim), batch_size=batch_size)

    return dl


if __name__ == "__main__":
    import os

    from config import config_options
    from reader import Reader

    options = config_options()
    os.makedirs(options.workpath+'models/', exist_ok=True)

    reader = Reader('/mnt/disk2/whn/etbl/TBL_1420_big/test/', options.datapath, options.img_size)

    dl_r = DataLoader(
        reader,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.n_cpu,
        prefetch_factor=1,
    )

    dl_f = dl_r # dataloader of the fake images should actually involve a generator network

    fid = FID(dl_r, dl_f, options.workpath+'models/inception_v3.pt')
    print('FID between identical distributions:', fid.calc(options.n_critic))


    from generator import Generator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gennet = Generator(options.latent_dim, options.img_size)
    gennet.load_state_dict(torch.load(options.workpath+'models/model_G.pt'))
    gennet.to(device)
    gennet.eval()
    dl_f = wrapped_dl_gen(gennet, options.latent_dim, options.batch_size)

    fid = FID(dl_r, dl_f, options.workpath+'models/inception_v3.pt')
    print('FID between generator and dataset:', fid.calc(options.n_critic))





