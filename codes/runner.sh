python main.py \
--datapath=../dataset_64x64/ \
--workpath=../results/ \
--n_gpu=3 \
--check_every=10 \
--latent_dim=128 \
--img_size=64 \
--batch_size=85 \
--lr=1e-4 \
--lr_decay=const \
--beta1=.5 \
--lambda_d1=0 \
--lambda_d2=0 \
--resume=78 \
