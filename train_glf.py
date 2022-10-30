import torch
from torch import optim
import torch.nn as nn
import numpy as np
from AE.convAE_infoGAN import ConvAE
import argparse
from tqdm import tqdm
import os

from utils import perceptual_loss
from utils.dataset import return_data
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import RNVPCouplingBlock, PermuteRandom


def distance_metric(sz, force_l2=False):
    if sz == 32 or sz == 28:
        return perceptual_loss._VGGDistance(3,device)
    elif sz == 64:
        return perceptual_loss._VGGDistance(4,device)

def gaussian_nice_loglkhd(h,device):
    return - 0.5*torch.sum(torch.pow(h,2),dim=1) - h.size(1)*0.5*torch.log(torch.tensor(2*np.pi).to(device))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for Training GLF models")

    parser.add_argument("--dataset", default='mnist', dest='dataset', choices=('mnist', 'cifar', 'celeba', 'Fashion-mnist'),
                        help="Dataset to train the GLF model on.")
    parser.add_argument("--dset_dir", dest='dset_dir', default="./data",
                        help="Where you store the dataset.")
    parser.add_argument("--epochs", dest='num_epochs', default=101, type=int,
                        help="Number of epochs to train on.")
    parser.add_argument("--batch_size", dest="batch_size", default=256, type=int,
                        help="Number of examples per batch.")
    parser.add_argument('--device', dest = 'device',default="cpu", type=str,
                        help='Device')
    parser.add_argument("--savedir", dest='savedir', default="./saved_models/mnist/glf/",
                        help="Where to save the trained model.")
    parser.add_argument('--loss_type', default='MSE', type=str, 
                        help='Type of loss',choices = ('MSE','Perceptual','cross_entropy'))
    #Auto Encoder settings:
    parser.add_argument("--num_latent",  default=20, type=int,
                        help="dimension of latent code z")
    parser.add_argument("--image_size",  default=28, type=int,
                        help="size of training image")
    
    #Flow settings:
    parser.add_argument("--fc_dim",  default=64, type=int,
                        help="dimension of FC layer in the flow")
    parser.add_argument("--num_block",  default=4, type=int,
                        help="number of affine coupling layers in the flow")
    
    #optimization settings
    parser.add_argument("--lr", default=1e-3, dest='lr', type=float,
                        help="Learning rate for ADAM optimizer. [0.001]")
    parser.add_argument("--beta1", default=0.8, dest='beta1', type=float,
                        help="beta1 for adam optimizer")
    parser.add_argument("--beta2", default=0.9, dest='beta2', type=float,
                        help="beta2 for adam optimizer")
    parser.add_argument("--decay", default=50, dest='decay', type=float,
                        help="number of epochs to decay the lr by half")
    
    parser.add_argument("--num_workers", dest="num_workers", default=8, type=int,
                        help="Number of workers when load in dataset.")
    args = parser.parse_args()
    
    
    if args.loss_type == 'cross_entropy':
        assert (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'),"Cross entropy should only be used for mnist or Fashion-mnist."
    
    use_gpu = torch.cuda.is_available()

    torch.manual_seed(123)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda:{}".format(args.device) if use_gpu else "cpu")
    
    # get data loader
    training_loader = return_data(args)

    # define reconstruction loss
    if args.loss_type == 'MSE':
        recon_loss_fn = nn.MSELoss(reduction = 'sum')
    elif args.loss_type == 'cross_entropy':
        recon_loss_fn = nn.BCELoss(reduction = 'sum')
    else:
        recon_loss_fn = distance_metric(args.image_size)
    
    
    # define AE and the flow
    modAE = ConvAE(args).to(device)
   
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, args.fc_dim), nn.ReLU(), nn.BatchNorm1d(args.fc_dim),
                         nn.Linear(args.fc_dim, args.fc_dim), nn.ReLU(), nn.BatchNorm1d(args.fc_dim), nn.Linear(args.fc_dim,  c_out))    
    
    nodes = [InputNode(args.num_latent, name='input')]
    for k in range(args.num_block):
        nodes.append(Node(nodes[-1],
                          RNVPCouplingBlock,
                          {'subnet_constructor':subnet_fc, 'clamp':2.0},
                          name=F'coupling_{k}'))
        nodes.append(Node(nodes[-1],
                          PermuteRandom,
                          {'seed':k},
                          name=F'permute_{k}'))
    
    nodes.append(OutputNode(nodes[-1], name='output'))
    
    modFlow = ReversibleGraphNet(nodes, verbose=False)
    modFlow = modFlow.to(device)
    
    
    
    #define optimizers
    optimizer1 = optim.Adam(modAE.parameters(), lr=args.lr)
    trainable_parameters = [p for p in modFlow.parameters() if p.requires_grad]
    optimizer2 = torch.optim.Adam(trainable_parameters, lr=args.lr, betas=(args.beta1, args.beta2),
                             eps=1e-6, weight_decay=2e-5)
    for param in trainable_parameters:
        param.data = 0.05*torch.randn_like(param)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1,args.decay, gamma=0.5, last_epoch=-1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, args.decay, gamma=0.5, last_epoch=-1)
    modAE.train()
    modFlow.train()
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)    
    for epoch in range(args.num_epochs):
        recon_losses = []
        like_losses = []
        log_D = []
        
        with tqdm(total=len(training_loader.dataset)) as progress_bar:
            for batch_idx, (dat, target) in enumerate(training_loader):
                dat = dat.to(device)
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                recon_batch,z= modAE(dat)
                
                #STOP THE GRADIENT OF NLL LOSS AT z
                zhat, logd = modFlow(z.data, jac=True)
                
                if args.loss_type == 'Perceptual' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
                    rbc =  torch.cat((recon_batch,recon_batch,recon_batch),dim=1)
                    datc = torch.cat((dat,dat,dat),dim=1)
                    loss_recon = recon_loss_fn(rbc,datc)
                else:
                    loss_recon = recon_loss_fn(recon_batch, dat)
                
                loss_ll = gaussian_nice_loglkhd(zhat,args.device) + logd
                loss_ll =  -loss_ll.mean()
                
                log_D.append(torch.mean(logd).item())
                like_losses.append(loss_ll.item())
                recon_losses.append(loss_recon.item()/dat.shape[0])
                
                total_loss = loss_recon/dat.size(0) + loss_ll
                total_loss.backward(retain_graph=True)
                
                optimizer1.step()
                optimizer2.step()
                progress_bar.set_postfix(loss= np.mean(recon_losses), logd = np.mean(log_D),
                    likloss = np.mean(like_losses))
                progress_bar.update(dat.size(0))
        
        scheduler1.step()
        scheduler2.step()    
        print('Train Epoch: {} Reconstruction-Loss: {:.4f} loglikelihood loss: {}  log det: {}'.format(
                    epoch, np.mean(recon_losses), np.mean(like_losses),np.mean(log_D)))
        

        #save model every 50 epochs
        if epoch % 50 == 0:
            torch.save(modFlow.state_dict(), os.path.join(args.savedir, 'flowModel_epo{}'.format(epoch)))
            torch.save(modAE.state_dict(), os.path.join(args.savedir, 'AEModel_epo{}'.format(epoch)))

