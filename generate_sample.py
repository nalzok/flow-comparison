import torch
import argparse
import torchvision.utils as t_utils
import torch.nn as nn
import os
from AE.convAE_infoGAN import ConvAE
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import RNVPCouplingBlock, PermuteRandom


def __makedir__(args):
    if not os.path.exists(args.savedir + "/from_dataset"):
        os.makedirs(args.savedir + "/from_dataset")
    if not os.path.exists(args.savedir + "/sample"):
        os.makedirs(args.savedir + "/sample")


def load_check_point(args, subnet_fc, device):
    model_ae = ConvAE(args).to(device)

    nodes = [InputNode(args.num_latent, name="input")]

    for k in range(args.num_block):
        nodes.append(
            Node(
                nodes[-1],
                RNVPCouplingBlock,
                {"subnet_constructor": subnet_fc, "clamp": 2.0},
                name=f"coupling_{k}",
            )
        )
        nodes.append(Node(nodes[-1], PermuteRandom, {"seed": k}, name=f"permute_{k}"))

    nodes.append(OutputNode(nodes[-1], name="output"))

    model_flow = ReversibleGraphNet(nodes, verbose=False)
    model_flow = model_flow.to(device)
    # change the name to the model you want to test
    state_flow = torch.load(
        os.path.join(args.model_dir, "flowModel_epo100"), map_location=device
    )
    state_ae = torch.load(
        os.path.join(args.model_dir, "AEModel_epo100"), map_location=device
    )
    model_ae.load_state_dict(state_ae)
    model_flow.load_state_dict(state_flow)
    return model_ae, model_flow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples to test.")
    parser.add_argument(
        "--dset_dir", default="./data/", type=str, help="dataset directory"
    )
    parser.add_argument("--dataset", default="mnist", type=str, help="dataset name")
    parser.add_argument("--image_size", default=28, type=int, help="image size")
    parser.add_argument(
        "--num_workers", default=8, type=int, help="dataloader num_workers"
    )
    parser.add_argument("--batch_size", default=500, type=int, help="batch size")
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        default="./saved_models/mnist/glf/",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--savedir", default="./samples/mnist", type=str, help="save image address"
    )
    parser.add_argument(
        "--num_latent", default=20, type=int, help="dimension of latent code z"
    )

    parser.add_argument(
        "--fc_dim",
        dest="fc_dim",
        default=64,
        type=int,
        help="Hidden size of inner layers of nonlinearity of flow",
    )
    parser.add_argument(
        "--num_block", dest="nbck", default=4, type=int, help="num of flow blocks"
    )

    parser.add_argument(
        "--device", dest="device", default=0, type=int, help="Index of device"
    )

    parser.add_argument(
        "--loss_type",
        default="MSE",
        type=str,
        help="Type of loss",
        choices=("MSE", "Perceptual", "cross_entropy"),
    )

    args = parser.parse_args()

    __makedir__(args)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.device) if use_gpu else "cpu")

    def subnet_fc(c_in, c_out):
        return nn.Sequential(
            nn.Linear(c_in, args.fc_dim),
            nn.ReLU(),
            nn.BatchNorm1d(args.fc_dim),
            nn.Linear(args.fc_dim, args.fc_dim),
            nn.ReLU(),
            nn.BatchNorm1d(args.fc_dim),
            nn.Linear(args.fc_dim, c_out),
        )

    modAE, modFlow = load_check_point(args, subnet_fc, device)
    modAE.train()
    modFlow.train()

    for i in range(20):
        noise = torch.randn(500, args.num_latent).to(device)
        zs = modFlow(noise, rev=True)
        sample = modAE.decode(zs)
        nc, h, w = sample.shape[1:]
        if nc == 1:
            sample = torch.cat((sample, sample, sample), dim=1)
        for j in range(500):
            sample[j, :, :, :] = (
                sample[j, :, :, :] - torch.min(sample[j, :, :, :])
            ) / (torch.max(sample[j, :, :, :] - torch.min(sample[j, :, :, :])))
        sample = sample * 255
        for j in range(sample.size(0)):
            t_utils.save_image(
                sample.view(500, 3, h, w)[j, :, :, :],
                (args.savedir + "/sample/{}.png").format(j + i * 500),
                normalize=True,
            )
        print(i)
