import PIL
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, colors, cm
from torch import nn
from torch.utils.data import DataLoader
import time
import argparse
from progress.bar import IncrementalBar
from dataset import TU_Graz
from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminator
from gan.criterion import GeneratorLoss, DiscriminatorLoss
from gan.utils import Logger, initialize_weights


parser = argparse.ArgumentParser(prog = 'top', description='Train Pix2Pix')
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="facades", help="Name of the dataset: ['facades', 'maps', 'cityscapes']")
parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
args = parser.parse_args()

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = T.Compose([T.Resize((256,256)),
                        T.ToTensor(),
                    ])
# models
print('Defining models!')
generator = UnetGenerator().to(device)
discriminator = ConditionalDiscriminator().to(device)
# optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
# loss functions
g_criterion = GeneratorLoss(alpha=100)
d_criterion = DiscriminatorLoss()

PATH_GENERATOR="./runs/generator.pt"
PATH_DISCRIMINATOR="./runs/discriminator.pt"
generator.load_state_dict(torch.load(PATH_GENERATOR))
discriminator.load_state_dict(torch.load(PATH_DISCRIMINATOR))


dataset=TU_Graz(root='./TU-Graz/test', transform=transforms)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,)
for real,x in dataloader:
    x = x.to(device)
    output=generator(x)
    x=x.cpu()
    output=output.cpu()



    x=x.reshape(3,256,256)
    real=real.reshape(3,256,256)
    output=output.reshape(3,256,256)


    fig=plt.figure(figsize=(12,12))
    fig.add_subplot(1,3,1)
    plt.imshow(x.permute(1,2,0).detach().numpy())
    fig.add_subplot(1,3,2)
    plt.imshow(real.permute(1,2,0).detach().numpy())
    fig.add_subplot(1,3,3)
    plt.imshow(output.permute(1,2,0).detach().numpy())
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()
    break