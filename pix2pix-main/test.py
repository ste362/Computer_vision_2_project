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
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])])
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


dataset=TU_Graz(root='./TU-Graz', transform=transforms, mode='train')
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,)
for x,real in dataloader:
    x = x.to(device)
    real = real.to(device)
    output=generator(x)
    x=x.cpu()
    output=output.cpu()
    #print(invTrans(x)[0].permute(1, 2, 0))
    #plt.imshow(invTrans(x)[0].permute(1, 2, 0))
    #norm1 = colors.LogNorm(output[0].mean() + 0.5 * output[0].std(), output[0].max(), clip='True')

    plt.imshow((x[0][0]+x[0][1]+x[0][2])/3, origin="lower")
    plt.show()
    output=output.detach().numpy()
    plt.imshow((output[0][0]+output[0][1]+output[0][2])/3,  origin="lower")
    #plt.imshow(invTrans(output)[0].permute(1, 2, 0))
    plt.show()

    break