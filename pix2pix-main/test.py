import torch
from matplotlib import pyplot as plt, colors, cm
from torch.utils.data import DataLoader
import argparse

from dataset import TU_Graz
from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminator
from gan.criterion import GeneratorLoss, DiscriminatorLoss

parser = argparse.ArgumentParser(prog='top', description='Train Pix2Pix')
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="facades",
                    help="Name of the dataset: ['facades', 'maps', 'cityscapes']")
parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(device)
transforms = T.Compose([
    T.Resize(size=(256, 256)),
    T.ToTensor()
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

PATH_GENERATOR = "./runs/augmented/generator.pt"
PATH_DISCRIMINATOR = "./runs/augmented/discriminator.pt"
generator.load_state_dict(torch.load(PATH_GENERATOR, map_location=torch.device(device)))
discriminator.load_state_dict(torch.load(PATH_DISCRIMINATOR, map_location=torch.device(device)))

dataset = TU_Graz(root='./TU-Graz/test', transform=transforms)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, )

_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance

fid = FrechetInceptionDistance(feature=64)

show = False
for real, x in dataloader:
    x = x.to(device)
    real = real.to(device)

    output = generator(x)

    # Generator`s loss
    fake = generator(x)
    fake_pred = discriminator(output, x)
    g_loss = g_criterion(output, real, fake_pred)

    real = real.cpu()
    output = output.cpu()
    fid.update((real * 255).to(torch.uint8), real=True)
    fid.update((output * 255).to(torch.uint8), real=False)

    if show:
        x = x.cpu()
        output = output.cpu()
        x = x.reshape(3, 256, 256)
        real = real.reshape(3, 256, 256)
        output = output.reshape(3, 256, 256)

        outmap_min, _ = torch.min(output, dim=1, keepdim=True)
        outmap_max, _ = torch.max(output, dim=1, keepdim=True)
        #output = (output - outmap_min) / (outmap_max - outmap_min) # Broadcasting rules apply

        #print(torch.min(output),torch.max(output))
        #o_n=output.detach().numpy()

        f = torch.nn.ReLU()
        output = f(output)

        fig = plt.figure(figsize=(12, 12))
        fig.add_subplot(1, 3, 1)
        plt.imshow(x.permute(1, 2, 0).detach().numpy())
        fig.add_subplot(1, 3, 2)
        plt.imshow(real.permute(1, 2, 0).detach().numpy())
        fig.add_subplot(1, 3, 3)
        plt.imshow(output.permute(1, 2, 0).detach().numpy())
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.show()
        break
total_distance = fid.compute()
print(total_distance)
