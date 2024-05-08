import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, random_split
import time
import argparse
from progress.bar import IncrementalBar
from dataset import TU_Graz
#from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.discriminator import ConditionalDiscriminator
from gan.criterion import GeneratorLoss, DiscriminatorLoss
from gan.utils import Logger, initialize_weights
from torchvision.transforms import v2 as T

parser = argparse.ArgumentParser(prog='top', description='Train Pix2Pix')
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="facades",
                    help="Name of the dataset: ['facades', 'maps', 'cityscapes']")
parser.add_argument("--batch_size", type=int, default=8, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

transforms = T.Compose([T.RandomRotation(degrees=45),
                        T.RandomResizedCrop(size=(256, 256), scale=(0.2, 1)),
                        T.RandomHorizontalFlip(p=0.5),
                        T.RandomVerticalFlip(p=0.5),
                        T.ToTensor()])
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

# dataset
dataset = TU_Graz(root='./TU-Graz/train', transform=transforms, mode='train')

PATH_GENERATOR = "runs/standard/generator.pt"
PATH_DISCRIMINATOR = "runs/standard/discriminator.pt"
generator.load_state_dict(torch.load(PATH_GENERATOR, map_location=torch.device(device)))
discriminator.load_state_dict(torch.load(PATH_DISCRIMINATOR, map_location=torch.device(device)))

#elif args.dataset=='maps':
#    dataset = Maps(root='.', transform=transforms, download=True, mode='train')
#else:
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, )
print('Start of training process!')
logger = Logger(filename=args.dataset)
for epoch in range(args.epochs):
    ge_loss = 0.
    de_loss = 0.
    start = time.time()
    bar = IncrementalBar(f'[Epoch {epoch + 1}/{args.epochs}]', max=len(dataloader))
    for real, x in dataloader:
        x = x.to(device)
        real = real.to(device)

        # Generator`s loss
        fake = generator(x)
        fake_pred = discriminator(fake, x)
        g_loss = g_criterion(fake, real, fake_pred)

        # Discriminator`s loss
        fake = generator(x).detach()
        fake_pred = discriminator(fake, x)
        real_pred = discriminator(real, x)
        d_loss = d_criterion(fake_pred, real_pred)

        # Generator`s params update
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Discriminator`s params update
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        # add batch losses
        ge_loss += g_loss.item()
        de_loss += d_loss.item()
        bar.next()
    bar.finish()
    # obtain per epoch losses
    g_loss = ge_loss / len(dataloader)
    d_loss = de_loss / len(dataloader)
    # count timeframe
    end = time.time()
    tm = (end - start)
    logger.add_scalar('generator_loss', g_loss, epoch + 1)
    logger.add_scalar('discriminator_loss', d_loss, epoch + 1)
    logger.save_weights(generator.state_dict(), 'generator')
    logger.save_weights(discriminator.state_dict(), 'discriminator')
    print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA: %.3fs" % (epoch + 1, args.epochs, g_loss, d_loss, tm))
logger.close()
print('End of training process!')
