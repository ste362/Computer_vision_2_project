import PIL
import numpy as np
import torch
from PIL import Image
from ignite.engine import Engine
from ignite.metrics import InceptionScore
from ignite.utils import manual_seed
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
from torchvision.transforms import v2 as T
import ssl
ssl._create_default_https_context = ssl._create_unverified_context






def compute_inception_score(image):
    from collections import OrderedDict

    import torch
    from torch import nn, optim


    # create default evaluator for doctests

    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)

    # create default optimizer for doctests

    param_tensor = torch.zeros([1], requires_grad=True)
    default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

    # create default trainer for doctests
    # as handlers could be attached to the trainer,
    # each test must define his own trainer using `.. testsetup:`

    def get_default_trainer():
        def train_step(engine, batch):
            return batch

        return Engine(train_step)

    # create default model for doctests

    default_model = nn.Sequential(OrderedDict([
        ('base', nn.Linear(4, 2)),
        ('fc', nn.Linear(2, 1))
    ]))

    manual_seed(666)

    metric = InceptionScore()
    metric.attach(default_evaluator, "is")
    y = image.reshape(1, 3, 256, 256)
    print(y)
    print(y.min(), y.max())

    # i want to normalize the image
    y = (y - y.min()) / (y.max() - y.min())

    print(y.min(), y.max())


    state = default_evaluator.run([y])
    return state.metrics["is"]





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



transforms = T.Compose([T.Resize(size=(256, 256)),
                        T.ToTensor()])
# models
print('Defining models!')
generator = UnetGenerator().to(device)
discriminator = ConditionalDiscriminator().to(device)

generator_aug = UnetGenerator().to(device)
discriminator_aug = ConditionalDiscriminator().to(device)
# optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
# loss functions
g_criterion = GeneratorLoss(alpha=100)
d_criterion = DiscriminatorLoss()

PATH_GENERATOR = "runs/standard/generator.pt"
PATH_DISCRIMINATOR = "runs/standard/discriminator.pt"
generator.load_state_dict(torch.load(PATH_GENERATOR, map_location=torch.device(device)))
discriminator.load_state_dict(torch.load(PATH_DISCRIMINATOR, map_location=torch.device(device)))

PATH_GENERATOR_AUG = "runs/augmented/generator_aug.pt"
PATH_DISCRIMINATOR_AUG = "runs/augmented/discriminator_aug.pt"
generator_aug.load_state_dict(torch.load(PATH_GENERATOR_AUG, map_location=torch.device(device)))
discriminator_aug.load_state_dict(torch.load(PATH_DISCRIMINATOR_AUG, map_location=torch.device(device)))

dataset = TU_Graz(root='./TU-Graz/test', transform=transforms)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

g_loss_total = 0
g_loss_aug_total = 0

for real, x in dataloader:
    x = x.to(device)
    real = real.to(device)

    # Generator`s loss
    output = generator(x)
    fake_pred = discriminator(output, x)
    g_loss = g_criterion(output, real, fake_pred)
    g_loss_total += g_loss.item()

    # Generator`s AUG loss
    output_aug = generator_aug(x)
    fake_aug_pred = discriminator_aug(output_aug, x)
    g_loss_aug = g_criterion(output_aug, real, fake_aug_pred)
    g_loss_aug_total += g_loss_aug.item()

    x = x.cpu()
    real = real.cpu()
    output = output.cpu()
    output_aug = output_aug.cpu()

    x = x.reshape(3, 256, 256)
    real = real.reshape(3, 256, 256)
    output = output.reshape(3, 256, 256)
    output_aug = output_aug.reshape(3, 256, 256)
    #output = (output - output.min()) / (output.max() - output.min())



    fig, ax = plt.subplots(1, 4, figsize=(10, 6))
    ax[0].imshow(x.permute(1, 2, 0).detach().numpy())
    ax[1].imshow(real.permute(1, 2, 0).detach().numpy())
    ax[2].imshow(output.permute(1, 2, 0).detach().numpy())
    ax[2].text(20, 300, '{}'.format(f'{g_loss.item():.2f}'))
    ax[2].text(20, 350, '{}'.format(f'{compute_inception_score(output)}'))
    ax[3].imshow(output_aug.permute(1, 2, 0).detach().numpy())
    ax[3].text(20, 300, '{}'.format(f'{g_loss_aug.item():.2f}'))
    ax[3].text(20, 350, '{}'.format(f'{compute_inception_score(output_aug)}'))
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


    # I want to calculate the inception score

    break


#print('Generator loss: {}'.format(g_loss_total / len(dataloader)))
#print('Generator AUG loss: {}'.format(g_loss_aug_total / len(dataloader)))
