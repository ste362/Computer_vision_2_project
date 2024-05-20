import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset import TU_Graz
from gan.generator import UnetGenerator
from torchvision.transforms import v2 as T

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
generator_std = UnetGenerator().to(device)
generator_kld = UnetGenerator().to(device)
generator_aug = UnetGenerator().to(device)
generator_aug_kld = UnetGenerator().to(device)

# load model
PATH_GENERATOR = "runs/standard/generator.pt"
generator_std.load_state_dict(torch.load(PATH_GENERATOR, map_location=torch.device(device)))

PATH_GENERATOR_KLD = "runs/only_kld0.2/generator.pt"
generator_kld.load_state_dict(torch.load(PATH_GENERATOR_KLD, map_location=torch.device(device)))

PATH_GENERATOR_AUG = "runs/aug_v2/generator.pt"
generator_aug.load_state_dict(torch.load(PATH_GENERATOR_AUG, map_location=torch.device(device)))

PATH_GENERATOR_AUG_KLD = "runs/aug+kld0.2/generator.pt"
generator_aug_kld.load_state_dict(torch.load(PATH_GENERATOR_AUG_KLD, map_location=torch.device(device)))

batch_size = 1
dataset = TU_Graz(root='./TU-Graz/test', transform=transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

i = 0
for real, x in dataloader:
    x = x.to(device)
    real = real.to(device)

    # Compute output
    output_std = generator_std(x)
    output_kld = generator_kld(x)
    output_aug = generator_aug(x)
    output_aug_kld = generator_aug_kld(x)

    # move tensor to cpu
    x = x.cpu()
    real = real.cpu()
    output_std = output_std.cpu()
    output_kld = output_kld.cpu()
    output_aug = output_aug.cpu()
    output_aug_kld = output_aug_kld.cpu()

    # reshape
    x = x.reshape(3, 256, 256)
    real = real.reshape(3, 256, 256)
    output_std = output_std.reshape(3, 256, 256)
    output_kld = output_kld.reshape(3, 256, 256)
    output_aug = output_aug.reshape(3, 256, 256)
    output_aug_kld = output_aug_kld.reshape(3, 256, 256)

    fig, ax = plt.subplots(1, 6, figsize=(20, 7))
    ax[0].imshow(x.permute(1, 2, 0).detach().numpy())
    ax[1].imshow(real.permute(1, 2, 0).detach().numpy())
    ax[2].imshow(output_std.permute(1, 2, 0).detach().numpy())
    ax[3].imshow(output_kld.permute(1, 2, 0).detach().numpy())
    ax[4].imshow(output_aug.permute(1, 2, 0).detach().numpy())
    ax[5].imshow(output_aug_kld.permute(1, 2, 0).detach().numpy())
    ax[0].text(50, 300, 'Input Image',fontsize=15)
    ax[1].text(50, 300, 'Ground truth',fontsize=15)
    ax[2].text(50, 300, 'Base model',fontsize=15)
    ax[3].text(50, 300, 'Kld model',fontsize=15)
    ax[4].text(65, 300, 'Aug model',fontsize=15)
    ax[5].text(30, 300, 'Aug+kld model',fontsize=15)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()

    i += 1

