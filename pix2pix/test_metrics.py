import torch
from torch.utils.data import DataLoader
from dataset import TU_Graz
from dataset import transforms as T
from gan.generator import UnetGenerator
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import VisualInformationFidelity
from torchmetrics import UniversalImageQualityIndex
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio


device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
transforms = T.Compose([
    T.Resize(size=(256,256)),
    T.ToTensor()
])

# models
print('Defining models!')
generator = UnetGenerator().to(device)

# load model
PATH_GENERATOR= "runs/standard/generator.pt"
generator.load_state_dict(torch.load(PATH_GENERATOR))

# load test set
batch_size=1
dataset=TU_Graz(root='./TU-Graz/test', transform=transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,)

# set seed
_=torch.manual_seed(123)

# metrics
vif = VisualInformationFidelity()
fid = FrechetInceptionDistance(feature=64)
uqi = UniversalImageQualityIndex()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
psnr = PeakSignalNoiseRatio()



for real,x in dataloader:
    x = x.to(device)
    real = real.to(device)
    output=generator(x)

    real = real.cpu()
    output = output.cpu()

    # metrics update
    fid.update((real*255).to(torch.uint8), real=True)
    fid.update((output*255).to(torch.uint8), real=False)
    vif.update(output, real)
    uqi.update(output, real)
    ssim.update(output, real)
    psnr.update(output, real)


# metrics compute
print("fid",fid.compute())
print("Vif",vif.compute())
print("Uqi",uqi.compute())
print("Ssim",ssim.compute())
print("PSNR",psnr.compute())