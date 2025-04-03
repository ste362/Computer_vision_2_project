import torch
from torch import nn

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=100, beta=0.2, loss_name=""):
        super().__init__()
        self.alpha=alpha
        self.beta=beta
        self.bce=nn.BCEWithLogitsLoss()
        self.l1=nn.L1Loss()
        self.kld=nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.loss_name=loss_name
        
    def forward(self, fake, real, fake_pred):
        fake_target = torch.ones_like(fake_pred)
        loss = self.bce(fake_pred, fake_target) + self.alpha * self.l1(fake, real)
        if self.loss_name == 'kld_beta':
            real_d = torch.log_softmax(real, dim=1)
            fake_d = torch.log_softmax(fake, dim=1)
            loss += self.beta*self.kld(fake_d, real_d)
        else:
            raise NotImplementedError

        return loss
    

class DiscriminatorLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        fake_loss = self.loss_fn(fake_pred, fake_target)
        real_loss = self.loss_fn(real_pred, real_target)
        loss = (fake_loss + real_loss)/2
        return loss