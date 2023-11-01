import torch
import torch.nn as nn

loss_names = ['l1', 'l2', 'both']

from dataclasses import dataclass
import torch
import torch.optim as optim

@dataclass
class Decay(object):
    """
    Template decay class
    """
    def __init__(self):
        self.mode = "current"

    def step(self):
        raise NotImplementedError

    def get_dr(self):
        raise NotImplementedError
        
class CosineDecay(Decay):

    def __init__(
        self,
        lambdas: float = 0.2,
        T_max: int = 1000,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        super().__init__()
        self._step = 0
        self.T_max = T_max

        self.sgd = optim.SGD(
            torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=lambdas
        )
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.sgd, T_max, eta_min, last_epoch
        )

    def step(self, step: int = -1):
        if step >= 0:
            if self._step < self.T_max:
                self.cosine_stepper.step(step)
                self._step = step + 1
            else:
                self._step = self.T_max
            return
        if self._step < self.T_max:
            self.cosine_stepper.step()
            self._step += 1

    def get_dr(self):
        return self.sgd.param_groups[0]["lr"]
    
    
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss

class MaskedMSELoss_IS(nn.Module):
    def __init__(self):
        super(MaskedMSELoss_IS, self).__init__()

    def forward(self, out, imout, depthout, target, lambdas):
        assert out.dim() == imout.dim() == depthout.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        # Total Loss
        diff = target - out
        diff = diff[valid_mask]
        comb_loss = (diff**2).mean()
        # Image Loss
        im_diff = target - imout
        im_diff = im_diff[valid_mask]
        im_loss = (im_diff**2).mean()
        # Image Loss
        depth_diff = target - depthout
        depth_diff = depth_diff[valid_mask]
        depth_loss = (depth_diff**2).mean()
        
        total_loss = comb_loss + lambdas * im_loss + lambdas * depth_loss
        
        return total_loss, comb_loss.item()

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class MaskedBothLoss(nn.Module):
    def __init__(self):
        super(MaskedBothLoss, self).__init__()
    
    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        
        l2_loss = (diff**2).mean()
        l1_loss = diff.abs().mean()
        
        return 0.5*l2_loss + 0.5*l1_loss