import torch

class ELR_reg(torch.nn.Module):
    """
    This code implements ELR regularization which is partially adapted from 
    https://github.com/shengliu66/ELR/blob/master/ELR/model/loss.py
    """
    def __init__(self, num, nb_classes, beta=0.1, lamb=3.0):
        # beta = {0.1,0.3,0.5,0.7,0.9,0.99}
        # lam = {1,3,5,7,9}
        super(ELR_reg, self).__init__()
        self.ema = torch.zeros(num, nb_classes).cuda()
        self.beta = beta
        self.lamb = lamb

    def forward(self, index, outputs, targets):
        y_pred = torch.nn.functional.softmax(outputs, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.ema[index] = self.beta * self.ema[index] + (1 - self.beta) * ((y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, ignore_index=-1)
        elr_reg = ((1 - (self.ema[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = self.lamb * elr_reg + ce_loss
        print(f' elr loss is {elr_reg.item()}')
        return final_loss