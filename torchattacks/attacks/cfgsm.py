import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attack import Attack


class CFGSM(Attack):
    r"""
    CFGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CFGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8/255):
        super().__init__("CFGSM", model)
        self.eps = eps
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = CoolCrossEntropyLoss()

        images.requires_grad = True
        outputs = self.get_logits(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images

class CoolCrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, weight = None, size_average=None, ignore_index = -100,
                 reduce=None, reduction = 'mean', label_smoothing = 0.0) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, input, target, dim = -1, min=1e-1, max=1):
        values, _ = torch.topk(input, 2, dim = dim)
        differences = values.select(dim, 0) - values.select(dim, 1)
        T = torch.clamp(differences.unsqueeze(dim)/math.log(input.shape[dim] - 1), min = min, max = max)
        return F.cross_entropy(input / T, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)