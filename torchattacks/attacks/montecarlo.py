import torch
import torch.nn as nn
import torchvision.transforms as T

from ..attack import Attack
from scipy.stats import binomtest
from transformers.utils import ModelOutput

class BruteForceUniform(Attack):
    r"""
    BruteForceUniform in the paper 'xx'
    [https://arxiv.org/abs/xx]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.BruteForceUniform(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8/255, alpha = 1e-2, mu = 5e-2, pop=128, verbose=False):
        super().__init__("BruteForceUniform", model)
        self.eps = eps
        self.alpha = alpha
        self.pop = pop
        self.mu = mu
        self.verbose = verbose
        self.supported_mode = ['default', 'targeted']

    def sample(self, x):
        xs = x.repeat(self.pop // len(x), 1, 1, 1, 1)
        ub = torch.clamp(x + self.eps, min = 0, max = 1)
        lb = torch.clamp(x - self.eps, min = 0, max = 1)
        xs = (ub - lb) * torch.rand_like(xs) + lb
        xs = (xs * 255).int() / 255.
        return xs

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        func = lambda k, n: bool(binomtest(k, n, p=self.mu, alternative='two-sided').pvalue < 2 * self.alpha)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = torch.empty_like(images)

        rejected = torch.zeros_like(labels).bool()

        K = None

        with torch.no_grad():

            while not rejected.all():
                
                x = images[~rejected]
                xs = self.sample(x)                

                outputs = self.get_logits(xs.view(-1, *xs.shape[2:])).view(xs.size(0), xs.size(1), -1)
                preds = outputs.argmax(-1)
                # preds = outputs.argmax(-1) == labels[~rejected]
                # adv_images[~rejected] = torch.gather(xs, 0, preds.int().argmin(0).view(-1, 1, 1, 1).expand(1, *xs.shape[1:])).squeeze(0)

                if K is None:
                    K = preds
                else:
                    K.masked_scatter_(~rejected.unsqueeze(1).expand_as(K), K[~rejected.unsqueeze(1).expand_as(K)] + preds.flatten())
                
                en = K.int().sum(1)
                ex = en - K.int().max(1)[0]

                rejected = torch.tensor(list(map(func, ex.tolist(), en.tolist())), device = rejected.device)

                if self.verbose:
                    print(f'rejected: {round(rejected.float().mean().item(), 5)}; K<N: {round((K<N).float().mean().item(), 5)}', K.sum().item())

        return ModelOutput(adv_images = None, K = K, N = N, certified = (1 - K/N) < self.mu, mu = self.mu, alpha = self.alpha)

class BruteForceRandomRotation(BruteForceUniform):

    def sample(self, x):
        num = self.pop // len(x)
        return torch.stack(list(map(
            # T.RandomRotation(degrees=self.eps),
            T.RandomAffine(degrees=self.eps, translate=[0.3, 0.3]),
            [x] * num
        )))

    # def sample(self, x):
    #     num = self.pop // len(x)
    #     device = x.device
    #     x = x.cpu()
    #     return torch.stack(list(map(
    #         T.RandomRotation(degrees=self.eps),
    #         x.repeat(num, 1, 1, 1)
    #     ))).view(num, *x.shape).to(device)

class BruteForceRandomTranslation(BruteForceUniform):

    def sample(self, x):
        num = self.pop // len(x)
        return torch.stack(list(map(
            T.RandomAffine(degrees=0, translate=self.eps),
            [x] * num
        )))

class BruteForceRandomAffine(BruteForceUniform):

    def sample(self, x):
        num = self.pop // len(x)
        return torch.stack(list(map(
            T.RandomAffine(degrees=self.eps[2], translate=self.eps[0:2]),
            [x] * num
        )))

class BruteForceRandomScale(BruteForceUniform):

    def sample(self, x):
        num = self.pop // len(x)
        return torch.stack(list(map(
            T.RandomResizedCrop(size=x.shape[-1], scale=self.eps, ratio=(1, 1)),# antialias = True),
            [x] * num
        )))

from math import ceil

class Proci(BruteForceUniform):

    def __init__(self, model, eps=8/255, alpha = 1e-2, mu = 0.95, pop=128, verbose=False, neighbour = 10, batch_size = 2048):
        super().__init__(model, eps, alpha, mu, pop, verbose)
        self.attack = "Proci"
        self.neighbour = neighbour
        self.batch_size = batch_size

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        inputs_ = inputs.repeat(self.neighbour, 1, 1, 1, 1)
        ub = torch.clamp(inputs_ + self.eps, min = 0, max = 1)
        lb = torch.clamp(inputs_ - self.eps, min = 0, max = 1)
        inputs_ = (ub - lb) * torch.rand_like(inputs_) + lb
        
        logits = []
        for k in range(ceil(self.neighbour * len(inputs) / self.batch_size)):
            logits.append(self.model(inputs_.view(-1, *inputs.shape[1:])[k * self.batch_size: (k + 1) * self.batch_size]))
        logits = torch.cat(logits, dim = 0).view(self.neighbour, len(inputs), -1)
        logits = logits.max(-1, keepdim = True)[0] == logits
        return logits.float().mean(0)