import math
import random

import torch
import torch.nn.functional as F
import torchvision

from transformers.utils import ModelOutput

from ..attack import Attack

class Square(Attack):
    r"""
    Square Attack in the paper 'Square Attack: a query-efficient black-box adversarial attack via random search'
    [https://arxiv.org/abs/1912.00049]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        n_queries (int): max number of queries (each restart). (Default: 5000)
        n_restarts (int): number of random restarts. (Default: 1)
        p_init (float): parameter to control size of squares. (Default: 0.8)
        loss (str): loss function optimized ['margin', 'ce'] (Default: 'margin')
        resc_schedule (bool): adapt schedule of p to n_queries (Default: True)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)
        targeted (bool): targeted. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.Square(model, norm='Linf', eps=8/255, n_queries=5000, n_restarts=1, p_init=.8, seed=0, verbose=False, loss='margin', resc_schedule=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, device=None, norm='Linf', eps=8/255, n_queries=5000, n_restarts=1,
                 p_init=.8, loss='margin', resc_schedule=True,
                 seed=0, verbose=False):
        super().__init__('Square', model, device)
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.supported_mode = ['default', 'targeted']

    def func(self, images, labels):
        with torch.no_grad():
            if self.loss == 'ce':
                logits = self.get_logits(images)
                y_corr = logits[torch.arange(len(labels)), labels].clone()
                logits[torch.arange(len(labels)), labels] = -float('inf')
                return y_corr - logits.max(dim=-1).values

            elif self.loss == 'cos':
                feature = self.model(images)
                return F.cosine_similarity(feature, labels) - 0.27

                


            elif self.loss == 'iou':

                boxes = self.model(images)
                boxes_ = unpad_sequence(boxes)
                labels_ = unpad_sequence(labels)
                
                return torch.stack([torchvision.ops.box_iou(b, l).max() for b, l in zip(boxes_, labels_)]) - 0.4


    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = torch.empty_like(images)
        best = torch.full((len(images), ), float('inf'), device = images.device)
        success = torch.zeros_like(best).bool()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        for i in range(self.n_restarts):

            adv_images = torch.where(success[:, None, None, None], adv_images, images).clone()
            best = torch.where(success, best, float('inf'))

            for j in range(self.n_queries):

                x = adv_images[~success].clone()

                b, c, h, w = x.shape

                if self.rescale_schedule:
                    j = int(j / self.n_queries * 10000)
                p = p_scheduler(j, self.p_init)
                s = max(int(round(math.sqrt(p *  h * w))), 1)

                vh = random.randint(0, h - s)
                vw = random.randint(0, w - s)

                x[:, :, vh:vh + s, vw:vw + s] += (torch.randint(0, 2, size=(1, c, 1, 1), device = x.device) * 2 - 1) * self.eps * 2
                x.clamp_(0, 1)
                x.clamp_(images[~success] - self.eps, images[~success] + self.eps)

                scores = self.func(x, labels[~success])

                change = scores < best[~success]

                x = torch.where(change[:, None, None, None], x, adv_images[~success])
                scores = torch.where(change, scores, best[~success])

                best.masked_scatter_(~success, scores)
                adv_images.masked_scatter_(~success[:, None, None, None], x)

                success = best < 0

        if self.verbose:
            return ModelOutput(adv_images = adv_images, success = success, best = best)


        return adv_images

def unpad_sequence(padded_sequences, padding_token = -1):

    padding_mask = (padded_sequences[:, :, 0] == padding_token)
    lengths = padded_sequences.size(1) - padding_mask.sum(dim=1)
    return [padded_sequences[i, :lengths[i]] for i in range(padded_sequences.size(0))]


def p_scheduler(i, p=0.8):
    if i > 8000:
        p /= 512
    elif i > 6000:
        p /= 256
    elif i > 4000:
        p /= 128
    elif i > 2000:
        p /= 64
    elif i > 1000:
        p /= 32
    elif i > 500:
        p /= 16
    elif i > 200:
        p /= 8
    elif i > 50:
        p /= 4
    elif i > 10:
        p /= 2

    return p
