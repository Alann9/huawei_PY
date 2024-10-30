import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
print(multinomial.Multinomial(10, fair_probs).sample())

