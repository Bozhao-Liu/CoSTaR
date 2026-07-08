import os
from typing import Union
import torch
import torch.nn as nn
import logging
from sklearn import metrics
import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

epslon = 1e-8

class ECE_loss(nn.Module):
	def __init__(self, weights: Union[tuple, int] = (1, 1)):
		super(ECE_loss, self).__init__()
		if type(weights) is int:
			self.weight = [weights, weights]
		else:
			self.weight = [weights[0], weights[1]]

	def forward(self, pred, label):
		pred = torch.clamp(pred, min=epslon, max=1-epslon)
		pos_loss = torch.mul(label, 1.0/(pred + epslon) - 1)
		neg_loss = torch.mul(1 - label, 1.0/(1 - pred + epslon) - 1)
		loss =  self.weight[0]*pos_loss + self.weight[1]*neg_loss
		return torch.mean(loss)
