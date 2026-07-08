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

class Focal_loss(nn.Module):
	def __init__(self, gamma: Union[tuple, int] =  (2, 2)):
		super(Focal_loss, self).__init__()
		if type(gamma) is int:
			self.gamma = [gamma, gamma]
		else:
			self.gamma = [gamma[0], gamma[1]]
			
	def forward(self, pred, label):
		pred = torch.clamp(pred, min=epslon, max=1-epslon)
		exp_pos = torch.mul(label, -torch.pow(1-pred, self.gamma[0]))
		pos_loss = torch.mul(exp_pos, torch.log(pred))
		exp_neg = torch.mul(1 - label, -torch.pow(pred, self.gamma[0]))
		neg_loss = torch.mul(exp_neg, torch.log(1-pred))
		loss =  pos_loss + neg_loss
		loss = torch.mean(loss) 
		if not np.isnan(loss.cpu().data.numpy().any()):	
			return loss
		else:
			print('#####NAN#####')
			return 1e5