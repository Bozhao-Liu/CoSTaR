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

class ASL(nn.Module):
	''' Notice - optimized version, minimizes memory allocation and gpu uploading,
	favors inplace operations'''

	def __init__(self, gamma_neg=4, gamma_pos=1):
		super(ASL, self).__init__()

		self.gamma_neg = gamma_neg
		self.gamma_pos = gamma_pos

	def forward(self, pred, label):
		""""
		Parameters
		----------
		x: input logits
		y: targets (multi-label binarized vector)
		"""
		pred = torch.clamp(pred, min=epslon, max=1-epslon)
		neg_label = 1 - label #1-y

		# Calculating Probabilities
		P_pos = pred
		P_neg = 1.0 - pred

		P_pos.clamp_(min=epslon)
		P_neg.add_(0.05).clamp_(max=1, min=epslon)

		# Basic CE calculation
		loss_pos = -torch.mul(label, torch.log(P_pos))
		loss_neg = -torch.mul(neg_label, torch.log(P_neg))

		# Asymmetric Focusing
		if self.gamma_neg > 0 or self.gamma_pos > 0:
			loss_pos = torch.mul(torch.pow(P_neg, self.gamma_pos), loss_pos)
			loss_neg = torch.mul(torch.pow(P_pos, self.gamma_neg), loss_neg)
						  
		loss = torch.add(loss_pos, loss_neg)
				
		return loss.mean()