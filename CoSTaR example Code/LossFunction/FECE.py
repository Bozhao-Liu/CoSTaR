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

class F_ECE_loss(nn.Module):
	def __init__(self, gamma: int = 2):
		super(F_ECE_loss, self).__init__()
		self.gamma = gamma

	def forward(self, pred, label):
		label = (label > 0.5).float()
		pred = torch.sigmoid(pred)
		pred = torch.clamp(pred, min=0.05, max=0.95)
		pos_loss = torch.mul(label, 1.0/(pred + epslon) - 1)
		
		exp_neg = torch.mul(1 - label, -torch.pow(pred, self.gamma))
		neg_loss = torch.mul(exp_neg, torch.log(1 - pred))
		loss =  pos_loss + neg_loss
		if torch.isnan(loss).any():
			print("NaN detected!")
			print("pred min/max:", pred.min().item(), pred.max().item())
			print("label min/max:", label.min().item(), label.max().item())
			print("pos_loss:", pos_loss.mean().item())
			print("neg_loss:", neg_loss.mean().item())
			exit()
		return torch.mean(loss)
		
class ECE_F_loss(nn.Module):
	def __init__(self, gamma: int = 2):
		super(ECE_F_loss, self).__init__()
		self.gamma = gamma

	def forward(self, pred, label):
		label = (label > 0.5).float()
		pred = torch.sigmoid(pred)
		pred = torch.clamp(pred, min=0.05, max=0.95)
		exp_pos = torch.mul(label, -torch.pow(1-pred, self.gamma))
		pos_loss = torch.mul(exp_pos, torch.log(pred))
		
		neg_loss = torch.mul(1-label, 1.0/(1-pred + epslon) - 1)
		loss =  pos_loss + neg_loss
		if torch.isnan(loss).any():
			print("NaN detected!")
			print("pred min/max:", pred.min().item(), pred.max().item())
			print("label min/max:", label.min().item(), label.max().item())
			print("pos_loss:", pos_loss.mean().item())
			print("neg_loss:", neg_loss.mean().item())
			exit()
		return torch.mean(loss)