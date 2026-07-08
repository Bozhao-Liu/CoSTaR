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

		
class RecallLoss(nn.Module):
	""" An unofficial implementation of
		<Recall Loss for Imbalanced Image Classification and Semantic Segmentation>
		Created by: Zhang Shuai
		Email: shuaizzz666@gmail.com
		recall = TP / (TP + FN)
	Args:
		weight: An array of shape [C,]
		predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
		target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
	Return:
		diceloss
	"""
	def __init__(self, weight=None):
		super(RecallLoss, self).__init__()
		if weight is not None:
			weight = torch.Tensor(weight)
			self.weight = weight / torch.sum(weight) # Normalized weight
		self.smooth = 1e-5

	def forward(self, input, target):
		N, C = input.size()[:2]
		_, ind = torch.max(input, 1)# # (N, C, ) ==> (N, 1,)
		predict = torch.zeros(input.size()).cuda()

		for i in range(len(ind)):
			predict[i][ind[i]]=1


		true_positive = torch.sum(predict * target, dim=0)
		positive = torch.sum(target, dim=0)

		recall = (true_positive + self.smooth) / (positive + self.smooth)  # (N, C)

		if hasattr(self, 'weight'):
			if self.weight.type() != input.type():
				self.weight = self.weight.type_as(input)
				recall = recall * self.weight * C  # (N, C)
		recall_loss = 1 - recall  # 1
		pos_loss = -torch.mul(recall_loss, torch.log(input))
		pos_loss = torch.mul(target, pos_loss)
		neg_loss = -torch.mul(1-recall_loss, torch.log(1-input))
		neg_loss = torch.mul(1-target, neg_loss)

		recall_loss = pos_loss + neg_loss

		return torch.mean(recall_loss)
