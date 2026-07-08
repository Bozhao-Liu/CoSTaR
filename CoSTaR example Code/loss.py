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

def get_loss(loss_name, Hyperparam):
	loss_name = loss_name.lower()
	loss_name = loss_name.replace('\r', '')
	loss_name = loss_name.replace(' ', '')
	if loss_name == 'bce':
		return nn.BCEWithLogitsLoss(reduction = 'none')
	elif loss_name == 'ece': 
		from LossFunction.ECE import ECE_loss
		return ECE_loss()

	elif loss_name == 'focal': 
		Hyperparam.gamma = 2
		from LossFunction.Focal import Focal_loss
		return Focal_loss(Hyperparam.gamma)

	elif loss_name == 'f-ece': 
		Hyperparam.gamma = 2
		from LossFunction.FECE import F_ECE_loss
		return F_ECE_loss(Hyperparam.gamma)

	elif loss_name == 'asl':
		from LossFunction.ASL import ASL
		return ASL()

	elif loss_name == 'recall': 
		from LossFunction.RecallLoss import RecallLoss
		return RecallLoss()

	elif loss_name == 'rangeloss':
		from LossFunction.Range import RangeLoss
		return RangeLoss(ids_per_batch=100, imgs_per_id=1)
	else:
		logging.error("No loss function with the name {} found, please check your spelling.".format(loss_name))
		logging.error("loss function List:")
		logging.error("	BCE")
		logging.error("	ECE")
		logging.error("	focal")
		logging.error("	ASL")
		logging.error("	F-ECE")
		logging.error("	Recall")
		logging.error("	RangeLoss")
		import sys
		sys.exit()