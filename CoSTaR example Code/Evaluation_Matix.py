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

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
		
	def __call__(self):
		return self.avg
		
class EvalMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, size:int = 12):
		self.size = size
		self.reset()
		

	def reset(self):
		self.tp = np.zeros(self.size)
		self.fp = np.zeros(self.size)
		self.tn = np.zeros(self.size)
		self.fn = np.zeros(self.size)

	def update(self, pred, label):
		assert pred.shape[1] == self.size, 'Expected prediction dimension of {}, but got {}'.format(self.size, pred.shape[1])
		
		assert label.shape[1] == self.size, 'Expected label dimension of {}, but got {}'.format(self.size, label.shape[1])

		self.tp = self.tp + np.sum(label * np.array(pred==label).astype(float), axis = (0,2,3))
		self.fp = self.fp + np.sum((1-label) * np.array(pred!=label).astype(float), axis = (0,2,3))
		self.tn = self.tn + np.sum((1-label) * np.array(pred==label).astype(float), axis = (0,2,3))
		self.fn = self.fn + np.sum(label * np.array(pred!=label).astype(float), axis = (0,2,3))

	def precision(self):
		return self.tp/(self.tp+ self.fp)

	def recall(self):
		return self.tp/(self.tp+ self.fn)

	def F1(self):
		return 2*self.precision()*self.recall()/(self.precision()+self.recall())
		
	def acc(self):
		return (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn)
		
	def dict(self):
		return {'precision':self.precision(), 'recall':self.recall(), 'F1':self.F1(), 'acc': self.acc()}
		
		
class ThresholdMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, num_classes:int = 12):
		self.num_classes = num_classes
		self.thresholds = np.array(range(40,77,2))/100
		self.reset()

	def reset(self):
		self.union = defaultdict(lambda: 0)
		self.intersection = defaultdict(lambda: 0)

	def update(self, pred, label):
		assert pred.shape[1] == self.num_classes, 'Expected prediction dimension of {}, but got {}'.format(self.num_classes, pred.shape[1])
		
		assert label.shape[1] == self.num_classes, 'Expected label dimension of {}, but got {}'.format(self.num_classes, label.shape[1])
		label = np.asarray(label).astype(bool)

		for threshold in self.thresholds:
			predict = np.asarray(pred)>threshold
			inter = np.sum(np.logical_and(predict, label))
			uni = np.sum(np.logical_or(predict, label))
			self.union[threshold] += uni
			self.intersection[threshold] += inter

	def __call__(self):
		IOU = 0
		IOUs = {}
		threshold = 0
		for key in self.union:
			IOUs[key] = self.intersection[key] /self.union[key]
			if self.intersection[key] /self.union[key]>IOU:
				IOU = self.intersection[key] /self.union[key]
				threshold = key
				
		return [threshold], IOUs
		
		
def get_AUC(outputs):
	AUC = []
	for i in range(outputs[0].shape[1]):
		fpr, tpr, thresholds = metrics.roc_curve(outputs[1][:, i], outputs[0][:, i], pos_label=1)
		AUC.append(metrics.auc(fpr, tpr))
	return np.mean(AUC)
	
def get_threshold(outputs, network:str, load_threshold:bool = True, CViter:tuple = (0, 1)):
	def get_dirname():
		path = './Result'
		if not os.path.exists(path):
			os.makedirs(path)
			
		path = os.path.join(path, 'Threshold')
		if not os.path.exists(path):
			os.makedirs(path)
			
		path = os.path.join(path, network)	
		if not os.path.exists(path):
			os.makedirs(path)
			
		cv_iter = '_'.join(tuple(map(str, CViter)))
		path = os.path.join(path, cv_iter)	
		if not os.path.exists(path):
			os.makedirs(path)
			load_threshold = False
			
		for i in range(outputs[0].shape[1]):
			subpath = os.path.join(path, str(i))
			if not os.path.exists(subpath):
				load_threshold = False
			
		return path
		
	def save_threshold(path, t):
		t =  np.array(t)
		for i in range(t.shape[0]):
			np.savetxt(os.path.join(path, str(i)), t)
		
	def load_threshold(path):
		t = []
		for i in range(output[0].shape[1]):
			t.append(np.loadtxt(os.path.join(path, str(i))))
			
		return t
	
	t_path = get_dirname()
	
	if load_threshold:
		return load_threshold(t_path)
		
	thresholds = []
	for i in range(output[0].shape[1]):
		threshold = []
		for x in range(output[0].shape[2]):
			thresholdx = []
			for y in range(output[0].shape[3]):
				precision, recall, t = metrics.precision_recall_curve(output[1][:, i, x, y], output[0][:, i, x, y], pos_label = 1 )
				f1_scores = 2*recall*precision/(recall+precision + 1e-8)
				ind = np.argmax(f1_scores)
				t = t[ind]
				thresholdx.append(t)
				#print(t)
			threshold.append(thresholdx)
		thresholds.append(threshold)
		
	save_threshold(t_path, thresholds)
		
	return thresholds
	
def get_eval_multi(outputs, thresholds: list = [0.5]*12):
	result = defaultdict(list)

	for i in range(outputs[0].shape[1]):
		fpr, tpr, _ = metrics.roc_curve(outputs[1][:, i], outputs[0][:, i], pos_label=1)
		result['AUC'].append(metrics.auc(fpr, tpr))		
		outputs[0][:, i] = outputs[0][:, i] > thresholds[i]
		
		result['threshold'].append(thresholds[i])
		result['acc'].append(metrics.accuracy_score(outputs[1][:, i], outputs[0][:, i]))
		result['Precision'].append(metrics.precision_score(outputs[1][:, i], outputs[0][:, i], average='weighted', zero_division = 0))
		result['Recall'].append(metrics.recall_score(outputs[1][:, i], outputs[0][:, i], average='weighted'))
		result['F0.5'].append(metrics.fbeta_score(outputs[1][:, i], outputs[0][:, i], average='weighted', beta=0.5))
		result['F0'].append(metrics.fbeta_score(outputs[1][:, i], outputs[0][:, i], average='weighted', beta=0))
		result['F1'].append(metrics.f1_score(outputs[1][:, i], outputs[0][:, i], average='weighted'))
	
	return result
	
'''
def plot_AUC_SD(netlist, evalmatices):
	plt.clf()
	possitive_ratio = np.loadtxt("./data/possitive_ratio.txt", dtype=float)
	logging.warning('	Creating standard diviation image for {}'.format('-'.join(netlist)))
	png_file = 'Crossvalidation_Analysis_{}.tex'.format('-'.join(netlist))

	if len(netlist) == 0:
		return


	plt.clf()
	fig, ax = plt.subplots(2)
	fig.suptitle('Accruacy, F1 for {}'.format('-'.join(netlist)))
	
	data = []
	for net in netlist:
		data.append(np.array(evalmatices[net]).T[0])

	ax[0].boxplot(data, showfliers=False)
	ax[0].set_ylabel('Accruacy')

	data = []
	for net in netlist:
		data.append(np.array(evalmatices[net]).T[1])

	ax[1].boxplot(data, showfliers=False)
	ax[1].set_ylabel('F1')
	ax[1].set_xticklabels(netlist, fontsize=10)
	
	import tikzplotlib
	
	logging.warning('	Saving standard diviation image for {} \n'.format('-'.join(netlist)))
	tikzplotlib.save(png_file)
'''
	
