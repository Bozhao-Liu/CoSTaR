from numpy import isnan, inf, savetxt
import numpy as np
import torch
import torch.nn.functional as F
import logging, gc
from torch.utils.checkpoint import checkpoint
try:
	from torch.amp import GradScaler, autocast
except ImportError:
	from torch.cuda.amp import GradScaler, autocast

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from Evaluation_Matix import *
from LossFunction.box_segmentation_loss import *
from utils import *
import model_loader
from data_loader import fetch_dataloader
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from contextlib import nullcontext

class Solver:
	def __init__(self, args, params, CViter):
		def init_weights(m):
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0.01)
		torch.cuda.empty_cache() 
		self.args = args
		self.params = params
		self.CViter = CViter
		self.dataloaders = fetch_dataloader(['train', 'val', 'test'], params, CViter) 
		self.model = model_loader.loadModel(args.network, params.channels).cuda().to(memory_format=torch.channels_last)
		#self.model.apply(init_weights)
		self.optimizer = torch.optim.Adam(	self.model.parameters(), 
							params.hyperparam.learning_rate, 
							betas=(0.9, 0.999), 
							eps=1e-08, 
							weight_decay = params.weight_decay, 
							amsgrad=False)

		
		loss_functions = {}
		from loss import get_loss
		loss_functions['BCE'] = nn.BCEWithLogitsLoss(reduction='none')
		from LossFunction.Diceloss import ContrastiveDiceLoss
		loss_functions['Dice'] = ContrastiveDiceLoss()
		#loss_functions['Tversky'] = MaskedTverskyLoss()
		#self.loss_functions = loss_functions
		self.loss_fn = SoftBoxLoss(loss_functions)
		self.contrastive = ContrastiveLoss()
		gpu_name = torch.cuda.get_device_name()

		if '3090' in gpu_name:
			self.use_amp = True
			self.amp_dtype = torch.float16
			self.use_scaler = True
		else:
			# A100 / newer GPUs
			self.use_amp = True
			self.amp_dtype = torch.bfloat16
			self.use_scaler = False   
			
		self.scaler = GradScaler(enabled=self.use_scaler)
		torch.backends.cuda.matmul.allow_tf32 = True 
		torch.backends.cudnn.allow_tf32 = True

		if '3090' in torch.cuda.get_device_name():
			self.autocast_dtype = torch.float16
		else:
			self.autocast_dtype = torch.bfloat16  # use this on A100
	
	def __step__(self, epoch):
		torch.cuda.empty_cache()
		logging.info("Training")

		losses = AverageMeter()
		#cons_meter = AverageMeter()

		self.model.train()

		if '3090' in torch.cuda.get_device_name():
			t = tqdm(total=len(self.dataloaders['train']), unit='batch', leave=0, desc='training')

		for i, (datas, label) in enumerate(self.dataloaders['train']):

			datas = datas.cuda(non_blocking=True).float().to(memory_format=torch.channels_last)
			label = label.cuda(non_blocking=True).float()
			with torch.no_grad():
				imgs_aug, masks, aug_list = apply_tta_augmentations(datas, label)

			B, A, C, H, W = masks.shape
			imgs_aug = imgs_aug.reshape(B*A, -1, H, W).to(memory_format=torch.channels_last)
			
			masks = masks.reshape(B*A, C, H, W)

			self.optimizer.zero_grad(set_to_none=True)

			# -----------------------------
			# 1. Forward (AMP ON)
			# -----------------------------
			with autocast(device_type='cuda',
						dtype=self.amp_dtype,
						enabled=self.use_amp):
				output = self.model(imgs_aug)   # (B*A,1,H,W)
				
			#del imgs_aug
			# -----------------------------
			# 2. TTA (no grad anyway)
			# -----------------------------
			output, masks, pseudo, variance = tta(
				output, masks, (B, A, C, H, W), aug_list
			)
			'''
			log_txt(
				f"debug/epoch_{epoch}_train.txt",{
					"tta_pred_std_across_views": float(variance.mean()),
					"tta_pseudo_range": (float(pseudo.min()), float(pseudo.max())),
				}, i)'''
			# -----------------------------
			# 3. LOSS
			# -----------------------------
			
			output   = output.float()
			pseudo   = pseudo.detach().float()
			variance = variance.float()
			masks	= masks.float()

			
			cost  = self.loss_fn(
				preds=output,
				boxes_mask=masks,
				current_epoch=epoch,
				avg_preds=pseudo,
				uncertainty=variance
			)
			
			if epoch >= 10:
				cost = cost + 1 * self.contrastive(output, pseudo, variance, masks)
				
			#variance encourage loss to prevent early collapse 
			if epoch < 5:
				#cost += 0.1 * (-variance * masks).mean()
				v_target = 0.2
				kernel = torch.ones(1,1,3,3, device=masks.device)
				eroded = (F.conv2d(masks, kernel, padding=1) == 9).float()
				boundary = masks - eroded
				cost += 0.01 * ((v_target - variance) * boundary).clamp(min=0).mean()
			else:
				v_target = 0.15
				cost += 0.01 * ((variance * masks - v_target).clamp(min=0)).mean()
			
			#punish variance outside ROI
			cost += 0.05 * (variance * (1 - masks)).mean()
			
			if self.use_scaler:
				# (3090)
				self.scaler.scale(cost).backward()
				self.scaler.unscale_(self.optimizer)
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
				self.scaler.step(self.optimizer)
				self.scaler.update()
				
			else:
				# (A100)
				cost.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
				self.optimizer.step()
				
			losses.update(cost.detach().cpu().item(), B)
			#cons_meter.update(consistency_loss.detach().cpu().item(), B)


			if '3090' in torch.cuda.get_device_name():
				t.set_postfix(
					gpu=torch.cuda.max_memory_allocated() / 1024**3,
					loss=losses()#,cost = cost.detach().cpu().mean().item()
					#cons=cons_meter()
				)
				t.update()

			del output, pseudo, variance, cost, masks
			gc.collect()
			
		if '3090' in torch.cuda.get_device_name():
			del t

		loss = losses()
		del losses#, cons_meter
		return loss
	
	
	def validate(self, epoch, dataset_type='val'):

		torch.cuda.empty_cache()
		logging.info("Validating")

		losses = AverageMeter()
		variances = AverageMeter()
		BCEloss = nn.BCEWithLogitsLoss(reduction='mean')
		from LossFunction.Diceloss import DiceLoss
		DICEloss = DiceLoss()

		self.model.eval()

		if '3090' in torch.cuda.get_device_name():
			t = tqdm(total=len(self.dataloaders[dataset_type]), unit='batch', leave=0, desc='validating')

		for i, (datas, label) in enumerate(self.dataloaders[dataset_type]):

			with torch.no_grad():
				datas = datas.cuda(non_blocking=True).float().to(memory_format=torch.channels_last)
				label = label.cuda(non_blocking=True).float()
					
				with torch.no_grad():
					imgs_aug, _, aug_list = apply_tta_augmentations(datas, label)
					
				B, A, C, H, W = imgs_aug.shape
				imgs_aug = imgs_aug.reshape(B*A, C, H, W).to(memory_format=torch.channels_last)

				with autocast(device_type='cuda',
							dtype=self.amp_dtype,
							enabled=self.use_amp):
					output = self.model(imgs_aug)   # (B*A,1,H,W)
				
				labels = torch.zeros((B, A, 1, H, W), device=label.device, dtype=label.dtype)
				
				for j, aug_fn in enumerate(aug_list):

					labels[:, j] = aug_fn(label, reverse=False, is_mask=True)
					
				labels = labels.reshape(B*A, 1, H, W).to(memory_format=torch.channels_last)	
				
				loss = BCEloss(output, labels.float())
				loss = loss + DICEloss(output, labels.float())
				
				# measure record cost
				losses.update(loss.cpu().data.numpy(), B*A)
				
				del output, labels, loss, imgs_aug
				if '3090' in torch.cuda.get_device_name():
					t.set_postfix(
						gpu=torch.cuda.max_memory_allocated() / 1024**3,
						loss=losses(),
					)
					t.update()
				
				gc.collect()
				torch.cuda.empty_cache()

		if '3090' in torch.cuda.get_device_name():
			del t

		loss = losses()
		var  = variances()

		del losses, variances

		return loss, 1#var
			
	def test(self, thr):
		from PIL import Image
		def saveimg(i, imgs, prop, label, path):	
			imgpath = os.path.join(path, 'minibatch_' +  str(i))
			if not os.path.exists(imgpath):
				os.makedirs(imgpath)
			for m in range(imgs.shape[0]):
				imgname = os.path.join(imgpath, 'img_' + str(m) + '.png')
				#print(imgs[img])
				save_image(torch.Tensor(imgs[m]), imgname)
				for p in range(prop.shape[1]):
					propname = os.path.join(imgpath, 'img_' + str(m) + '_condition_' + str(p) + 'prop.png')
					labelname = os.path.join(imgpath, 'img_' + str(m) + '_condition_' + str(p) + 'label_' + str(np.sum(label[m, p])>0) + '.png')
					
					img = Image.fromarray(np.uint8(label[m, p] * 255) , 'L')
					img.save(labelname)
					img = Image.fromarray(np.uint8(prop[m, p] * 255) , 'L')
					img.save(propname)
					
		def createpath():
			import shutil
			path = './Result'
			if not os.path.exists(path):
				os.makedirs(path)
				
			path = os.path.join(path, 'prediction')
			if not os.path.exists(path):
				os.makedirs(path)
				
			path = os.path.join(path, self.args.network)	
			if not os.path.exists(path):
				os.makedirs(path)
				
			path = os.path.join(path, self.args.loss)					
			if not os.path.exists(path):
				os.makedirs(path)
			shutil.copyfile('getmatrix.py', os.path.join(path, 'getmatrix.py'))	
			shutil.copyfile('printtable.py', os.path.join(path, 'printtable.py'))
			if os.path.exists(os.path.join(path, 'matrix.json')):
				os.remove(os.path.join(path, 'matrix.json'))	
			
			cv_iter = '_'.join(tuple(map(str, self.CViter)))
			path = os.path.join(path, cv_iter)	
			if os.path.exists(path):
				import shutil
				shutil.rmtree(path, ignore_errors=True)
				
			os.makedirs(path)
			
			return path
			
		torch.cuda.empty_cache()
		logging.warning("testing")

		self.__resume_checkpoint__('best')
		savepath = createpath()

		self.model.eval()

		print('gpu before testing: ', torch.cuda.max_memory_allocated() / 1024**3)
		with tqdm(total=len(self.dataloaders['test']), unit='batch', leave=0, desc='testing') as t:
			for i, (datas, label) in enumerate(self.dataloaders['test']):
				with torch.no_grad():
					label = label.cpu().float().numpy()

				datas = datas.cuda(non_blocking=True).float().to(memory_format=torch.channels_last)
				B, C, H, W = datas.shape

				prob = self.model(datas)
				prob = torch.sigmoid(prob).detach()

				# ---- convert for saving ----
				with torch.no_grad():
					datas = datas.cpu().float().numpy()
					prob  = prob.cpu().float().numpy()

				saveimg(i, datas, prob, label, savepath)
				
				del datas, prob, label

				gc.collect()

				t.update()
		
	def thresholds(self, load:bool = True):
		def get_dirname(load):
			path = './Result'
			if not os.path.exists(path):
				os.makedirs(path)
				
			path = os.path.join(path, 'Threshold')
			if not os.path.exists(path):
				os.makedirs(path)
				
			path = os.path.join(path, self.args.network)	
			if not os.path.exists(path):
				os.makedirs(path)
				
			path = os.path.join(path, self.args.loss)	
			if not os.path.exists(path):
				os.makedirs(path)
				
			cv_iter = '_'.join(tuple(map(str, self.CViter)))
			path = os.path.join(path, cv_iter + '.txt')	
			if not os.path.exists(path):
				load = False
				
			return path, load
		
		def save_threshold(path, t):
			np.savetxt(path, np.array(t[0]))
			import json
			with open(path.replace("txt", "json"), 'w') as file:
				json.dump(t[1], file, indent=4)	
			
		def load_threshold(path):
			return np.loadtxt(path)
			
		t_path, load = get_dirname(load)	
		if load:
			return load_threshold(t_path)
			
		torch.cuda.empty_cache() 
		logging.info("testing")
		epoch, _ = self.__resume_checkpoint__('best')
		print('best epoch at', epoch)
			
		meter = ThresholdMeter(self.params.channels)
		# switch to evaluate mode
		self.model.eval()
		if '3090' in torch.cuda.get_device_name():
			t = tqdm(total=len(self.dataloaders['val']), unit = 'batch', leave = 0, desc = 'testing')

		for i, (datas, label) in enumerate(self.dataloaders['val']):

			datas = datas.cuda(non_blocking=True).float().to(memory_format=torch.channels_last)

			with torch.no_grad():
				with autocast(device_type='cuda', dtype=self.autocast_dtype):
					output = self.model(datas)

			prob = torch.sigmoid(output).cpu().float().numpy()
			label_np = label.cpu().float().numpy()

			meter.update(prob, label_np)

			del output, prob, label_np
			gc.collect()

			if '3090' in torch.cuda.get_device_name():
				t.update()
				
		save_threshold(t_path, meter())
		
		if '3090' in torch.cuda.get_device_name():
			del t
		th = meter()[0]
		del meter
		return th


	def train(self):
		start_epoch = 0
		ESschedular = EarlyStopComposite()
		torch.autograd.set_detect_anomaly(True)

		if self.args.resume:
			logging.info('Resuming Checkpoint')
			updates = {}
			start_epoch, scheduler = self.__resume_checkpoint__('')
			ESschedular.update(scheduler)
			if not start_epoch < self.params.epochs:
				logging.info('Skipping training for finished model\n')
				return 0
		
		logging.info('	Starting With Best loss = {best_score:.4f}'.format(best_score = ESschedular.best_score))
		logging.info('Initialize training from {} to {} epochs'.format(start_epoch, self.params.epochs))
		with tqdm(total=self.params.epochs - start_epoch, leave = 0) as t:
			for epoch in range(start_epoch, self.params.epochs):
				logging.info('CV [{}], Training Epoch: [{}/{}]'.format('_'.join(tuple(map(str, self.CViter))), epoch+1, self.params.epochs))
				
				if epoch == 0:
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = self.params.hyperparam.learning_rate * 0.1
				elif epoch == 10:
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = self.params.hyperparam.learning_rate * 0.5
				elif epoch == 20:
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = self.params.hyperparam.learning_rate
				
				self.__step__(epoch)
				gc.collect()
				# evaluate on validation set
				loss, var = self.validate(epoch)
				score = ESschedular.step(loss, var )
				
				gc.collect()

				# remember best model and save checkpoint
				logging.info('	loss {loss:.4f};\n'.format(loss = loss))		
				if score['improved']:
					self.__save_checkpoint__({
						'epoch': epoch + 1,
						'state_dict': self.model.state_dict(),
						'loss': score['best_score'],
						'optimizer' : self.optimizer.state_dict(),
						'ESschedular': ESschedular.dicti()
						}, 'best')
					logging.info('	Saved Best model with  \n{} \n'.format(score['best_score']))
					
				if score['stalled']:
					self.__learning_rate_decay__()

				self.__save_checkpoint__({
						'epoch': epoch + 1,
						'state_dict': self.model.state_dict(),
						'loss': score['best_score'],
						'optimizer' : self.optimizer.state_dict(),
						'ESschedular': ESschedular.dicti()
						}, '')
				
				t.set_postfix(gpu = torch.cuda.max_memory_allocated() / 1024**3, loss = loss, var = var, best_score = score['best_score'])
				t.update()
			
		best_score = ESschedular.best_score
		del ESschedular
		gc.collect()

		logging.info('Training finalized with best average loss {}\n'.format(best_score))
		return best_score
		
	def __save_checkpoint__(self, state, checkpoint_type):
		checkpointpath, checkpointfile = get_checkpointname(	self.args, 
									checkpoint_type, 
									self.CViter)
		if not os.path.isdir(checkpointpath):
			os.mkdir(checkpointpath)
			
		torch.save(state, checkpointfile)


	def __resume_checkpoint__(self, checkpoint_type):
		_, checkpointfile = get_checkpointname(self.args, checkpoint_type, self.CViter)
		
		if not os.path.isfile(checkpointfile):
			return 0, EarlyStopComposite().dicti()
		else:
			logging.info("Loading checkpoint {}".format(checkpointfile))
			checkpoint = torch.load(checkpointfile, weights_only = False)

			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
				
			return checkpoint['epoch'], checkpoint['ESschedular']
			
	def __learning_rate_decay__(self):
		if self.params.hyperparam.lrDecay < 1:
			for param_group in self.optimizer.param_groups:
				param_group['lr'] = param_group['lr']*self.params.hyperparam.lrDecay
