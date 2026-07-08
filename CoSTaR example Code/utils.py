import json
import logging
import os
import shutil
import torch
import torch.nn as nn
import numpy as np
from itertools import product
import random

import torch.nn.functional as F
import torchvision.transforms.functional as TF
try:
	from torch.hub import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url

try:
	from configparser import ConfigParser
except ImportError:
	from ConfigParser import ConfigParser  # ver. < 3.0

class Params():
	"""Class that loads hyperparameters from a json file.

	Example:
	```
	params = Params(json_path)
	print(params.learning_rate)
	params.learning_rate = 0.5  # change the value of learning_rate in params
	```
	"""

	def __init__(self, model_dir, network, paramtype = 'params', loss_fn = '', checkpoint = False):
		assert paramtype in ['params','Hyperparams'], 'Param type {} not found'.format(paramtype)

		self.Best_loss = np.inf
		json_path = os.path.join(model_dir, network)
		loss_fn = loss_fn.replace(" ", "")
		self.__json_file = os.path.join(json_path, '{}{}.json'.format(paramtype, loss_fn))
		if not os.path.isfile(self.__json_file):
			logging.warning("Can not find json file {}".format(self.__json_file))
			self.__json_file = os.path.join(json_path, '{}.json'.format(paramtype))
			
		#logging.warning("Loading json file {}".format(self.__json_file))
		assert os.path.isfile(self.__json_file), "Can not find File {}".format(self.__json_file)
		with open(self.__json_file) as f:
			params = json.load(f)

		self.__dict__.update(params)
		self.__json_file = os.path.join(json_path, '{}{}.json'.format(paramtype, loss_fn))
			
		if checkpoint and paramtype in ['Hyperparams']:
			self.__json_file = os.path.join(json_path, '{}_{}_checkpoint.json'.format(paramtype, loss_fn))
		
			logging.warning("Loading json file {}".format(self.__json_file))
				
			if os.path.isfile(self.__json_file):
				with open(self.__json_file) as f:
					params = json.load(f)
					self.__dict__.update(params)
			else:
				self.__dict__ = list(param_search_list(self))[0]

	def update(self, params):
		json_file = self.__json_file
		self.__dict__.update(params)
		self.__json_file = json_file
		
	def save(self):
		json_file = self.__json_file
		del self.__json_file
		with open(json_file, 'w') as f:
			json.dump(self.__dict__, f, indent=4)
		self.__json_file = json_file
		
	def reload(self):
		"""Loads parameters from json file"""
		with open(self.__json_file) as f:
			params = json.load(f)
			self.__dict__.update(params)

	def dict(self):
		"""Gives dict-like access to Params instance by `params.dict['learning_rate']"""
		return self.__dict__
		
def set_params(model_dir, network, paramtype):
	params = Params(model_dir, network, paramtype)

	# use GPU if available
	params.cuda = torch.cuda.is_available()

	# Set the random seed for reproducible experiments
	torch.manual_seed(230)
	if params.cuda: 
		torch.cuda.manual_seed(230)

	return params
	
def param_search_list(hyperparams):
	from collections import defaultdict
	params = defaultdict(list)
	for key in hyperparams.dict().keys():
		if key == 'learning_rate':
			params[key] = [1e-5, 5e-6, 1e-6]
		elif key == 'dropout_rate':
			params[key] = list(np.arange(0.1, 1.0, 0.2, dtype = float))	
		elif key == 'lrDecay':
			params[key] = list(np.arange(0.85, 0.96, 0.02, dtype = float))	
		elif 'layer' in key:
			params[key] = [96, 128, 192, 256]
		else:
			params[key] = [hyperparams.dict()[key]]
	keys = params.keys()
	vals = params.values()
	for instance in product(*vals):
		yield dict(zip(keys, instance))		
				
def set_logger(model_dir, network, level = 'info'):
	"""Set the logger to log info in terminal and file `log_path`.

	In general, it is useful to have a logger so that every output to the terminal is saved
	in a permanent file. Here we save it to `model_dir/train.log`.

	Example:
	```
	logging.info("Starting training...")
	```

	Args:
	log_path: (string) where to log
	"""
	log_path = os.path.join(model_dir, network)
	assert os.path.isdir(log_path), "Can not find Path {}".format(log_path)
	log_path = os.path.join(log_path, 'train.log')
	print('Saving {} log to {}'.format(level, log_path))
	level = level.lower()
	logger = logging.getLogger()
	if level == 'warning':
		level = logging.WARNING
	elif level == 'debug':
		level = logging.DEBUG
	elif level == 'error':
		level = logging.ERROR
	elif level == 'critical':
		level = logging.CRITICAL
	else:
		level = logging.INFO
	logger.setLevel(level)

	if not logger.handlers:
		# Logging to a file
		file_handler = logging.FileHandler(log_path)
		file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
		logger.addHandler(file_handler)

		# Logging to console
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(logging.Formatter('%(message)s'))
		logger.addHandler(stream_handler)



def save_dict_to_json(d, json_path):
	"""Saves dict of floats in json file

	Args:
	d: (dict) of float-castable values (np.float, int, float, etc.)
	json_path: (string) path to json file
	"""
	with open(json_path, 'w') as f:
		# We need to convert the values to float for json (it doesn't accept np.array, np.float, )
		d = {k: float(v) for k, v in d.items()}
		json.dump(d, f, indent=4)

	
def get_checkpointname(args, checkpoint_type, CViter):
	checkpointpath = os.path.join(os.path.join(args.model_dir,'Model'), args.network)
	checkpointpath = os.path.join(checkpointpath, 'Checkpoints' + str(checkpoint_type))
	checkpointfile = os.path.join(checkpointpath, 
				'{network}_{cv_iter}_{gamma}.pth.tar'.format(network = args.network+args.loss, 
										 cv_iter = '_'.join(tuple(map(str, CViter))), gamma = args.gamma))
	return checkpointpath, checkpointfile										 

from torchvision.transforms import InterpolationMode
from torchvision.transforms import InterpolationMode
import torch
import torch.nn.functional as F
import math
import random

def make_random_geometric_augs(num_augs=7):

	aug_fns = []

	def identity(x, reverse=False, is_mask=False):
		return x
	aug_fns.append(identity)

	# -------------------------
	# discrete angle pool
	# -------------------------
	#angle_pool = [30, 45, 60, 90, 120, 135, 150]
	angle_pool = np.linspace(1, 179, 179)

	for i in range(num_augs):

		angle = random.choice(angle_pool)#angle_pool[i]
		theta = angle * math.pi / 180.0

		do_hflip = random.random() < 0.5
		do_vflip = random.random() < 0.5

		def aug_fn(x, reverse=False, is_mask=False,
				   angle=angle, theta=theta,
				   do_hflip=do_hflip, do_vflip=do_vflip):

			B, C, H, W = x.shape
			device = x.device
			dtype = x.dtype

			# -------------------------
			# FORWARD: H → V → R
			# REVERSE: R⁻¹ → V → H
			# -------------------------
			if not reverse:
				if do_hflip:
					x = torch.flip(x, [3])
				if do_vflip:
					x = torch.flip(x, [2])

				rot_angle = angle
				rot_theta = theta

			else:
				rot_angle = -angle
				rot_theta = -theta

			# -------------------------
			# rotation
			# -------------------------
			if abs(rot_angle) % 90 == 0:
				# exact rotation (no interpolation)
				k = (rot_angle // 90) % 4
				x = torch.rot90(x, int(k), dims=[2, 3])

			else:
				# affine rotation
				cos_t = math.cos(rot_theta)
				sin_t = math.sin(rot_theta)

				affine = torch.tensor([
					[cos_t, -sin_t, 0.0],
					[sin_t,  cos_t, 0.0]
				], dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)

				grid = F.affine_grid(
					affine,
					size=x.size(),
					align_corners=True
				)

				mode = 'nearest' if is_mask else 'bilinear'

				x = F.grid_sample(
					x,
					grid,
					mode=mode,
					padding_mode='border',
					align_corners=True
				)

			# -------------------------
			# inverse flips
			# -------------------------
			if reverse:
				if do_vflip:
					x = torch.flip(x, [2])
				if do_hflip:
					x = torch.flip(x, [3])

			return x

		aug_fns.append(aug_fn)

	return aug_fns

def mask_to_box_mask(
	mask: torch.Tensor,
	max_margin_ratio: float = 0.1,
	min_margin_ratio: float = 0.0,
	fallback: str = "full",  # ["full", "center", "skip"]
):
	"""
	Binary mask → bounding box mask (robust).

	Args:
		mask: (B,1,H,W) or (B,H,W), binary {0,1}
	Returns:
		box_mask: (B,1,H,W)
	"""

	# -----------------------------
	# 0. shape handling
	# -----------------------------
	if mask.dim() == 4:
		mask = mask.squeeze(1)

	mask = mask.bool()
	B, H, W = mask.shape
	device = mask.device

	# -----------------------------
	# 1. detect valid masks
	# -----------------------------
	area = mask.sum(dim=(1, 2))
	valid = area > 0  # ONLY empty masks are invalid

	# -----------------------------
	# 2. row/col occupancy
	# -----------------------------
	rows = mask.any(dim=2)   # (B,H)
	cols = mask.any(dim=1)   # (B,W)

	# -----------------------------
	# 3. bounding box indices
	# -----------------------------
	y_min = rows.float().argmax(dim=1)
	y_max = H - 1 - torch.flip(rows, dims=[1]).float().argmax(dim=1)

	x_min = cols.float().argmax(dim=1)
	x_max = W - 1 - torch.flip(cols, dims=[1]).float().argmax(dim=1)

	# -----------------------------
	# 4. fallback for empty masks
	# -----------------------------
	invalid = ~valid

	if invalid.any():
		if fallback == "full":
			y_min[invalid] = 0
			y_max[invalid] = H - 1
			x_min[invalid] = 0
			x_max[invalid] = W - 1

		elif fallback == "center":
			cy, cx = H // 2, W // 2
			bh, bw = H // 4, W // 4

			y_min[invalid] = cy - bh
			y_max[invalid] = cy + bh
			x_min[invalid] = cx - bw
			x_max[invalid] = cx + bw

		elif fallback == "skip":
			pass  # handled later

	# -----------------------------
	# 5. box size
	# -----------------------------
	box_h = (y_max - y_min + 1).clamp(min=1)
	box_w = (x_max - x_min + 1).clamp(min=1)

	# -----------------------------
	# 6. stable margins
	# -----------------------------
	rand = torch.rand(B, 4, device=device)

	scale = (max_margin_ratio - min_margin_ratio)

	m_top	= (rand[:, 0] * scale + min_margin_ratio) * box_h
	m_bottom = (rand[:, 1] * scale + min_margin_ratio) * box_h
	m_left   = (rand[:, 2] * scale + min_margin_ratio) * box_w
	m_right  = (rand[:, 3] * scale + min_margin_ratio) * box_w

	m_top	= m_top.long()
	m_bottom = m_bottom.long()
	m_left   = m_left.long()
	m_right  = m_right.long()
	
	min_pixels = 5

	m_top	= torch.maximum(m_top, torch.full_like(m_top, min_pixels))
	m_bottom = torch.maximum(m_bottom, torch.full_like(m_bottom, min_pixels))
	m_left   = torch.maximum(m_left, torch.full_like(m_left, min_pixels))
	m_right  = torch.maximum(m_right, torch.full_like(m_right, min_pixels))

	# -----------------------------
	# 7. expanded box
	# -----------------------------
	y1 = (y_min - m_top).clamp(min=0)
	y2 = (y_max + m_bottom).clamp(max=H - 1)
	x1 = (x_min - m_left).clamp(min=0)
	x2 = (x_max + m_right).clamp(max=W - 1)

	# -----------------------------
	# 8. build mask
	# -----------------------------
	yy = torch.arange(H, device=device).view(1, H, 1)
	xx = torch.arange(W, device=device).view(1, 1, W)

	box_mask = (
		(yy >= y1.view(B, 1, 1)) &
		(yy <= y2.view(B, 1, 1)) &
		(xx >= x1.view(B, 1, 1)) &
		(xx <= x2.view(B, 1, 1))
	)

	# -----------------------------
	# 9. skip behavior
	# -----------------------------
	if fallback == "skip":
		box_mask[invalid] = 0

	return box_mask.unsqueeze(1).float()
	

from collections import deque

class EarlyStopComposite:
	def __init__(self, patience=5, eps=1e-8, history=5, gamma=2, lambda_=5.0):
		self.patience = patience
		self.eps = eps
		self.best_score = float('inf')
		self.best_var = float('inf')
		self.stale = 0
		self.switched = False
		self.hist_loss = deque(maxlen=history)
		self.gamma = gamma
		self.lambda_ = lambda_

	def update(self, status):
		assert type(status) is dict
		for k,v in status.items():
			self.__dict__[k] = v

	def dicti(self):
		return self.__dict__

	def step(self, loss, var):
		improved = False
		# -------------------------
		# Phase 1: variance-driven
		# -------------------------
		if not self.switched:

			# robust switch using deque
			if len(self.hist_loss) == self.hist_loss.maxlen:
				prev_mean = sum(self.hist_loss) / len(self.hist_loss)

				if loss < prev_mean:
					self.switched = True
					score = loss * (1.0 + self.lambda_ * ((var + self.eps) ** self.gamma))
					self.best_score = score
					
			if var < self.best_var:
				self.best_var = var
				improved = True
				self.stale = 0
			else:
				self.stale += 1

		# -------------------------
		# Phase 2
		# -------------------------
		else:
			score = loss * (1.0 + self.lambda_ * ((var + self.eps) ** self.gamma))

			if score < self.best_score:
				self.best_score = score
				improved = True
				self.stale = 0
			else:
				self.stale += 1

		self.hist_loss.append(loss)

		stalled = self.stale >= self.patience

		return {
			"score": loss if not self.switched else score,
			"best_score": self.best_score,
			"best_var": self.best_var,
			"improved": improved,
			"stalled": stalled
		}
		
	
def apply_tta_augmentations(imgs, masks):
	aug_list = make_random_geometric_augs(num_augs=7)

	B, C, H, W = imgs.shape
	imgs_list, masks_list = [], []

	for aug_fn in aug_list:
		aug_img  = aug_fn(imgs,  is_mask=False)
		aug_mask = aug_fn(masks, is_mask=True)

		# binarize before box
		aug_mask = (aug_mask > 0.5).float()

		# box AFTER augmentation
		aug_box = mask_to_box_mask(aug_mask)

		imgs_list.append(aug_img)
		masks_list.append(aug_box)

	imgs_aug = torch.stack(imgs_list, dim=1)   # (B,A,C,H,W)
	masks_aug = torch.stack(masks_list, dim=1) # (B,A,1,H,W)
	
	imgs_aug = torch.nan_to_num(imgs_aug, nan=0.0, posinf=1.0, neginf=0.0)
	imgs_aug = imgs_aug.clamp(0.0, 1.0)
	masks_aug = torch.nan_to_num(masks_aug, nan=0.0, posinf=1.0, neginf=0.0)
	masks_aug = masks_aug.clamp(0.0, 1.0)

	return imgs_aug, masks_aug, aug_list
	
def check_tta_roi(lab_j, lab_float, thresh=0.5):
	"""
	lab_j:	 (B,C,H,W) single sample expected
	lab_float: (B,C,H,W) original canonical mask

	Returns:
		ok (bool), density (float), center_ratio (float)
	"""

	# -------------------------
	# binarize
	# -------------------------
	lab_j_bin = (lab_j > thresh)
	orig_bin  = (lab_float > thresh)

	# -------------------------
	# Check 1: density
	# -------------------------
	non_zero = (lab_j > 0.05)  # use 0.05 instead of 0 to avoid mistakes from interpolation

	if non_zero.sum() == 0:
		return False, 0.0, 0.0

	ones_ratio = lab_j_bin[non_zero].float().mean().item()

	density_ok = ones_ratio >= 0.30

	# -------------------------
	# Check 2: center 50% of ones
	# -------------------------
	ys, xs = torch.where(lab_j_bin[0,0])

	if len(ys) == 0:
		return False, ones_ratio, 0.0

	# centroid
	cy = ys.float().mean()
	cx = xs.float().mean()

	# distance to centroid
	dist = (ys.float() - cy)**2 + (xs.float() - cx)**2
	idx = torch.argsort(dist)

	# take center 50%
	k = max(1, len(idx) // 2)
	ys_c = ys[idx[:k]]
	xs_c = xs[idx[:k]]

	# check overlap with original ROI
	center_ratio = orig_bin[0,0,ys_c, xs_c].float().mean().item()

	center_ok = center_ratio >= 0.1  # reasonable threshold

	# -------------------------
	# final decision
	# -------------------------
	ok = density_ok and center_ok

	return ok#, ones_ratio, center_ratio
	
def tta(output, label, shape, aug_list):
	"""
	Args:
		output: (B*A, C, H, W) logits
		label:  (B*A, C, H, W)  # per-view box masks (no assumption of equality)
		shape:  (B, A, C, H, W)

	Returns:
		out_sel:   (B*A, C, H, W)
		lab_sel:   (B*A, C, H, W)
		pseudo:	(B*A, C, H, W)
		var_norm:  (B*A, C, H, W)
	"""
	B, A, C, H, W = shape
	assert list(output.shape) == [B*A, C, H, W], '{} vs {}'.format(list(output.shape), [B*A, C, H, W])

	base_dtype = output.dtype

	# reshape
	out_BA = output.reshape(B, A, C, H, W)
	lab_BA = label.detach().reshape(B, A, C, H, W)

	with torch.no_grad():

		# -----------------------------
		# 1. probability space
		# -----------------------------
		preds = torch.sigmoid(out_BA).to(torch.float32) # (B,A,C,H,W)
		lab_float = lab_BA.float()
		# -----------------------------
		# 2. reverse augmentations
		# -----------------------------
		preds_rev = []
		labs_rev  = []

		for j, aug_fn in enumerate(aug_list):
			preds_rev.append(aug_fn(preds[:, j], reverse=True, is_mask=False))
			labs_rev.append(aug_fn(lab_float[:, j], reverse=True, is_mask=True))

		preds = torch.stack(preds_rev, dim=1)
		lab_float = torch.stack(labs_rev, dim=1)
		# -----------------------------
		# 3. aggregate
		# -----------------------------
		pseudo = preds.mean(dim=1)			  # (B,C,H,W)
		
		lab_out = lab_float.mean(dim=1)
		support = lab_float.max(dim=1).values
		lab_out = lab_out * support
		lab_out = (lab_out > 0.9).float()
		del lab_float

		var = preds.var(dim=1, correction=0)	  # (B,C,H,W)
		del preds

		lo = torch.quantile(var.reshape(B, -1), 0.05, dim=1).reshape(B,1,1,1)
		hi = torch.quantile(var.reshape(B, -1), 0.95, dim=1).reshape(B,1,1,1)
		var_out = ((var - lo) / (hi - lo + 1e-6)).clamp(0,1)
		var_out = var_out * (lab_out > 0)
		del hi
		del lo
		
		pseudo_list = []
		var_list = []
		lab_list = []

		for j, aug_fn in enumerate(aug_list):

			
			var_j	= aug_fn(var_out, reverse=False, is_mask=False)
			lab_j	= aug_fn(lab_out, reverse=False, is_mask=True)
			pseudo_j = aug_fn(pseudo, reverse=False, is_mask=False)*(lab_j>0.1)
			
			assert check_tta_roi(lab_j, lab_BA[:, j]), 'tta center misaligned'

			pseudo_list.append(pseudo_j)
			var_list.append(var_j)
			lab_list.append(lab_j)
			
		
		pseudo  = torch.stack(pseudo_list, dim=1)   # (B,A,C,H,W)
		var_out = torch.stack(var_list, dim=1)
		lab_out = torch.stack(lab_list, dim=1)
			
	lab_out = lab_out.reshape(B * A, C, H, W)
	pseudo  = pseudo.clone().detach().reshape(B * A, C, H, W)
	var_out = var_out.reshape(B * A, C, H, W)
	
	return (
		output.to(base_dtype),
		lab_out.to(base_dtype),
		pseudo,
		var_out
	)
	
from PIL import Image
import os
import torch

def to_img(tensor):
	"""
	Convert tensor (C,H,W) or (H,W) → uint8 image (H,W)
	"""
	t = tensor.detach().cpu()

	if t.ndim == 3:
		t = t[0]

	# logits → prob
	if t.max() > 1 or t.min() < 0:
		t = torch.sigmoid(t)

	# normalize
	t = t.clamp(0,1)
	t = (t * 255).to(torch.uint8)

	return t.numpy()


def save_grid(img, out, mask, pseudo, var, gt, path):
	"""
	Save a horizontal grid of 5 images
	"""
	imgs = [
		to_img(img),
		to_img(out),
		to_img(mask),
		to_img(pseudo),
		to_img(var),
		to_img(gt)
	]

	# convert to PIL
	imgs = [Image.fromarray(x) for x in imgs]

	# assume same size
	W, H = imgs[0].size

	grid = Image.new('L', (W, H * len(imgs)))

	for i, im in enumerate(imgs):
		grid.paste(im, (0, i * H))

	os.makedirs(os.path.dirname(path), exist_ok=True)
	grid.save(path)

if __name__ == "__main__":

	import os
	import torch
	import matplotlib.pyplot as plt

	torch.manual_seed(0)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# -----------------------------
	# 1. synthetic data
	# -----------------------------
	B, C, H, W = 2, 1, 128, 128

	imgs = torch.zeros((B, C, H, W), device=device)
	masks = torch.zeros((B, 1, H, W), device=device)

	# draw simple shapes
	# -----------------------------
	# small, asymmetric ROI
	# -----------------------------
	for b in range(B):

		# small object (10–20 px)
		h = torch.randint(8, 16, (1,)).item()
		w = torch.randint(10, 18, (1,)).item()

		# random location (not centered)
		y1 = torch.randint(10, H - h - 10, (1,)).item()
		x1 = torch.randint(10, W - w - 10, (1,)).item()

		y2 = y1 + h
		x2 = x1 + w

		masks[b, 0, y1:y2, x1:x2] = 1.0

		# add slight intensity variation
		imgs[b, 0] = masks[b, 0] * 0.9 + 0.1 * torch.rand_like(masks[b, 0])

	print("Input stats:", imgs.mean().item(), masks.mean().item())

	os.makedirs("debug_smoke", exist_ok=True)

	# -----------------------------
	# 2. mask_to_box_mask
	# -----------------------------
	box_mask = mask_to_box_mask(masks)

	print("Box mask stats:", box_mask.mean().item())

	# quick visualization
	plt.figure(figsize=(6,3))
	plt.subplot(1,2,1)
	plt.title("mask")
	plt.imshow(masks[0,0].cpu(), cmap="gray")
	plt.axis("off")

	plt.subplot(1,2,2)
	plt.title("box")
	plt.imshow(box_mask[0,0].cpu(), cmap="gray")
	plt.axis("off")

	plt.savefig("debug_smoke/mask_to_box.png")
	plt.close()

	# -----------------------------
	# 3. apply TTA augmentations
	# -----------------------------
	imgs_aug, masks_aug, aug_list = apply_tta_augmentations(imgs, masks)

	B, A, C, H, W = imgs_aug.shape
	print("Aug shape:", imgs_aug.shape)

	# visualize augmentations
	fig, axes = plt.subplots(3, A, figsize=(2*A, 6))

	for j in range(A):

		# -------------------------
		# row 1: augmented images
		# -------------------------
		axes[0, j].imshow(imgs_aug[0, j, 0].cpu(), cmap="gray")
		axes[0, j].axis("off")
		axes[0, j].set_title(f"img_{j}")

		# -------------------------
		# row 2: augmented box masks
		# -------------------------
		axes[1, j].imshow(masks_aug[0, j, 0].cpu(), cmap="gray")
		axes[1, j].axis("off")
		axes[1, j].set_title(f"box_{j}")

		# -------------------------
		# row 3: reversed masks (CRITICAL)
		# -------------------------
		rev_mask = aug_list[j](
			masks_aug[0, j:j+1],  # keep batch dim
			reverse=True,
			is_mask=True
		)

		axes[2, j].imshow(rev_mask[0, 0].cpu(), cmap="gray")
		axes[2, j].axis("off")
		axes[2, j].set_title("rev_box")

	plt.savefig("debug_smoke/tta_aug_with_reverse.png")
	plt.close()

	# -----------------------------
	# 4. test augmentation invertibility
	# -----------------------------
	print("\n=== Augmentation invertibility test ===")
	for i, aug_fn in enumerate(aug_list):
		x = imgs.clone()
		y = aug_fn(x, reverse=False)
		x_rec = aug_fn(y, reverse=True)

		err = (x - x_rec).abs().mean().item()
		print(f"Aug {i}: reconstruction error = {err:.6f}")

		if err > 1e-2:
			print("WARNING: large reconstruction error!")

	# -----------------------------
	# 5. simulate model output
	# -----------------------------
	output = masks_aug.reshape(B*A, C, H, W)#torch.randn((B*A, C, H, W), device=device)

	label = imgs_aug.reshape(B*A, 1, H, W)

	# -----------------------------
	# 6. run TTA
	# -----------------------------
	out_sel, lab_sel, pseudo, var = tta(
		output,
		label,
		(B, A, C, H, W),
		aug_list
	)

	print("\nTTA outputs:")
	print("out_sel:", out_sel.shape)
	print("lab_sel:", lab_sel.shape)
	print("pseudo:", pseudo.shape)
	print("var:", var.shape)

	# -----------------------------
	# 7. visualize TTA alignment
	# -----------------------------
	fig, axes = plt.subplots(3, A, figsize=(2*A,6))

	preds = torch.sigmoid(output.reshape(B, A, C, H, W))

	for j in range(A):
		# BEFORE reverse
		axes[0,j].imshow(preds[0,j,0].cpu(), cmap="gray")
		axes[0,j].axis("off")
		axes[0,j].set_title(f"pred_{j}")

		# pseudo (same for all j)
		axes[1,j].imshow(pseudo[0,j,0].cpu(), cmap="gray")
		axes[1,j].axis("off")
		axes[1,j].set_title("pseudo")

		# variance
		axes[2,j].imshow(var[0,j,0].cpu(), cmap="hot")
		axes[2,j].axis("off")
		axes[2,j].set_title("var")

	plt.savefig("debug_smoke/tta_results.png")
	plt.close()

	# -----------------------------
	# 8. critical sanity checks
	# -----------------------------
	print("\n=== Sanity Checks ===")

	# variance range
	print("var min/max:", var.min().item(), var.max().item())

	# pseudo range
	print("pseudo min/max:", pseudo.min().item(), pseudo.max().item())

	# label consistency
	lab_mean = lab_sel.mean().item()
	print("lab mean:", lab_mean)

	if torch.isnan(pseudo).any():
		print("ERROR: NaNs in pseudo")

	if torch.isnan(var).any():
		print("ERROR: NaNs in var")

	print("\nSmoke test completed. Check debug_smoke/ for images.")
